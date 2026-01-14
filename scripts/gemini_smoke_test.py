#!/usr/bin/env python3
"""
Gemini Smoke Test - Direct LLM testing with Google Gemini API.

Tests prompts/payloads directly against Gemini 2.0 Flash, including:
- derived_facts generation
- claim_validator checks
- narrative schema validation

Usage:
    python scripts/gemini_smoke_test.py --payload logs/payloads/payload_puebla_mazatlan.json
    python scripts/gemini_smoke_test.py --payload logs/payloads/payload_puebla_mazatlan.json --model gemini-2.0-flash-lite

Environment:
    GEMINI_API_KEY: Google Gemini API key (required)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from app.llm.narrative_generator import (
    build_narrative_prompt,
    parse_json_response,
    validate_narrative_json,
)
from app.llm.claim_validator import (
    validate_narrative_claims,
    sanitize_narrative_body,
    should_reject_narrative,
    get_rejection_reason,
    PROMPT_VERSION,
)
from app.llm.derived_facts import build_derived_facts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Gemini API endpoints
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"


def get_api_key() -> str:
    """Get Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.error("GEMINI_API_KEY not set in environment")
        sys.exit(1)
    return key


def hash_prompt(prompt: str) -> str:
    """Generate SHA256 hash of prompt for tracking."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def enrich_payload_with_derived_facts(payload: dict) -> dict:
    """
    Add derived_facts to payload if not present.
    """
    if payload.get("derived_facts"):
        logger.info("Payload already has derived_facts, using existing")
        return payload

    home_goals = payload.get("home_goals", 0) or 0
    away_goals = payload.get("away_goals", 0) or 0
    home_team = payload.get("home_team", "")
    away_team = payload.get("away_team", "")
    events = payload.get("events", [])
    stats = payload.get("stats", {})

    # Build market_odds from payload
    market_odds = None
    if payload.get("odds_home") and payload.get("odds_draw") and payload.get("odds_away"):
        market_odds = {
            "home": payload["odds_home"],
            "draw": payload["odds_draw"],
            "away": payload["odds_away"],
        }

    # Build model_probs from prediction
    model_probs = None
    prediction = payload.get("prediction", {})
    if prediction:
        model_probs = {
            "home": prediction.get("home_win_prob"),
            "draw": prediction.get("draw_prob"),
            "away": prediction.get("away_win_prob"),
        }

    # Extract value_bet info
    value_bet = payload.get("value_bet")
    if value_bet:
        value_bet["is_value_bet"] = True
        value_bet["outcome"] = value_bet.get("selection")

    # Determine prediction correctness
    prediction_correct = None
    if prediction.get("predicted_result") and payload.get("actual_result"):
        prediction_correct = prediction["predicted_result"].upper() == payload["actual_result"].upper()

    derived_facts = build_derived_facts(
        home_goals=home_goals,
        away_goals=away_goals,
        home_team=home_team,
        away_team=away_team,
        events=events,
        stats=stats,
        match_status=payload.get("match_status", "FT"),
        market_odds=market_odds,
        model_probs=model_probs,
        value_bet=value_bet,
        prediction_correct=prediction_correct,
    )

    payload["derived_facts"] = derived_facts
    logger.info(f"Generated derived_facts: winner={derived_facts.get('result', {}).get('winner')}")

    return payload


def call_gemini(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 450,
    temperature: float = 0.28,
    top_p: float = 0.9,
) -> dict:
    """
    Call Gemini API generateContent endpoint.

    Returns:
        Response dict with status, output, timing info.
    """
    url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }
    }

    logger.info(f"Calling Gemini {model} (max_tokens={max_tokens}, temp={temperature})")
    start_time = time.time()

    try:
        response = requests.post(url, json=payload, timeout=60)
        elapsed_ms = (time.time() - start_time) * 1000

        if response.status_code != 200:
            return {
                "status": "ERROR",
                "error": f"HTTP {response.status_code}: {response.text[:500]}",
                "_client_elapsed_ms": elapsed_ms,
            }

        data = response.json()
        data["_client_elapsed_ms"] = elapsed_ms
        data["status"] = "COMPLETED"
        return data

    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "error": "Request timed out after 60s"}
    except requests.exceptions.RequestException as e:
        return {"status": "ERROR", "error": str(e)}


def extract_gemini_text(response: dict) -> Optional[str]:
    """
    Extract text from Gemini response.
    """
    candidates = response.get("candidates", [])
    if not candidates:
        return None

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    if not parts:
        return None

    return parts[0].get("text")


def extract_usage(response: dict) -> dict:
    """Extract token usage from Gemini response."""
    usage_metadata = response.get("usageMetadata", {})
    return {
        "input": usage_metadata.get("promptTokenCount"),
        "output": usage_metadata.get("candidatesTokenCount"),
        "total": usage_metadata.get("totalTokenCount"),
    }


def run_smoke_test(
    payload_path: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 450,
    temperature: float = 0.28,
    top_p: float = 0.9,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run smoke test against Gemini.
    """
    api_key = get_api_key()

    # Load payload
    logger.info(f"Loading payload from {payload_path}")
    try:
        with open(payload_path) as f:
            payload = json.load(f)
    except Exception as e:
        return {"status": "error", "error": f"Failed to load payload: {e}"}

    match_id = payload.get("match_id", 0)
    logger.info(f"Match ID: {match_id}")

    # Enrich with derived_facts if needed
    payload = enrich_payload_with_derived_facts(payload)

    # Build prompt
    logger.info("Building narrative prompt")
    prompt, home_pack, away_pack = build_narrative_prompt(payload)
    prompt_hash = hash_prompt(prompt)
    prompt_size = len(prompt)

    logger.info(f"Prompt built: {prompt_size} chars, hash={prompt_hash}")

    # Call Gemini
    response = call_gemini(
        prompt, api_key, model, max_tokens, temperature, top_p
    )

    # Build results
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "match_id": match_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash,
        "prompt_size_chars": prompt_size,
        "model": model,
        "gemini": {
            "status": response.get("status"),
            "client_elapsed_ms": response.get("_client_elapsed_ms"),
            "model_version": response.get("modelVersion"),
        },
    }

    # Check for error
    if response.get("status") != "COMPLETED":
        results["validation"] = {
            "status": "error",
            "error": response.get("error", "Unknown error"),
        }
        _save_and_print_results(results, output_path)
        return results

    # Extract and parse LLM output
    raw_text = extract_gemini_text(response)

    if not raw_text:
        results["validation"] = {
            "status": "error",
            "error": "No text in Gemini response",
        }
        results["raw_response"] = response
        _save_and_print_results(results, output_path)
        return results

    results["raw_output_length"] = len(raw_text)

    # Token usage
    usage = extract_usage(response)
    results["tokens"] = usage

    # Calculate cost
    input_cost = (usage.get("input", 0) or 0) * 0.10 / 1_000_000
    output_cost = (usage.get("output", 0) or 0) * 0.40 / 1_000_000
    results["cost_usd"] = {
        "input": round(input_cost, 8),
        "output": round(output_cost, 8),
        "total": round(input_cost + output_cost, 8),
    }

    # Parse JSON
    parsed = parse_json_response(raw_text)

    if not parsed:
        results["validation"] = {
            "status": "error",
            "error": "Failed to parse JSON from LLM output",
        }
        results["raw_output_preview"] = raw_text[:1000]
        _save_and_print_results(results, output_path)
        return results

    # Validate schema
    schema_valid = validate_narrative_json(parsed, match_id)

    # Extract narrative for claim validation
    narrative = parsed.get("narrative", {})
    narrative_body = narrative.get("body", "")
    narrative_title = narrative.get("title", "")

    # Sanitize control tokens
    sanitized_body, sanitize_warnings = sanitize_narrative_body(narrative_body)

    # Claim validation
    claim_errors = validate_narrative_claims(
        narrative_text=sanitized_body,
        match_data=payload,
        strict=True,
    )

    # Determine overall validation status
    should_reject = should_reject_narrative(claim_errors)
    rejection_reason = get_rejection_reason(claim_errors) if should_reject else None

    results["validation"] = {
        "status": "rejected" if should_reject else ("warning" if claim_errors else "ok"),
        "schema_valid": schema_valid,
        "claim_errors": claim_errors,
        "sanitize_warnings": sanitize_warnings,
        "rejection_reason": rejection_reason,
    }

    # Narrative preview
    results["narrative_preview"] = {
        "title": narrative_title,
        "body": sanitized_body,
        "body_preview": sanitized_body[:300] + "..." if len(sanitized_body) > 300 else sanitized_body,
        "word_count": len(sanitized_body.split()),
        "tone": narrative.get("tone"),
    }

    # Include derived_facts used
    results["derived_facts_used"] = payload.get("derived_facts")

    _save_and_print_results(results, output_path)
    return results


def _save_and_print_results(results: dict, output_path: Optional[str]):
    """Save results to file and print summary."""
    if output_path:
        output_file = Path(output_path)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        match_id = results.get("match_id", "unknown")
        output_file = PROJECT_ROOT / "logs" / f"gemini_smoke_{match_id}_{ts}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("GEMINI SMOKE TEST RESULTS")
    print("=" * 60)

    gemini = results.get("gemini", {})
    print(f"Model:          {results.get('model', 'N/A')}")
    print(f"Status:         {gemini.get('status', 'N/A')}")
    elapsed = gemini.get('client_elapsed_ms')
    print(f"Latency:        {elapsed:.0f} ms" if elapsed else "Latency: N/A")

    tokens = results.get("tokens", {})
    if tokens:
        print(f"Tokens In/Out:  {tokens.get('input', 'N/A')} / {tokens.get('output', 'N/A')}")

    cost = results.get("cost_usd", {})
    if cost:
        print(f"Cost:           ${cost.get('total', 0):.6f} USD")

    validation = results.get("validation", {})
    val_status = validation.get("status", "unknown")
    print(f"\nValidation:     {val_status.upper()}")

    if validation.get("rejection_reason"):
        print(f"Rejection:      {validation['rejection_reason']}")

    errors = validation.get("claim_errors", [])
    if errors:
        print(f"Claim Issues:   {len(errors)}")
        for err in errors[:3]:
            print(f"  - {err.get('type')}: {err.get('claim')} ({err.get('severity')})")

    warnings = validation.get("sanitize_warnings", [])
    if warnings:
        print(f"Sanitized:      {len(warnings)} control tokens stripped")

    preview = results.get("narrative_preview", {})
    if preview:
        print(f"\nTitle:          {preview.get('title', 'N/A')}")
        print(f"Word Count:     {preview.get('word_count', 'N/A')}")
        print(f"Tone:           {preview.get('tone', 'N/A')}")
        print(f"\nBody Preview:")
        print(f"  {preview.get('body_preview', 'N/A')}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Smoke Test - Direct LLM testing with Google Gemini",
    )

    parser.add_argument(
        "--payload",
        required=True,
        help="Path to match payload JSON file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=450,
        help="Max tokens to generate (default: 450)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.28,
        help="Sampling temperature (default: 0.28)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--out",
        help="Output file path",
    )

    args = parser.parse_args()

    results = run_smoke_test(
        payload_path=args.payload,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=args.out,
    )

    validation = results.get("validation", {})
    if validation.get("status") == "rejected":
        sys.exit(2)
    elif validation.get("status") == "error":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
