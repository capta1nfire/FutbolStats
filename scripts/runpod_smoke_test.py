#!/usr/bin/env python3
"""
RunPod Smoke Test - Direct LLM testing without backend.

Tests prompts/payloads directly against RunPod, including:
- derived_facts generation
- claim_validator checks
- narrative schema validation

Usage:
    python scripts/runpod_smoke_test.py --payload logs/payloads/payload_70509.json
    python scripts/runpod_smoke_test.py --payload logs/payloads/payload_70509.json --sync
    python scripts/runpod_smoke_test.py --payload logs/payloads/payload_70509.json --max-tokens 2048

Environment:
    RUNPOD_API_KEY: RunPod API key (required)
    RUNPOD_ENDPOINT_ID: Endpoint ID (default: a49n0iddpgsv7r)
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

# Default endpoint
DEFAULT_ENDPOINT_ID = "a49n0iddpgsv7r"
RUNPOD_BASE_URL = "https://api.runpod.ai/v2"


def get_api_key() -> str:
    """Get RunPod API key from environment."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        logger.error("RUNPOD_API_KEY not set in environment")
        sys.exit(1)
    return key


def get_endpoint_id() -> str:
    """Get RunPod endpoint ID from environment or default."""
    return os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT_ID)


def hash_prompt(prompt: str) -> str:
    """Generate SHA256 hash of prompt for tracking."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def enrich_payload_with_derived_facts(payload: dict) -> dict:
    """
    Add derived_facts to payload if not present.

    This allows testing v1.3+ prompts with older payloads.
    """
    if payload.get("derived_facts"):
        logger.info("Payload already has derived_facts, using existing")
        return payload

    # Extract required fields
    home_goals = payload.get("home_goals", 0) or 0
    away_goals = payload.get("away_goals", 0) or 0
    home_team = payload.get("home_team", "")
    away_team = payload.get("away_team", "")
    events = payload.get("events", [])
    stats = payload.get("stats", {})

    # Build derived_facts
    derived_facts = build_derived_facts(
        home_goals=home_goals,
        away_goals=away_goals,
        home_team=home_team,
        away_team=away_team,
        events=events,
        stats=stats,
        match_status="FT",  # Assume finished for smoke test
    )

    payload["derived_facts"] = derived_facts
    logger.info(f"Generated derived_facts: winner={derived_facts.get('result', {}).get('winner')}")

    return payload


def call_runpod_sync(
    prompt: str,
    api_key: str,
    endpoint_id: str,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> dict:
    """
    Call RunPod /runsync endpoint (synchronous).

    Returns:
        Response dict with status, output, timing info.
    """
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/runsync"
    headers = {
        "authorization": api_key,  # No Bearer prefix
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        }
    }

    logger.info(f"Calling /runsync (max_tokens={max_tokens}, temp={temperature})")
    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        elapsed_ms = (time.time() - start_time) * 1000

        data = response.json()
        data["_client_elapsed_ms"] = elapsed_ms
        return data

    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "error": "Request timed out after 120s"}
    except requests.exceptions.RequestException as e:
        return {"status": "ERROR", "error": str(e)}


def call_runpod_async(
    prompt: str,
    api_key: str,
    endpoint_id: str,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    top_p: float = 0.9,
    poll_interval: float = 2.0,
    max_polls: int = 60,
) -> dict:
    """
    Call RunPod /run endpoint (async) and poll /status.

    Returns:
        Response dict with status, output, timing info.
    """
    run_url = f"{RUNPOD_BASE_URL}/{endpoint_id}/run"
    headers = {
        "authorization": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        }
    }

    logger.info(f"Calling /run (async, max_tokens={max_tokens}, temp={temperature})")
    start_time = time.time()

    # Submit job
    try:
        response = requests.post(run_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        submit_data = response.json()
        job_id = submit_data.get("id")

        if not job_id:
            return {"status": "ERROR", "error": "No job_id in response"}

        logger.info(f"Job submitted: {job_id}")

    except requests.exceptions.RequestException as e:
        return {"status": "ERROR", "error": f"Submit failed: {e}"}

    # Poll for completion
    status_url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"

    for poll_num in range(max_polls):
        try:
            time.sleep(poll_interval)
            response = requests.get(status_url, headers=headers, timeout=30)
            response.raise_for_status()
            status_data = response.json()

            status = status_data.get("status", "UNKNOWN")

            if status == "COMPLETED":
                elapsed_ms = (time.time() - start_time) * 1000
                status_data["_client_elapsed_ms"] = elapsed_ms
                logger.info(f"Job completed after {poll_num + 1} polls ({elapsed_ms:.0f}ms)")
                return status_data

            elif status in ("FAILED", "CANCELLED"):
                return status_data

            else:
                logger.debug(f"Poll {poll_num + 1}: status={status}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Poll {poll_num + 1} failed: {e}")

    return {"status": "TIMEOUT", "error": f"Max polls ({max_polls}) exceeded", "id": job_id}


def extract_llm_text(response: dict) -> Optional[str]:
    """
    Extract text from RunPod response.

    Handles different response formats from vLLM.
    """
    output = response.get("output")

    if not output:
        return None

    # Handle list of outputs (common vLLM format)
    if isinstance(output, list) and len(output) > 0:
        first_output = output[0]
        if isinstance(first_output, dict):
            # Check for choices format
            choices = first_output.get("choices", [])
            if choices and isinstance(choices, list):
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    # Token-based output
                    tokens = first_choice.get("tokens", [])
                    if tokens:
                        return "".join(tokens) if isinstance(tokens, list) else str(tokens)
                    # Text-based output
                    text = first_choice.get("text")
                    if text:
                        return text
            # Direct text field
            if first_output.get("text"):
                return first_output["text"]

    # String output
    if isinstance(output, str):
        return output

    return None


def run_smoke_test(
    payload_path: str,
    sync: bool = False,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    top_p: float = 0.9,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run smoke test against RunPod.

    Args:
        payload_path: Path to match payload JSON
        sync: Use /runsync instead of /run + polling
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        output_path: Path to save results (optional)

    Returns:
        Test results dict
    """
    api_key = get_api_key()
    endpoint_id = get_endpoint_id()

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
    logger.info("Building narrative prompt (v7)")
    prompt, home_pack, away_pack = build_narrative_prompt(payload)
    prompt_hash = hash_prompt(prompt)
    prompt_size = len(prompt)

    logger.info(f"Prompt built: {prompt_size} chars, hash={prompt_hash}")

    # Call RunPod
    if sync:
        response = call_runpod_sync(
            prompt, api_key, endpoint_id, max_tokens, temperature, top_p
        )
    else:
        response = call_runpod_async(
            prompt, api_key, endpoint_id, max_tokens, temperature, top_p
        )

    # Build results
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "match_id": match_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash,
        "prompt_size_chars": prompt_size,
        "mode": "sync" if sync else "async",
        "runpod": {
            "endpoint_id": endpoint_id,
            "job_id": response.get("id"),
            "status": response.get("status"),
            "delay_time_ms": response.get("delayTime"),
            "execution_time_ms": response.get("executionTime"),
            "client_elapsed_ms": response.get("_client_elapsed_ms"),
            "worker_id": response.get("workerId"),
        },
    }

    # Extract and parse LLM output
    raw_text = extract_llm_text(response)

    if not raw_text:
        results["validation"] = {
            "status": "error",
            "error": "No text in LLM response",
        }
        results["raw_output"] = response.get("output")
        _save_and_print_results(results, output_path)
        return results

    results["raw_output_length"] = len(raw_text)

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

    # Claim validation (includes derived_facts checks)
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

    # Token usage (if available)
    output_list = response.get("output", [])
    if isinstance(output_list, list) and len(output_list) > 0:
        usage = output_list[0].get("usage", {})
        results["tokens"] = {
            "input": usage.get("input"),
            "output": usage.get("output"),
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
    # Save to file
    if output_path:
        output_file = Path(output_path)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        match_id = results.get("match_id", "unknown")
        output_file = PROJECT_ROOT / "logs" / f"runpod_smoke_{match_id}_{ts}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize before saving (remove API key if accidentally included)
    safe_results = _sanitize_results(results)

    with open(output_file, "w") as f:
        json.dump(safe_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)

    runpod = results.get("runpod", {})
    print(f"Job ID:         {runpod.get('job_id', 'N/A')}")
    print(f"Status:         {runpod.get('status', 'N/A')}")
    print(f"Delay Time:     {runpod.get('delay_time_ms', 'N/A')} ms")
    print(f"Execution Time: {runpod.get('execution_time_ms', 'N/A')} ms")
    elapsed = runpod.get('client_elapsed_ms')
    print(f"Client Elapsed: {elapsed:.0f} ms" if elapsed else "Client Elapsed: N/A")

    tokens = results.get("tokens", {})
    if tokens:
        print(f"Tokens In/Out:  {tokens.get('input', 'N/A')} / {tokens.get('output', 'N/A')}")

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


def _sanitize_results(results: dict) -> dict:
    """Remove any sensitive data from results before saving."""
    # Deep copy to avoid modifying original
    import copy
    safe = copy.deepcopy(results)

    # Remove any fields that might contain API keys
    for key in list(safe.keys()):
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            if isinstance(safe[key], str) and len(safe[key]) > 20:
                safe[key] = "[REDACTED]"

    return safe


def main():
    parser = argparse.ArgumentParser(
        description="RunPod Smoke Test - Direct LLM testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Async mode (default)
    python scripts/runpod_smoke_test.py --payload logs/payloads/payload_70509.json

    # Sync mode (faster for single tests)
    python scripts/runpod_smoke_test.py --payload logs/payloads/payload_70509.json --sync

    # Custom parameters
    python scripts/runpod_smoke_test.py --payload logs/payloads/payload_6648.json --max-tokens 2048 --temperature 0.5
        """,
    )

    parser.add_argument(
        "--payload",
        required=True,
        help="Path to match payload JSON file",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use /runsync instead of /run + polling",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Max tokens to generate (default: 1500)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--out",
        help="Output file path (default: logs/runpod_smoke_<match_id>_<ts>.json)",
    )

    args = parser.parse_args()

    # Run test
    results = run_smoke_test(
        payload_path=args.payload,
        sync=args.sync,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=args.out,
    )

    # Exit with appropriate code
    validation = results.get("validation", {})
    if validation.get("status") == "rejected":
        sys.exit(2)
    elif validation.get("status") == "error":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
