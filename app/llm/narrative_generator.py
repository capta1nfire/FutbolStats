"""
Post-match narrative generator using RunPod LLM.

Builds compact prompts and validates JSON responses.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union

from app.config import get_settings
from app.llm.runpod_client import RunPodClient, RunPodError, RunPodJobResult
from app.llm.gemini_client import GeminiClient
from app.llm.team_aliases import (
    get_team_alias_pack,
    get_reference_rules_for_prompt,
    validate_nickname_usage,
    validate_venue_usage,
)

logger = logging.getLogger(__name__)


# Required keys in LLM JSON response (v2 schema)
REQUIRED_KEYS = {"match_id", "lang", "result", "narrative"}
# Result sub-keys
REQUIRED_RESULT_KEYS = {"ft_score", "outcome", "bet_won"}
# Narrative sub-keys
REQUIRED_NARRATIVE_KEYS = {"title", "body", "key_factors", "tone"}

# Guardrail: minimum stats required for quality narrative
REQUIRED_STATS_KEYS = {"ball_possession", "total_shots", "shots_on_goal"}


@dataclass
class NarrativeResult:
    """Result from narrative generation."""

    status: str  # ok, ok_retry, error, disabled, skipped
    narrative_json: Optional[dict] = None
    model: str = "qwen-vllm"
    delay_ms: int = 0
    exec_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    worker_id: str = ""
    error: Optional[str] = None
    error_code: Optional[str] = None  # runpod_http_error, runpod_timeout, schema_invalid, json_parse_error, gating_skipped, empty_output, unknown
    request_id: str = ""  # RunPod job ID
    attempts: int = 1  # Number of attempts (1 or 2)


def _determine_outcome(home_goals: int, away_goals: int) -> str:
    """Determine match outcome string."""
    if home_goals > away_goals:
        return "HOME"
    elif away_goals > home_goals:
        return "AWAY"
    return "DRAW"


def check_stats_gating(match_data: dict) -> tuple[bool, str]:
    """
    Guardrail: Check if match has minimum required stats for quality narrative.

    Args:
        match_data: Dict with match info including stats.

    Returns:
        Tuple of (passes_gating, reason).
    """
    stats = match_data.get("stats", {})
    home_stats = stats.get("home", {})
    away_stats = stats.get("away", {})

    # Check home stats
    home_present = {k for k in REQUIRED_STATS_KEYS if home_stats.get(k) is not None}
    home_missing = REQUIRED_STATS_KEYS - home_present

    # Check away stats
    away_present = {k for k in REQUIRED_STATS_KEYS if away_stats.get(k) is not None}
    away_missing = REQUIRED_STATS_KEYS - away_present

    if home_missing or away_missing:
        missing_desc = []
        if home_missing:
            missing_desc.append(f"home missing: {home_missing}")
        if away_missing:
            missing_desc.append(f"away missing: {away_missing}")
        return False, f"Stats gating failed: {', '.join(missing_desc)}"

    return True, "Stats gating passed"


def log_llm_evaluation(
    match_id: int,
    bet_won: Optional[bool],
    tone: Optional[str],
    tokens_in: int,
    tokens_out: int,
    exec_ms: int,
    schema_valid: bool,
    status: str,
    error: Optional[str] = None,
) -> None:
    """
    Log LLM call evaluation for monitoring.

    Logs: match_id, bet_won, tone, tokens_in/out, executionTime, schema_valid, status
    """
    log_data = {
        "match_id": match_id,
        "bet_won": bet_won,
        "tone": tone,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "exec_ms": exec_ms,
        "schema_valid": schema_valid,
        "status": status,
    }
    if error:
        log_data["error"] = error

    # Use special logger for evaluation tracking
    logger.info(f"[LLM_EVAL] {json.dumps(log_data)}")


def _determine_bet_won(prediction: dict, home_goals: int, away_goals: int) -> bool:
    """Check if prediction was correct."""
    if not prediction:
        return False
    predicted = prediction.get("predicted_result", "")
    actual = _determine_outcome(home_goals, away_goals)
    return predicted == actual


def build_narrative_prompt(match_data: dict) -> tuple[str, dict, dict]:
    """
    Build prompt v9 for post-match narrative generation.

    v9 improvements (on top of v7):
    - New ESTILO block for "human_medium" voice without losing grounding
    - Explicit rules for allowed/prohibited editorial language
    - narrative_style hints support (energy, voice, tone_mode)
    - Mandatory mention of total_shots and conflict_flag when present

    Args:
        match_data: Dict with match info (teams, score, stats, prediction, etc.)

    Returns:
        Tuple of (prompt_string, home_alias_pack, away_alias_pack)
    """
    # Extract key fields
    match_id = match_data.get("match_id", 0)
    home_team = match_data.get("home_team", "Local")
    away_team = match_data.get("away_team", "Visitante")
    home_team_id = match_data.get("home_team_id")
    away_team_id = match_data.get("away_team_id")
    league = match_data.get("league_name", "Liga")
    match_date = match_data.get("date", "")
    home_goals = match_data.get("home_goals", 0) or 0
    away_goals = match_data.get("away_goals", 0) or 0

    # Venue (stadium) - only include if available from DB
    venue = match_data.get("venue", {}) or {}
    venue_name = venue.get("name")
    venue_city = venue.get("city")

    # Get alias packs with deterministic selection
    home_pack = get_team_alias_pack(home_team, external_id=home_team_id, match_id=match_id, is_home=True)
    away_pack = get_team_alias_pack(away_team, external_id=away_team_id, match_id=match_id, is_home=False)

    # Build aliases lists for prompt (include team name + nicknames)
    home_aliases = [home_team] + home_pack.get("nicknames_allowed", []) + ["los locales"]
    away_aliases = [away_team] + away_pack.get("nicknames_allowed", []) + ["los visitantes"]
    home_aliases_json = json.dumps(home_aliases, ensure_ascii=False)
    away_aliases_json = json.dumps(away_aliases, ensure_ascii=False)

    # Slogans (if available)
    home_slogan = home_pack.get("slogan")
    away_slogan = away_pack.get("slogan")
    slogan_note = ""
    if home_slogan or away_slogan:
        slogan_parts = []
        if home_slogan:
            slogan_parts.append(f'LOCAL: "{home_slogan}"')
        if away_slogan:
            slogan_parts.append(f'VISITANTE: "{away_slogan}"')
        slogan_note = f"\n   - Slogans permitidos (opcional): {', '.join(slogan_parts)}"

    # Stats as JSON (only include non-null values)
    stats = match_data.get("stats", {})
    home_stats = stats.get("home", {})
    away_stats = stats.get("away", {})

    # Filter out None/null values for cleaner prompt
    home_stats_clean = {k: v for k, v in home_stats.items() if v is not None}
    away_stats_clean = {k: v for k, v in away_stats.items() if v is not None}

    home_stats_json = json.dumps(home_stats_clean, ensure_ascii=False) if home_stats_clean else "null"
    away_stats_json = json.dumps(away_stats_clean, ensure_ascii=False) if away_stats_clean else "null"

    # Prediction as JSON
    prediction = match_data.get("prediction", {})
    if prediction:
        # Compute bet_won if not already present
        if "correct" not in prediction:
            prediction["correct"] = _determine_bet_won(prediction, home_goals, away_goals)
        prediction_json = json.dumps(prediction, ensure_ascii=False)
    else:
        prediction_json = "null"

    # Events (max 10) as JSON
    events = match_data.get("events", [])[:10]
    events_json = json.dumps(events, ensure_ascii=False) if events else "[]"

    # Market odds as JSON
    odds = match_data.get("market_odds", {})
    if odds and odds.get("home"):
        market_odds_json = json.dumps(odds, ensure_ascii=False)
    else:
        market_odds_json = "null"

    # Venue as JSON (null if not available)
    if venue_name:
        venue_json = json.dumps({"name": venue_name, "city": venue_city}, ensure_ascii=False)
    else:
        venue_json = "null"

    # Derived facts (pre-computed verifiable facts) as JSON
    derived_facts = match_data.get("derived_facts", {})
    derived_facts_json = json.dumps(derived_facts, ensure_ascii=False) if derived_facts else "null"

    # Narrative style hints (v9)
    narrative_style = match_data.get("narrative_style", {})
    style_energy = narrative_style.get("energy", "medium")
    style_voice = narrative_style.get("voice", "humano_tecnico")
    style_tone_mode = narrative_style.get("tone_mode", "neutral")
    narrative_style_json = json.dumps(narrative_style, ensure_ascii=False) if narrative_style else "null"

    prompt = f"""You are a football analyst. Write in SPANISH (español), human but technical voice, no emojis.

RULES (v11):
1) Return ONLY valid JSON. No text before/after. Ensure valid JSON syntax.
2) Use digits: "1", "2", "54%", never words for numbers.
3) Don't repeat score in body. Focus on WHY.
4) TEAM REFS: Use official name, or nickname in quotes if in allowed list. Generic refs OK without quotes.
   - HOME: {home_team} | Allowed: {home_aliases_json}
   - AWAY: {away_team} | Allowed: {away_aliases_json}
5) FORBIDDEN: made-up nicknames, fan chants, "cuadro blanco", markdown.
6) Use ONLY DATA provided. Don't invent. match_id exact.
7) If data missing, don't mention it. Paragraphs with "\\n\\n".
8) tone: "reinforce_win" if prediction correct, "mitigate_loss" if wrong. NEVER in body text.
9) DERIVED_FACTS is primary source. Don't recalculate stats.
10) MANDATORY: mention total_shots ("X vs Y"), early goals if <15'.
11) STYLE v11: "arranque rápido" only if goal<15'. FORBIDDEN: "robo", "inmerecido", "épico". Max 2 editorial phrases.
12) LENGTH: 120-150 words. title max 8 words.
13) ANTI-LEAK: NEVER write JSON field names literally in body (conflict_flag, betting_context, etc). Express ideas in natural language instead.
14) PLAYER NAMES: Use ONLY surname (last name). Never "K. Mbappé" or "R. Asensio" - write "Mbappé", "Asensio".

FORMATO DE SALIDA (SCHEMA OBLIGATORIO):

{{
  "match_id": <int>,
  "lang": "es",
  "result": {{
    "ft_score": "{home_goals}-{away_goals}",
    "outcome": "HOME|DRAW|AWAY",
    "bet_won": true|false
  }},
  "prediction": {{
    "selection": "HOME|DRAW|AWAY",
    "confidence": <0-1>,
    "probabilities": {{"home": <0-1>, "draw": <0-1>, "away": <0-1>}}
  }},
  "market_odds": {{"home": number|null, "draw": number|null, "away": number|null}},
  "narrative": {{
    "title": "string corto (máx 8 palabras)",
    "body": "2-3 párrafos con \\n\\n (140-240 palabras)",
    "key_factors": [
      {{"label": "Stats", "evidence": "máx 120 chars con cifras", "direction": "pro-pick|anti-pick|neutral"}},
      {{"label": "Events", "evidence": "máx 120 chars con minuto", "direction": "pro-pick|anti-pick|neutral"}},
      {{"label": "Efficiency", "evidence": "máx 120 chars", "direction": "pro-pick|anti-pick|neutral"}}
    ],
    "tone": "reinforce_win|mitigate_loss",
    "responsible_note": "1 frase corta"
  }}
}}

DATOS (usa SOLO esto):
match_id: {match_id}
home_team: {home_team}
away_team: {away_team}
team_aliases.home: {home_aliases_json}
team_aliases.away: {away_aliases_json}
league_name: {league}
date: {match_date}
final_score: {home_goals}-{away_goals}
venue: {venue_json}

stats.home: {home_stats_json}
stats.away: {away_stats_json}

prediction: {prediction_json}

events: {events_json}

market_odds: {market_odds_json}

derived_facts: {derived_facts_json}

narrative_style: {narrative_style_json}

RECUERDA: JSON válido (verifica sintaxis), sin marcador en body, solo aliases permitidos (entre comillas), key_factors cortos y distintos del body. USA derived_facts como fuente primaria. MENCIONA total_shots. NUNCA escribas nombres de campos JSON en el texto. Máx 2 frases editoriales."""

    return prompt, home_pack, away_pack


def parse_json_response(text: str) -> Optional[dict]:
    """
    Parse JSON from LLM response.

    Handles common issues like markdown code blocks.

    Args:
        text: Raw text from LLM.

    Returns:
        Parsed dict or None if invalid.
    """
    # Clean up common issues
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try to find JSON object
    start = text.find("{")
    if start < 0:
        logger.warning("No JSON object found in response")
        return None

    # Find the matching closing brace (handle nested objects)
    depth = 0
    end = start
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if depth != 0:
        # Fallback to rfind if depth doesn't balance
        end = text.rfind("}") + 1

    json_text = text[start:end]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")

        # Try to fix unescaped quotes inside string values
        # This is a common LLM issue where quotes like "team name" appear unescaped
        fixed_text = _fix_unescaped_quotes(json_text)
        if fixed_text != json_text:
            try:
                result = json.loads(fixed_text)
                logger.info("JSON parsed successfully after fixing unescaped quotes")
                return result
            except json.JSONDecodeError as e2:
                logger.warning(f"JSON still invalid after quote fix: {e2}")

        # Log full text length and content for debugging
        logger.warning(f"Raw text length: {len(json_text)}")
        logger.warning(f"Raw text (first 500 chars): {json_text[:500]}")
        return None


def _fix_unescaped_quotes(json_text: str) -> str:
    """
    Fix unescaped double quotes and literal newlines inside JSON string values.

    LLMs sometimes generate JSON with:
    1. Unescaped quotes like: "body": "The "team" won"
    2. Literal newlines inside strings (should be \\n)

    This function processes the text to fix these issues.
    """
    import re

    # First, replace curly/smart quotes with straight quotes
    text = json_text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Fix literal newlines inside string values
    # Pattern: find "key": "...value with newline..." for specific keys
    for key in ['body', 'title', 'evidence', 'responsible_note']:
        # Match the key and capture everything until we find the closing pattern
        # The closing pattern is: ", followed by newline and next key OR end of object
        pattern = rf'("{key}":\s*")(.*?)("\s*,?\s*\n\s*"(?:key_factors|tone|title|body|label|evidence|direction|responsible_note)"|"\s*\n\s*\}})'

        def fix_string_value(match):
            prefix = match.group(1)
            content = match.group(2)
            suffix = match.group(3)

            # Fix literal newlines - replace with \\n
            # But don't replace already escaped \\n
            fixed = content.replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')

            # Fix unescaped quotes
            result = ""
            i = 0
            while i < len(fixed):
                if fixed[i] == '"':
                    if i > 0 and fixed[i-1] == '\\':
                        result += '"'
                    else:
                        result += '\\"'
                else:
                    result += fixed[i]
                i += 1

            return prefix + result + suffix

        text = re.sub(pattern, fix_string_value, text, flags=re.DOTALL)

    # Also try the simpler line-by-line approach for single-line values
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        for key in ['body', 'title', 'evidence', 'responsible_note']:
            prefix = f'"{key}": "'
            if prefix in line:
                start_idx = line.index(prefix) + len(prefix)
                rest = line[start_idx:]
                end_content = rest.rstrip()

                if end_content.endswith('",'):
                    value_content = end_content[:-2]
                    suffix = '",'
                elif end_content.endswith('"'):
                    value_content = end_content[:-1]
                    suffix = '"'
                else:
                    continue

                # Escape unescaped quotes
                fixed_content = ""
                i = 0
                while i < len(value_content):
                    if value_content[i] == '"':
                        if i > 0 and value_content[i-1] == '\\':
                            fixed_content += '"'
                        else:
                            fixed_content += '\\"'
                    else:
                        fixed_content += value_content[i]
                    i += 1

                line = line[:start_idx - len(prefix)] + prefix + fixed_content + suffix
                break

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def _normalize_narrative_object(narrative: dict) -> tuple[dict, Optional[dict]]:
    """
    Normalize malformed narrative objects from LLM (Schema v3.2).

    Handles cases where LLM generates extra keys (e.g., newlines as keys)
    by concatenating all string values into body.

    Valid narrative keys (v3.2):
    - title: string (required)
    - body: string (required)
    - tone: string ("reinforce_win", "mitigate_loss", "neutral")
    - keyFactors: array of {label, evidence, direction}
    - responsibleNote: string

    Args:
        narrative: The narrative dict from LLM response.

    Returns:
        Tuple of (normalized_dict, normalization_warning or None)
    """
    # Valid narrative keys per schema v3.2
    VALID_NARRATIVE_KEYS = {"title", "body", "tone", "keyFactors", "responsibleNote", "key_factors", "responsible_note"}

    # Extract known fields
    title = narrative.get("title", "")
    body = narrative.get("body", "")
    tone = narrative.get("tone")
    # Support both camelCase (LLM) and snake_case variants
    key_factors = narrative.get("keyFactors") or narrative.get("key_factors")
    responsible_note = narrative.get("responsibleNote") or narrative.get("responsible_note")

    # Collect extra string values (malformed keys like "\n\nMore text...")
    extra_keys = []
    extra_parts = []
    for key, value in narrative.items():
        if key not in VALID_NARRATIVE_KEYS and isinstance(value, str) and value.strip():
            extra_keys.append(key[:50])  # Truncate key for logging
            extra_parts.append(value.strip())

    normalization_warning = None

    # If there are extra parts, append them to body
    if extra_parts:
        logger.info(f"Normalizing narrative: found {len(extra_parts)} extra string keys, concatenating to body")
        if body:
            body = body + "\n\n" + "\n\n".join(extra_parts)
        else:
            body = "\n\n".join(extra_parts)

        # Create warning for tracking/monitoring
        normalization_warning = {
            "type": "narrative_normalized",
            "severity": "warning",
            "extra_keys_count": len(extra_keys),
            "extra_keys_sample": extra_keys[:3],  # First 3 keys for debugging
        }

    # Build normalized narrative preserving all valid fields
    normalized = {"title": title, "body": body}
    if tone:
        normalized["tone"] = tone
    if key_factors:
        normalized["keyFactors"] = key_factors
    if responsible_note:
        normalized["responsibleNote"] = responsible_note

    return normalized, normalization_warning


def validate_narrative_json(data: dict, match_id: int) -> bool:
    """
    Validate that JSON has required keys and correct structure (v2 schema).

    Args:
        data: Parsed JSON dict.
        match_id: Expected match ID.

    Returns:
        True if valid.
    """
    if not isinstance(data, dict):
        return False

    # Check top-level required keys
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        logger.warning(f"Missing required top-level keys: {missing}")
        return False

    # Validate result sub-object
    result = data.get("result", {})
    if isinstance(result, dict):
        missing_result = REQUIRED_RESULT_KEYS - set(result.keys())
        if missing_result:
            logger.warning(f"Missing result keys: {missing_result}")
            # Don't fail, just warn - LLM might use slightly different structure
    elif isinstance(result, str):
        # Legacy format (v1): result was just a string like "2-1"
        logger.info("Result is string (v1 format), accepting")

    # Validate and normalize narrative sub-object
    narrative = data.get("narrative", {})
    if isinstance(narrative, dict):
        # Normalize malformed narrative (extra keys -> concatenate to body)
        normalized, normalization_warning = _normalize_narrative_object(narrative)
        data["narrative"] = normalized  # Update in place

        # Store normalization warning as metadata for tracking
        if normalization_warning:
            data["_normalization_warning"] = normalization_warning

        if not normalized.get("body"):
            logger.warning("Narrative body is empty after normalization")
            return False

        missing_narrative = REQUIRED_NARRATIVE_KEYS - set(normalized.keys())
        if missing_narrative:
            logger.warning(f"Missing narrative keys: {missing_narrative}")
            # Don't fail, just warn
    elif isinstance(narrative, str):
        # Legacy format (v1): narrative was just a string
        logger.info("Narrative is string (v1 format), accepting")

    # Validate match_id matches
    if data.get("match_id") != match_id:
        logger.warning(f"match_id mismatch: expected {match_id}, got {data.get('match_id')}")
        # Don't fail on this, just warn

    return True


class NarrativeGenerator:
    """Generates post-match narratives using configured LLM provider (Gemini or RunPod)."""

    def __init__(self):
        self.settings = get_settings()
        self.provider_name = self.settings.NARRATIVE_PROVIDER.lower()

        # Initialize appropriate LLM client based on NARRATIVE_PROVIDER
        if self.provider_name == "gemini":
            self.client = GeminiClient()
            logger.info("NarrativeGenerator using Gemini provider")
        else:
            self.client = RunPodClient()
            logger.info("NarrativeGenerator using RunPod provider")

        self.enabled = self.settings.NARRATIVE_LLM_ENABLED

    async def close(self):
        """Close resources."""
        await self.client.close()

    async def generate(self, match_data: dict) -> NarrativeResult:
        """
        Generate narrative for a match.

        Args:
            match_data: Dict with match info.

        Returns:
            NarrativeResult with status and data.
        """
        if not self.enabled:
            return NarrativeResult(status="disabled", error_code="disabled")

        # Check API key based on provider
        if self.provider_name == "gemini":
            if not self.settings.GEMINI_API_KEY:
                return NarrativeResult(status="error", error="GEMINI_API_KEY not configured", error_code="gemini_auth")
        else:
            if not self.settings.RUNPOD_API_KEY:
                return NarrativeResult(status="error", error="RUNPOD_API_KEY not configured", error_code="runpod_auth")

        match_id = match_data.get("match_id", 0)

        # Guardrail 1: Stats gating
        passes_gating, gating_reason = check_stats_gating(match_data)
        if not passes_gating:
            logger.info(f"[LLM_SKIP] match_id={match_id}: {gating_reason}")
            return NarrativeResult(status="skipped", error=gating_reason, error_code="gating_skipped")

        # Extract bet_won for logging
        prediction = match_data.get("prediction", {})
        home_goals = match_data.get("home_goals", 0) or 0
        away_goals = match_data.get("away_goals", 0) or 0
        bet_won = _determine_bet_won(prediction, home_goals, away_goals) if prediction else None

        try:
            # First attempt
            prompt, home_pack, away_pack = build_narrative_prompt(match_data)
            result = await self.client.generate(prompt)

            # Debug: log token usage and raw output
            logger.info(f"LLM response: tokens_in={result.tokens_in}, tokens_out={result.tokens_out}, text_len={len(result.text)}")
            logger.info(f"LLM raw_output keys: {list(result.raw_output.keys())}")
            if "output" in result.raw_output:
                logger.info(f"LLM output structure: {result.raw_output['output'][:1] if result.raw_output['output'] else 'empty'}")

            parsed = parse_json_response(result.text)
            schema_valid = parsed is not None and validate_narrative_json(parsed, match_id)

            # Validate nickname usage if schema is valid
            if schema_valid and parsed:
                narrative_body = ""
                narrative_obj = parsed.get("narrative", {})
                if isinstance(narrative_obj, dict):
                    narrative_body = narrative_obj.get("body", "")
                elif isinstance(narrative_obj, str):
                    narrative_body = narrative_obj

                nickname_errors = validate_nickname_usage(narrative_body, home_pack, away_pack)
                if nickname_errors:
                    logger.warning(f"Nickname validation errors for match {match_id}: {nickname_errors}")
                    # Don't fail, just log - LLM might use valid aliases we don't detect

                # Validate venue usage (detect deduced stadiums when payload had null)
                venue_data = match_data.get("venue", {}) or {}
                venue_name = venue_data.get("name")
                venue_errors = validate_venue_usage(narrative_body, venue_name)
                if venue_errors:
                    logger.warning(f"Venue validation errors for match {match_id}: {venue_errors}")
                    # Don't fail, just log - this is for monitoring deduction issues

                tone = narrative_obj.get("tone") if isinstance(narrative_obj, dict) else None
                log_llm_evaluation(
                    match_id=match_id,
                    bet_won=bet_won,
                    tone=tone,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    exec_ms=result.exec_ms,
                    schema_valid=True,
                    status="ok",
                )
                return NarrativeResult(
                    status="ok",
                    narrative_json=parsed,
                    delay_ms=result.delay_ms,
                    exec_ms=result.exec_ms,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    worker_id=result.worker_id,
                    request_id=result.job_id,
                    attempts=1,
                )

            # Retry with stricter prompt
            logger.warning(f"First attempt failed for match {match_id}, retrying with strict prompt")
            strict_prompt = f"""STRICT JSON ONLY. No explanations.

{prompt}

IMPORTANTE: Responde ÚNICAMENTE con JSON válido. Nada más."""

            # Temporarily reduce tokens and temperature for retry
            self.client.max_tokens = 500
            self.client.temperature = 0.2

            result2 = await self.client.generate(strict_prompt)

            # Restore settings
            self.client.max_tokens = self.settings.NARRATIVE_LLM_MAX_TOKENS
            self.client.temperature = self.settings.NARRATIVE_LLM_TEMPERATURE

            parsed2 = parse_json_response(result2.text)
            schema_valid2 = parsed2 is not None and validate_narrative_json(parsed2, match_id)

            if schema_valid2 and parsed2:
                tone2 = parsed2.get("narrative", {}).get("tone") if isinstance(parsed2.get("narrative"), dict) else None
                log_llm_evaluation(
                    match_id=match_id,
                    bet_won=bet_won,
                    tone=tone2,
                    tokens_in=result2.tokens_in,
                    tokens_out=result2.tokens_out,
                    exec_ms=result2.exec_ms,
                    schema_valid=True,
                    status="ok_retry",
                )
                return NarrativeResult(
                    status="ok",
                    narrative_json=parsed2,
                    delay_ms=result2.delay_ms,
                    exec_ms=result2.exec_ms,
                    tokens_in=result2.tokens_in,
                    tokens_out=result2.tokens_out,
                    worker_id=result2.worker_id,
                    request_id=result2.job_id,
                    attempts=2,
                )

            # Both attempts failed
            error_msg = "JSON validation failed after retry"
            logger.error(f"Narrative generation failed for match {match_id}: {error_msg}")
            log_llm_evaluation(
                match_id=match_id,
                bet_won=bet_won,
                tone=None,
                tokens_in=result2.tokens_in,
                tokens_out=result2.tokens_out,
                exec_ms=result2.exec_ms,
                schema_valid=False,
                status="error",
                error=error_msg,
            )
            return NarrativeResult(
                status="error",
                error=error_msg,
                error_code="schema_invalid",
                delay_ms=result2.delay_ms,
                exec_ms=result2.exec_ms,
                tokens_in=result2.tokens_in,
                tokens_out=result2.tokens_out,
                worker_id=result2.worker_id,
                request_id=result2.job_id,
                attempts=2,
            )

        except RunPodError as e:
            error_str = str(e)
            # Classify RunPod errors
            if "timed out" in error_str.lower():
                error_code = "runpod_timeout"
            elif "401" in error_str or "403" in error_str or "auth" in error_str.lower():
                error_code = "runpod_auth"
            elif "Missing output" in error_str or "empty" in error_str.lower():
                error_code = "empty_output"
            else:
                error_code = "runpod_http_error"

            logger.error(f"RunPod error for match {match_id}: {e}")
            log_llm_evaluation(
                match_id=match_id,
                bet_won=bet_won,
                tone=None,
                tokens_in=0,
                tokens_out=0,
                exec_ms=0,
                schema_valid=False,
                status="runpod_error",
                error=error_str,
            )
            return NarrativeResult(status="error", error=error_str[:500], error_code=error_code)
        except Exception as e:
            error_str = str(e)
            logger.error(f"Unexpected error for match {match_id}: {e}")
            log_llm_evaluation(
                match_id=match_id,
                bet_won=bet_won,
                tone=None,
                tokens_in=0,
                tokens_out=0,
                exec_ms=0,
                schema_valid=False,
                status="unexpected_error",
                error=error_str,
            )
            return NarrativeResult(status="error", error=error_str[:500], error_code="unknown")
