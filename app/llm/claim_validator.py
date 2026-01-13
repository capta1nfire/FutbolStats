"""
Claim Validator: Anti-hallucination guardrails for LLM narratives.

Validates that claims in the narrative are supported by the match data.
Prevents Qwen from inventing red cards, penalties, or other events not in the data.
"""

import hashlib
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Current prompt version - bump when changing prompt template
PROMPT_VERSION = "v1.0"

# Claims that require evidence in events
RED_CARD_PATTERNS = [
    r"superioridad\s+numérica",
    r"inferioridad\s+numérica",
    r"con\s+uno\s+menos",
    r"con\s+un\s+hombre\s+menos",
    r"jugó\s+con\s+diez",
    r"quedó\s+con\s+diez",
    r"expulsión",
    r"expulsado",
    r"tarjeta\s+roja",
    r"roja\s+directa",
    r"segunda\s+amarilla",
    r"doble\s+amarilla",
]

PENALTY_PATTERNS = [
    r"penal(?:ti)?",
    r"pena\s+máxima",
    r"desde\s+los\s+once\s+metros",
    r"desde\s+el\s+punto\s+de\s+penal",
    r"tiro\s+desde\s+los\s+doce\s+pasos",
]

# Goal minute patterns - matches "gol al minuto X", "anotó al X'", etc.
GOAL_MINUTE_PATTERN = r"(?:gol|anotó|marcó|convirtió).*?(?:al\s+)?(?:minuto\s+)?(\d{1,3})['\s]"


def canonicalize_json(data: dict) -> str:
    """Canonicalize JSON for consistent hashing."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def compute_payload_hash(data: dict) -> str:
    """Compute SHA256 hash of canonicalized JSON."""
    canonical = canonicalize_json(data)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def sanitize_payload_for_storage(match_data: dict) -> dict:
    """
    Create a sanitized copy of match_data for DB storage.

    Removes potentially large or sensitive fields while keeping
    what's needed for debugging claims.
    """
    # Fields to keep for traceability
    keep_fields = [
        "match_id",
        "league_id",
        "league_name",
        "date",
        "kickoff_time",
        "home_team",
        "away_team",
        "score",
        "venue",
        "prediction",
        "market_odds",
        "events",  # Critical for claim validation
        "home_alias_pack",
        "away_alias_pack",
    ]

    sanitized = {}
    for field in keep_fields:
        if field in match_data:
            value = match_data[field]
            # Truncate events list if too long
            if field == "events" and isinstance(value, list) and len(value) > 15:
                sanitized[field] = value[:15]
                sanitized["events_truncated"] = True
            else:
                sanitized[field] = value

    # Include stats summary (not full stats)
    if "stats" in match_data and match_data["stats"]:
        stats = match_data["stats"]
        sanitized["stats_summary"] = {
            "home_shots": stats.get("home", {}).get("Shots on Goal"),
            "away_shots": stats.get("away", {}).get("Shots on Goal"),
            "home_possession": stats.get("home", {}).get("Ball Possession"),
            "away_possession": stats.get("away", {}).get("Ball Possession"),
            "home_xg": stats.get("home", {}).get("expected_goals"),
            "away_xg": stats.get("away", {}).get("expected_goals"),
        }

    return sanitized


def _has_red_card_evidence(events: list) -> bool:
    """Check if events contain red card evidence."""
    if not events:
        return False

    for event in events:
        event_type = event.get("type", "").lower()
        detail = event.get("detail", "").lower()

        # Direct red card
        if event_type == "card" and "red" in detail:
            return True
        # Second yellow (shown as red)
        if "second yellow" in detail or "segunda amarilla" in detail.lower():
            return True

    return False


def _has_penalty_evidence(events: list) -> bool:
    """Check if events contain penalty evidence."""
    if not events:
        return False

    for event in events:
        detail = event.get("detail", "").lower()
        event_type = event.get("type", "").lower()

        if "penalty" in detail or "penal" in detail:
            return True
        if event_type == "goal" and "penalty" in detail:
            return True

    return False


def _get_goal_minutes(events: list) -> set[int]:
    """Get set of minutes when goals were scored."""
    minutes = set()
    if not events:
        return minutes

    for event in events:
        if event.get("type", "").lower() == "goal":
            minute = event.get("minute")
            if minute is not None:
                minutes.add(int(minute))

    return minutes


def validate_narrative_claims(
    narrative_text: str,
    match_data: dict,
    strict: bool = True
) -> list[dict]:
    """
    Validate that claims in the narrative are supported by match data.

    Args:
        narrative_text: The generated narrative text
        match_data: The payload sent to the LLM (contains events, stats, etc.)
        strict: If True, unsupported claims are errors. If False, warnings only.

    Returns:
        List of validation errors/warnings:
        [{"type": "unsupported_claim", "claim": "red_card", "pattern": "...", "severity": "error"}]
    """
    errors = []
    narrative_lower = narrative_text.lower()
    events = match_data.get("events", [])

    # 1. Red card claims
    has_red = _has_red_card_evidence(events)
    for pattern in RED_CARD_PATTERNS:
        if re.search(pattern, narrative_lower):
            if not has_red:
                errors.append({
                    "type": "unsupported_claim",
                    "claim": "red_card",
                    "pattern": pattern,
                    "matched_text": re.search(pattern, narrative_lower).group(0),
                    "severity": "error" if strict else "warning",
                    "evidence_found": False,
                })
                logger.warning(
                    f"[CLAIM_VALIDATOR] Red card claim '{pattern}' without evidence in events"
                )
            break  # One error per category is enough

    # 2. Penalty claims
    has_penalty = _has_penalty_evidence(events)
    for pattern in PENALTY_PATTERNS:
        if re.search(pattern, narrative_lower):
            if not has_penalty:
                errors.append({
                    "type": "unsupported_claim",
                    "claim": "penalty",
                    "pattern": pattern,
                    "matched_text": re.search(pattern, narrative_lower).group(0),
                    "severity": "error" if strict else "warning",
                    "evidence_found": False,
                })
                logger.warning(
                    f"[CLAIM_VALIDATOR] Penalty claim '{pattern}' without evidence in events"
                )
            break

    # 3. Goal minute claims (optional - less strict)
    # Only validate if narrative mentions specific minutes
    goal_minutes = _get_goal_minutes(events)
    minute_matches = re.findall(GOAL_MINUTE_PATTERN, narrative_lower)
    for minute_str in minute_matches:
        try:
            claimed_minute = int(minute_str)
            # Allow ±2 minute tolerance
            if not any(abs(claimed_minute - actual) <= 2 for actual in goal_minutes):
                errors.append({
                    "type": "unsupported_claim",
                    "claim": "goal_minute",
                    "claimed_minute": claimed_minute,
                    "actual_goal_minutes": list(goal_minutes),
                    "severity": "warning",  # Less strict for minute mismatches
                    "evidence_found": False,
                })
                logger.warning(
                    f"[CLAIM_VALIDATOR] Goal at minute {claimed_minute} claimed but "
                    f"actual goals at {goal_minutes}"
                )
        except ValueError:
            pass

    return errors


def should_reject_narrative(validation_errors: list[dict]) -> bool:
    """
    Determine if narrative should be rejected based on validation errors.

    Returns True if there are any severity=error claims.
    """
    return any(e.get("severity") == "error" for e in validation_errors)


def get_rejection_reason(validation_errors: list[dict]) -> str:
    """Get human-readable rejection reason for error_detail field."""
    error_claims = [e for e in validation_errors if e.get("severity") == "error"]
    if not error_claims:
        return ""

    claims = [e.get("claim", "unknown") for e in error_claims]
    return f"Unsupported claims: {', '.join(claims)}"
