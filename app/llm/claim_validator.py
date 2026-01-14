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
PROMPT_VERSION = "v1.5"

# Control tokens that should NEVER appear in narrative body
# These are internal prompt instructions that LLM sometimes echoes
CONTROL_TOKENS = [
    "mitigate_loss",
    "mitigate_win",
    "reinforce_win",
    "reinforce_loss",
    "analyze_match",
    "pro-pick",
    "anti-pick",
    # JSON field names that should never appear literally
    "conflict_flag",
    "betting_context",
    "pred_ok_value_lost",
    "pred_fail_value_won",
    "value_bet_present",
    "derived_facts",
]

# Patterns for detecting team attribution in narrative (Spanish)
LOCAL_PATTERNS = [
    r"\blocal(?:es)?\b",
    r"\bel\s+equipo\s+local\b",
    r"\blos\s+locales\b",
    r"\bdel\s+local\b",
    r"\banfitrión(?:es)?\b",
]

VISITOR_PATTERNS = [
    r"\bvisitante(?:s)?\b",
    r"\bel\s+equipo\s+visitante\b",
    r"\blos\s+visitantes\b",
    r"\bdel\s+visitante\b",
    r"\bforáneo(?:s)?\b",
]

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
# More restrictive: goal keyword must be within 40 non-digit chars of minute
# Using [^0-9] to prevent the pattern from crossing other numbers
GOAL_MINUTE_PATTERN = r"(?:gol|anot[óo]|marc[óo]|convirti[óo]|diana|remat[óoe])[^0-9]{0,40}(?:al\s+)?(?:minuto\s+)?(\d{1,3})['′\s]"

# Card/expulsion keywords - if these appear near "minuto X", it's NOT a goal claim
CARD_CONTEXT_KEYWORDS = [
    "roja", "expuls", "tarjeta", "card", "amonest", "amarilla", "sanci"
]

# v9: Style violation - prohibited editorial language (without extreme evidence)
STYLE_BLACKLIST_PATTERNS = [
    (r"\brobo\s+arbitral\b", "robo_arbitral"),
    (r"\bescándalo\s+arbitral\b", "escandalo_arbitral"),
    (r"\bvergüenza\b", "verguenza"),
    (r"\binmerecid[oa]\b", "inmerecido"),
    (r"\binjust[oa]\b", "injusto"),
    (r"\bhumillante\b", "humillante"),
    (r"\baplastan(?:te|do)\b", "aplastante"),
    (r"\bgoleada\s+histórica\b", "goleada_historica"),
    (r"\bépic[oa]\b", "epico"),
    (r"\bmilagros[oa]\b", "milagroso"),
    (r"\bincreíble\b", "increible"),
    (r"\bdesastre\b", "desastre"),
    (r"\bcatástrofe\b", "catastrofe"),
    (r"\bpapelón\b", "papelon"),
]


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
        home_stats = stats.get("home", {}) or {}
        away_stats = stats.get("away", {}) or {}

        # Helper to get first non-None value (handles 0 correctly, unlike `or`)
        def first_defined(*values):
            for v in values:
                if v is not None:
                    return v
            return None

        sanitized["stats_summary"] = {
            "home_shots": first_defined(home_stats.get("shots_on_goal"), home_stats.get("Shots on Goal")),
            "away_shots": first_defined(away_stats.get("shots_on_goal"), away_stats.get("Shots on Goal")),
            "home_possession": first_defined(home_stats.get("ball_possession"), home_stats.get("Ball Possession")),
            "away_possession": first_defined(away_stats.get("ball_possession"), away_stats.get("Ball Possession")),
            "home_xg": home_stats.get("expected_goals"),
            "away_xg": away_stats.get("expected_goals"),
        }

    return sanitized


# Normalized event detail variants (multilenguaje/multi-provider)
RED_CARD_EVENT_VARIANTS = [
    "red card",
    "red",
    "tarjeta roja",
    "roja directa",
    "second yellow card",
    "second yellow",
    "segunda amarilla",
    "2nd yellow",
]

PENALTY_EVENT_VARIANTS = [
    "penalty",
    "penal",
    "penalti",
    "from the spot",
    "desde el punto",
]


def _safe_lower(value) -> str:
    """Safely convert value to lowercase string."""
    if isinstance(value, str):
        return value.lower()
    return ""


def _has_red_card_evidence(events: list) -> bool:
    """Check if events contain red card evidence (multilenguaje)."""
    if not events:
        return False

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = _safe_lower(event.get("type", ""))
        detail = _safe_lower(event.get("detail", ""))

        # Check if it's a card event with red card variant
        if event_type == "card":
            for variant in RED_CARD_EVENT_VARIANTS:
                if variant in detail:
                    return True

        # Also check detail directly (some providers put full info there)
        for variant in RED_CARD_EVENT_VARIANTS:
            if variant in detail:
                return True

    return False


def _has_penalty_evidence(events: list) -> bool:
    """Check if events contain penalty evidence (multilenguaje)."""
    if not events:
        return False

    for event in events:
        if not isinstance(event, dict):
            continue
        detail = _safe_lower(event.get("detail", ""))
        event_type = _safe_lower(event.get("type", ""))

        # Check against all penalty variants
        for variant in PENALTY_EVENT_VARIANTS:
            if variant in detail:
                return True

        # Goal scored from penalty
        if event_type == "goal":
            for variant in PENALTY_EVENT_VARIANTS:
                if variant in detail:
                    return True

    return False


def _get_goal_minutes(events: list) -> set[int]:
    """Get set of minutes when goals were scored."""
    minutes = set()
    if not events:
        return minutes

    for event in events:
        if not isinstance(event, dict):
            continue
        if _safe_lower(event.get("type", "")) == "goal":
            minute = event.get("minute")
            if minute is not None:
                try:
                    minutes.add(int(minute))
                except (ValueError, TypeError):
                    pass

    return minutes


def _get_red_card_side(events: list, match_data: dict) -> Optional[str]:
    """
    Determine which side (home/away) received a red card.

    Returns:
        "home", "away", or None if no red card found.
    """
    if not events:
        return None

    home_team = _safe_lower(match_data.get("home_team", ""))
    away_team = _safe_lower(match_data.get("away_team", ""))

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = _safe_lower(event.get("type", ""))
        detail = _safe_lower(event.get("detail", ""))

        # Check if it's a red card event
        is_red = False
        if event_type == "card":
            for variant in RED_CARD_EVENT_VARIANTS:
                if variant in detail:
                    is_red = True
                    break

        if is_red:
            team_name = _safe_lower(event.get("team_name", ""))
            # Match against home/away team names
            if team_name and home_team and team_name in home_team or home_team in team_name:
                return "home"
            if team_name and away_team and team_name in away_team or away_team in team_name:
                return "away"

    # Fallback to stats if events don't have team_name
    stats = match_data.get("stats", {})
    if stats:
        home_reds = stats.get("home", {}).get("red_cards", 0) or 0
        away_reds = stats.get("away", {}).get("red_cards", 0) or 0
        if home_reds > 0 and away_reds == 0:
            return "home"
        if away_reds > 0 and home_reds == 0:
            return "away"

    return None


def _detect_team_mention_near_claim(narrative, claim_patterns: list) -> Optional[str]:
    """
    Detect if narrative mentions 'local' or 'visitante' near a claim pattern.

    Args:
        narrative: Can be str, dict (with 'body' key), or None
        claim_patterns: List of regex patterns to search for

    Returns:
        "local", "visitante", or None if no clear attribution.
    """
    # P0 FIX: Handle dict narrative (e.g., {"title": ..., "body": ...})
    if isinstance(narrative, dict):
        narrative = narrative.get("body", "") or ""
    narrative_lower = _safe_lower(narrative)

    # Find all claim matches
    for pattern in claim_patterns:
        match = re.search(pattern, narrative_lower)
        if match:
            # Get surrounding context (100 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(narrative_lower), match.end() + 100)
            context = narrative_lower[start:end]

            # Check for local/visitante in context
            for local_pattern in LOCAL_PATTERNS:
                if re.search(local_pattern, context):
                    return "local"
            for visitor_pattern in VISITOR_PATTERNS:
                if re.search(visitor_pattern, context):
                    return "visitante"

    return None


def sanitize_narrative_body(body: str) -> tuple[str, list[dict]]:
    """
    Remove control tokens from narrative body.

    Args:
        body: The raw narrative body text.

    Returns:
        Tuple of (sanitized_body, list of stripped token warnings).
    """
    if not body or not isinstance(body, str):
        return body or "", []

    warnings = []
    sanitized = body

    for token in CONTROL_TOKENS:
        # Match token as standalone word or on its own line
        patterns = [
            rf"\b{re.escape(token)}\b",  # Word boundary
            rf"^\s*{re.escape(token)}\s*$",  # Own line (multiline)
        ]

        for pattern in patterns:
            if re.search(pattern, sanitized, re.IGNORECASE | re.MULTILINE):
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
                warnings.append({
                    "type": "control_token_stripped",
                    "token": token,
                })
                logger.warning(f"[CLAIM_VALIDATOR] Stripped control token '{token}' from narrative")
                break  # Don't double-count same token

    # Remove unwanted footer lines (LLM sometimes adds signatures)
    footer_patterns = [
        r"^\s*an[áa]lisis\s+realizado\s+por.*$",  # "Análisis realizado por..."
        r"^\s*este\s+an[áa]lisis\s+fue.*$",  # "Este análisis fue..."
        r"^\s*informe\s+elaborado\s+por.*$",  # "Informe elaborado por..."
        r"^\s*reporte\s+generado\s+por.*$",  # "Reporte generado por..."
    ]
    for pattern in footer_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE | re.MULTILINE):
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
            warnings.append({
                "type": "footer_stripped",
                "pattern": pattern,
            })
            logger.info(f"[CLAIM_VALIDATOR] Stripped footer matching '{pattern}'")

    # Clean up extra whitespace/newlines left behind
    sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)  # Max 2 consecutive newlines
    sanitized = re.sub(r' {2,}', ' ', sanitized)  # Max 1 space
    sanitized = sanitized.strip()

    return sanitized, warnings


def validate_narrative_claims(
    narrative_text,
    match_data: dict,
    strict: bool = True
) -> list[dict]:
    """
    Validate that claims in the narrative are supported by match data.

    Args:
        narrative_text: The generated narrative text (str, dict, or None)
        match_data: The payload sent to the LLM (contains events, stats, etc.)
        strict: If True, unsupported claims are errors. If False, warnings only.

    Returns:
        List of validation errors/warnings:
        [{"type": "unsupported_claim", "claim": "red_card", "pattern": "...", "severity": "error"}]
    """
    errors = []

    # P0 FIX: Normalize narrative_text to string (handle dict with body, None, etc.)
    if isinstance(narrative_text, str):
        narrative_str = narrative_text
    elif isinstance(narrative_text, dict):
        # Extract body from dict narrative (e.g., {"title": ..., "body": ...})
        narrative_str = narrative_text.get("body", "") or ""
        # Fallback: if no body, try to get any string content
        if not narrative_str and "title" in narrative_text:
            narrative_str = narrative_text.get("title", "")
    else:
        narrative_str = str(narrative_text or "")

    narrative_lower = _safe_lower(narrative_str)
    events = match_data.get("events", []) if isinstance(match_data, dict) else []

    # 1. Red card claims - existence check
    has_red = _has_red_card_evidence(events)
    red_card_mentioned = False
    for pattern in RED_CARD_PATTERNS:
        if re.search(pattern, narrative_lower):
            red_card_mentioned = True
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

    # 1b. Red card claims - TEAM ATTRIBUTION check (P0 fix)
    if has_red and red_card_mentioned:
        red_card_side = _get_red_card_side(events, match_data)
        mentioned_side = _detect_team_mention_near_claim(narrative_str, RED_CARD_PATTERNS)

        if red_card_side and mentioned_side:
            # Check for mismatch
            is_mismatch = (
                (red_card_side == "away" and mentioned_side == "local") or
                (red_card_side == "home" and mentioned_side == "visitante")
            )
            if is_mismatch:
                errors.append({
                    "type": "wrong_team_attribution",
                    "claim": "red_card",
                    "expected_side": red_card_side,
                    "mentioned": mentioned_side,
                    "severity": "error",
                })
                logger.warning(
                    f"[CLAIM_VALIDATOR] Red card attribution mismatch: "
                    f"actual={red_card_side}, narrative says={mentioned_side}"
                )

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
    # Only validate if narrative mentions specific minutes with goal context
    goal_minutes = _get_goal_minutes(events)
    # Use finditer to get match objects with positions
    for match in re.finditer(GOAL_MINUTE_PATTERN, narrative_lower):
        minute_str = match.group(1)
        try:
            claimed_minute = int(minute_str)

            # P1 FIX: Check if this is actually a card/expulsion context, not a goal
            # Get surrounding context (50 chars before and after the match)
            context_start = max(0, match.start() - 50)
            context_end = min(len(narrative_lower), match.end() + 50)
            context = narrative_lower[context_start:context_end]

            # Skip if card/expulsion keywords are nearby (false positive)
            is_card_context = any(kw in context for kw in CARD_CONTEXT_KEYWORDS)
            if is_card_context:
                logger.debug(
                    f"[CLAIM_VALIDATOR] Skipping minute {claimed_minute} - card/expulsion context"
                )
                continue

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

    # 4. Validate against derived_facts (P1: reduce inference hallucinations)
    derived_facts = match_data.get("derived_facts", {})
    if derived_facts:
        errors.extend(_validate_against_derived_facts(narrative_str, derived_facts, strict))

    # 5. Style validation (v9: check for prohibited editorial language)
    style_warnings = _validate_style(narrative_str, match_data)
    errors.extend(style_warnings)

    return errors


def _validate_style(narrative: str, match_data: dict) -> list[dict]:
    """
    Validate narrative style for prohibited editorial language.

    v9: Check for blacklisted terms that should only be used with extreme evidence.
    Returns warnings (not errors) to avoid blocking narratives.

    Args:
        narrative: The narrative text
        match_data: Match data for context checking

    Returns:
        List of style validation warnings
    """
    warnings = []
    narrative_lower = _safe_lower(narrative)

    # Get goal margin for context
    derived_facts = match_data.get("derived_facts", {}) if isinstance(match_data, dict) else {}
    result = derived_facts.get("result", {})
    margin = result.get("margin", 0) or 0

    for pattern, term_id in STYLE_BLACKLIST_PATTERNS:
        match = re.search(pattern, narrative_lower)
        if match:
            # Check if term has sufficient evidence
            has_evidence = False

            # "aplastante", "goleada histórica" - only with 5+ goal margin
            if term_id in ("aplastante", "goleada_historica", "humillante"):
                has_evidence = margin >= 5

            # "épico", "milagroso", "increíble" - only for comebacks of 2+ goals
            elif term_id in ("epico", "milagroso", "increible"):
                timeline = derived_facts.get("timeline", {})
                lead_changes = timeline.get("lead_changes_count", 0) or 0
                has_evidence = lead_changes >= 1 and margin >= 0  # Comeback

            # Other terms never allowed without extreme circumstances
            else:
                has_evidence = False

            if not has_evidence:
                warnings.append({
                    "type": "style_violation",
                    "term": term_id,
                    "matched_text": match.group(0),
                    "severity": "warning",  # Never block, just log
                    "reason": f"Editorial term '{term_id}' used without sufficient evidence",
                })
                logger.warning(
                    f"[CLAIM_VALIDATOR] Style violation: '{term_id}' found in narrative"
                )

    return warnings


# Patterns for HT score mentions
HT_SCORE_PATTERNS = [
    r"al\s+descanso",
    r"al\s+medio\s+tiempo",
    r"al\s+intermedio",
    r"en\s+el\s+descanso",
    r"primer\s+tiempo\s+termin[óo]",
    r"al\s+finalizar\s+(?:el\s+)?primer\s+tiempo",
]

# Patterns for stats comparisons
POSSESSION_LEADER_PATTERNS = [
    (r"(?:los\s+)?locales\s+(?:dominaron|tuvieron\s+(?:más|mayor))\s+(?:la\s+)?posesi[óo]n", "home"),
    (r"(?:los\s+)?visitantes\s+(?:dominaron|tuvieron\s+(?:más|mayor))\s+(?:la\s+)?posesi[óo]n", "away"),
    (r"(?:el\s+)?local\s+(?:dominó|tuvo\s+(?:más|mayor))\s+(?:la\s+)?posesi[óo]n", "home"),
    (r"(?:el\s+)?visitante\s+(?:dominó|tuvo\s+(?:más|mayor))\s+(?:la\s+)?posesi[óo]n", "away"),
    (r"mayor\s+posesi[óo]n\s+(?:del|de\s+los)\s+locales", "home"),
    (r"mayor\s+posesi[óo]n\s+(?:del|de\s+los)\s+visitantes", "away"),
]

SHOTS_LEADER_PATTERNS = [
    (r"(?:los\s+)?locales\s+(?:tuvieron|generaron)\s+(?:más|mayor(?:es)?)\s+(?:disparos|remates|tiros)", "home"),
    (r"(?:los\s+)?visitantes\s+(?:tuvieron|generaron)\s+(?:más|mayor(?:es)?)\s+(?:disparos|remates|tiros)", "away"),
    (r"(?:el\s+)?local\s+(?:tuvo|generó)\s+(?:más|mayor(?:es)?)\s+(?:disparos|remates|tiros)", "home"),
    (r"(?:el\s+)?visitante\s+(?:tuvo|generó)\s+(?:más|mayor(?:es)?)\s+(?:disparos|remates|tiros)", "away"),
    (r"más\s+(?:disparos|remates|tiros)(?:\s+al\s+arco)?\s+(?:del|de\s+los)\s+locales", "home"),
    (r"más\s+(?:disparos|remates|tiros)(?:\s+al\s+arco)?\s+(?:del|de\s+los)\s+visitantes", "away"),
]


def _validate_against_derived_facts(
    narrative,
    derived_facts: dict,
    strict: bool = True
) -> list[dict]:
    """
    Validate narrative claims against derived_facts.

    Checks for:
    - HT score mentions when ht_score is null
    - Stats leader contradictions (possession, shots)
    - Red card side contradictions (already checked in main validate, but double-check here)

    Args:
        narrative: The narrative text (str or dict with 'body' key)
        derived_facts: The derived_facts dict from payload
        strict: If True, some checks are errors instead of warnings

    Returns:
        List of validation errors/warnings
    """
    errors = []
    # P0 FIX: Handle dict narrative
    if isinstance(narrative, dict):
        narrative = narrative.get("body", "") or ""
    narrative_lower = _safe_lower(narrative)

    # 4a. HT score validation - if ht_score is null, narrative shouldn't mention it
    result_facts = derived_facts.get("result", {})
    ht_score = result_facts.get("ht_score")

    if ht_score is None:
        for pattern in HT_SCORE_PATTERNS:
            if re.search(pattern, narrative_lower):
                errors.append({
                    "type": "unsupported_claim",
                    "claim": "ht_score",
                    "pattern": pattern,
                    "severity": "warning",  # Warning since HT might be implied from events
                    "derived_facts_value": None,
                    "reason": "ht_score is null but narrative mentions half-time result",
                })
                logger.warning(
                    f"[CLAIM_VALIDATOR] HT score mentioned but derived_facts.ht_score is null"
                )
                break

    # 4b. Possession leader validation
    stats_leaders = derived_facts.get("stats_leaders", {})
    possession_info = stats_leaders.get("possession", {})
    actual_possession_leader = possession_info.get("leader")

    if actual_possession_leader and actual_possession_leader != "tie":
        for pattern, claimed_leader in POSSESSION_LEADER_PATTERNS:
            if re.search(pattern, narrative_lower):
                if claimed_leader != actual_possession_leader:
                    errors.append({
                        "type": "derived_facts_conflict",
                        "claim": "possession_leader",
                        "pattern": pattern,
                        "claimed": claimed_leader,
                        "actual": actual_possession_leader,
                        "severity": "warning",  # Warning for stats conflicts
                    })
                    logger.warning(
                        f"[CLAIM_VALIDATOR] Possession leader conflict: "
                        f"narrative says {claimed_leader}, derived_facts says {actual_possession_leader}"
                    )
                break

    # 4c. Shots leader validation
    shots_info = stats_leaders.get("shots_on_goal", {})
    actual_shots_leader = shots_info.get("leader")

    if actual_shots_leader and actual_shots_leader != "tie":
        for pattern, claimed_leader in SHOTS_LEADER_PATTERNS:
            if re.search(pattern, narrative_lower):
                if claimed_leader != actual_shots_leader:
                    errors.append({
                        "type": "derived_facts_conflict",
                        "claim": "shots_leader",
                        "pattern": pattern,
                        "claimed": claimed_leader,
                        "actual": actual_shots_leader,
                        "severity": "warning",
                    })
                    logger.warning(
                        f"[CLAIM_VALIDATOR] Shots leader conflict: "
                        f"narrative says {claimed_leader}, derived_facts says {actual_shots_leader}"
                    )
                break

    # 4d. Red card side validation using derived_facts (more reliable than events parsing)
    discipline = derived_facts.get("discipline", {})
    first_red = discipline.get("first_red_card", {})

    if first_red.get("exists") and first_red.get("side"):
        actual_side = first_red["side"]  # "home" or "away"
        mentioned_side = _detect_team_mention_near_claim(narrative, RED_CARD_PATTERNS)

        if mentioned_side:
            # Map mentioned side to home/away
            is_mismatch = (
                (actual_side == "away" and mentioned_side == "local") or
                (actual_side == "home" and mentioned_side == "visitante")
            )
            if is_mismatch:
                # Only add if not already caught by main validation
                existing_types = [e.get("type") for e in errors]
                if "wrong_team_attribution" not in existing_types:
                    errors.append({
                        "type": "derived_facts_conflict",
                        "claim": "red_card_side",
                        "derived_side": actual_side,
                        "narrative_says": mentioned_side,
                        "severity": "error",  # This is a serious factual error
                    })
                    logger.warning(
                        f"[CLAIM_VALIDATOR] Red card side conflict via derived_facts: "
                        f"actual={actual_side}, narrative says={mentioned_side}"
                    )

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
