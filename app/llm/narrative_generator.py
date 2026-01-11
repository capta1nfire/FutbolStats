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


def build_narrative_prompt(match_data: dict) -> str:
    """
    Build prompt v4 for post-match narrative generation.

    v4 improvements:
    - Only digits for numbers (never "uno/dos/tres")
    - No score repetition in body (user sees it in UI)
    - Team names max 1x each, then use aliases from team_aliases only
    - key_factors ultra-short (max 120 chars each)
    - Shorter body (140-240 words)
    - team_aliases provided to prevent hallucinated nicknames

    Args:
        match_data: Dict with match info (teams, score, stats, prediction, team_aliases, etc.)

    Returns:
        Prompt string for LLM.
    """
    # Extract key fields
    match_id = match_data.get("match_id", 0)
    home_team = match_data.get("home_team", "Local")
    away_team = match_data.get("away_team", "Visitante")
    league = match_data.get("league_name", "Liga")
    match_date = match_data.get("date", "")
    home_goals = match_data.get("home_goals", 0) or 0
    away_goals = match_data.get("away_goals", 0) or 0

    # Team aliases (curated, safe to use)
    team_aliases = match_data.get("team_aliases", {})
    home_aliases = team_aliases.get("home", [home_team, "los locales"])
    away_aliases = team_aliases.get("away", [away_team, "los visitantes"])
    home_aliases_json = json.dumps(home_aliases, ensure_ascii=False)
    away_aliases_json = json.dumps(away_aliases, ensure_ascii=False)

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

    prompt = f"""Eres un analista de fútbol profesional. Escribes en español neutral, serio, sin hype y sin emojis.

REGLAS CRÍTICAS v4 (OBLIGATORIAS):

1) DEVUELVE SOLO JSON VÁLIDO. No incluyas texto antes ni después.

2) SOLO DÍGITOS PARA NÚMEROS: Escribe "1", "2", "54%", nunca "uno", "dos", "cincuenta y cuatro por ciento".

3) NO REPITAS EL MARCADOR en narrative.body. El usuario ya ve el resultado en la UI. Enfócate en el "por qué", no en el "qué".

4) NOMBRES DE EQUIPOS - REGLA ESTRICTA (CRÍTICA):
   - Menciona cada nombre de equipo MÁXIMO 1 vez en todo el body.
   - Después usa SOLO aliases de la lista team_aliases proporcionada abajo.
   - PROHIBIDO inventar apodos/sobrenombres que no estén en team_aliases.
   - Aliases permitidos para LOCAL: {home_aliases_json}
   - Aliases permitidos para VISITANTE: {away_aliases_json}
   - Si usas un alias que NO está en estas listas, tu respuesta será RECHAZADA.

5) Usa SOLO los datos proporcionados en "DATOS". NO inventes jugadores, lesiones, alineaciones, tácticas, xG, tarjetas, ni nada que no esté explícitamente en el JSON.

6) match_id debe ser EXACTAMENTE el número recibido en DATOS.

7) Si un dato no existe (ej. expected_goals, ball_possession), NO lo menciones.

8) En narrative.body usa saltos de párrafo como "\\n\\n".

9) Redondeo:
   - prediction.confidence y prediction.probabilities a 2 decimales.
   - probabilities debe sumar 1 ± 0.01. Si no suma, renormaliza.

10) LONGITUD REDUCIDA: narrative.body debe tener 2-3 párrafos, 140-240 palabras. Sé conciso.

11) key_factors NO DUPLICA el body:
    - Cada evidence máx 120 caracteres.
    - Debe contener al menos 1 cifra (stats) o 1 minuto (events).
    - REGLA EVENTS VACÍO: Si el array events está vacío ([]), en key_factors[label="Events"] usa:
      * direction: "neutral"
      * evidence: "Eventos no disponibles" (EXACTAMENTE este texto, no "No hubo eventos")

12) Tono según resultado de predicción:
    - Si prediction.correct = true: refuerza con 2-3 evidencias numéricas.
    - Si prediction.correct = false: matiza con 1-2 factores de los datos.
    - Nunca uses "suerte/varianza" sin evidencia cuantitativa (xG vs goles).

13) title: máx 8 palabras, sin comillas, sin \\n.

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

stats.home: {home_stats_json}
stats.away: {away_stats_json}

prediction: {prediction_json}

events: {events_json}

market_odds: {market_odds_json}

RECUERDA: JSON válido, sin marcador en body, solo aliases permitidos, key_factors cortos y distintos del body."""

    return prompt


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
        # Log full text length and content for debugging
        logger.warning(f"Raw text length: {len(json_text)}")
        logger.warning(f"Raw text (first 500 chars): {json_text[:500]}")
        return None


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

    # Validate narrative sub-object
    narrative = data.get("narrative", {})
    if isinstance(narrative, dict):
        missing_narrative = REQUIRED_NARRATIVE_KEYS - set(narrative.keys())
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
    """Generates post-match narratives using RunPod LLM."""

    def __init__(self):
        self.settings = get_settings()
        self.client = RunPodClient()
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
            prompt = build_narrative_prompt(match_data)
            result = await self.client.generate(prompt)

            # Debug: log token usage and raw output
            logger.info(f"LLM response: tokens_in={result.tokens_in}, tokens_out={result.tokens_out}, text_len={len(result.text)}")
            logger.info(f"LLM raw_output keys: {list(result.raw_output.keys())}")
            if "output" in result.raw_output:
                logger.info(f"LLM output structure: {result.raw_output['output'][:1] if result.raw_output['output'] else 'empty'}")

            parsed = parse_json_response(result.text)
            schema_valid = parsed is not None and validate_narrative_json(parsed, match_id)

            if schema_valid and parsed:
                tone = parsed.get("narrative", {}).get("tone") if isinstance(parsed.get("narrative"), dict) else None
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
