"""
Post-match narrative generator using RunPod LLM.

Builds compact prompts and validates JSON responses.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from app.config import get_settings
from app.llm.runpod_client import RunPodClient, RunPodError, RunPodJobResult

logger = logging.getLogger(__name__)


# Required keys in LLM JSON response
REQUIRED_KEYS = {"match_id", "lang", "result", "narrative"}


@dataclass
class NarrativeResult:
    """Result from narrative generation."""

    status: str  # ok, error, disabled
    narrative_json: Optional[dict] = None
    model: str = "qwen-vllm"
    delay_ms: int = 0
    exec_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    worker_id: str = ""
    error: Optional[str] = None


def build_narrative_prompt(match_data: dict) -> str:
    """
    Build a compact prompt for post-match narrative generation.

    Args:
        match_data: Dict with match info (teams, score, stats, prediction, etc.)

    Returns:
        Prompt string for LLM.
    """
    # Extract key fields
    match_id = match_data.get("match_id", 0)
    home_team = match_data.get("home_team", "Local")
    away_team = match_data.get("away_team", "Visitante")
    league = match_data.get("league_name", "Liga")
    match_date = match_data.get("date", "")
    home_goals = match_data.get("home_goals", 0)
    away_goals = match_data.get("away_goals", 0)

    # Stats (compact)
    stats = match_data.get("stats", {})
    home_stats = stats.get("home", {})
    away_stats = stats.get("away", {})

    stats_summary = {
        "possession": f"{home_stats.get('ball_possession', '?')} vs {away_stats.get('ball_possession', '?')}",
        "shots": f"{home_stats.get('total_shots', '?')} vs {away_stats.get('total_shots', '?')}",
        "shots_on_target": f"{home_stats.get('shots_on_goal', '?')} vs {away_stats.get('shots_on_goal', '?')}",
        "xG": f"{home_stats.get('expected_goals', '?')} vs {away_stats.get('expected_goals', '?')}",
    }

    # Prediction info (if available)
    prediction = match_data.get("prediction", {})
    pred_info = ""
    if prediction:
        probs = prediction.get("probabilities", {})
        pred_result = prediction.get("predicted_result", "")
        confidence = prediction.get("confidence", 0)
        pred_info = f"""
Predicción pre-partido:
- Probabilidades: H={probs.get('home', 0):.1%}, D={probs.get('draw', 0):.1%}, A={probs.get('away', 0):.1%}
- Resultado predicho: {pred_result} (confianza: {confidence:.1%})
- ¿Acertó?: {prediction.get('correct', False)}"""

    # Events summary (cap at 5 most important)
    events = match_data.get("events", [])[:5]
    events_str = ""
    if events:
        events_str = "\nEventos clave:\n" + "\n".join(
            f"- min {e.get('minute', '?')}: {e.get('type', '')} - {e.get('detail', '')}"
            for e in events
        )

    # Market odds (if available)
    odds = match_data.get("market_odds", {})
    odds_str = ""
    if odds and odds.get("home"):
        odds_str = f"\nCuotas mercado: H={odds.get('home')}, D={odds.get('draw')}, A={odds.get('away')}"

    prompt = f"""Eres un analista de fútbol. Genera una narrativa post-partido en español.

INSTRUCCIONES CRÍTICAS:
1. DEVUELVE SOLO JSON válido, sin texto antes ni después.
2. Usa SOLO los datos proporcionados. NO inventes información.
3. Tono: profesional, serio, sin exageraciones ni emojis.
4. Máximo ~500 palabras en el campo "narrative".

DATOS DEL PARTIDO:
- match_id: {match_id}
- Liga: {league}
- Fecha: {match_date}
- {home_team} vs {away_team}
- Resultado final: {home_goals} - {away_goals}

Estadísticas:
- Posesión: {stats_summary['possession']}
- Tiros: {stats_summary['shots']}
- Tiros a puerta: {stats_summary['shots_on_target']}
- xG: {stats_summary['xG']}
{pred_info}
{events_str}
{odds_str}

SCHEMA JSON OBLIGATORIO:
{{
  "match_id": {match_id},
  "lang": "es",
  "result": "{home_goals}-{away_goals}",
  "narrative": "Párrafos de análisis del partido...",
  "key_factors": ["factor1", "factor2"],
  "prediction_analysis": "Análisis de si la predicción acertó o falló y por qué",
  "usage_hint": "Breve nota sobre cómo usar esta narrativa"
}}

DEVUELVE SOLO EL JSON:"""

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
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        return None


def validate_narrative_json(data: dict, match_id: int) -> bool:
    """
    Validate that JSON has required keys and correct match_id.

    Args:
        data: Parsed JSON dict.
        match_id: Expected match ID.

    Returns:
        True if valid.
    """
    if not isinstance(data, dict):
        return False

    # Check required keys
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        logger.warning(f"Missing required keys: {missing}")
        return False

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
            return NarrativeResult(status="disabled")

        if not self.settings.RUNPOD_API_KEY:
            return NarrativeResult(status="error", error="RUNPOD_API_KEY not configured")

        match_id = match_data.get("match_id", 0)

        try:
            # First attempt
            prompt = build_narrative_prompt(match_data)
            result = await self.client.generate(prompt)

            parsed = parse_json_response(result.text)
            if parsed and validate_narrative_json(parsed, match_id):
                return NarrativeResult(
                    status="ok",
                    narrative_json=parsed,
                    delay_ms=result.delay_ms,
                    exec_ms=result.exec_ms,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    worker_id=result.worker_id,
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
            if parsed2 and validate_narrative_json(parsed2, match_id):
                return NarrativeResult(
                    status="ok",
                    narrative_json=parsed2,
                    delay_ms=result2.delay_ms,
                    exec_ms=result2.exec_ms,
                    tokens_in=result2.tokens_in,
                    tokens_out=result2.tokens_out,
                    worker_id=result2.worker_id,
                )

            # Both attempts failed
            error_msg = "JSON validation failed after retry"
            logger.error(f"Narrative generation failed for match {match_id}: {error_msg}")
            return NarrativeResult(
                status="error",
                error=error_msg,
                delay_ms=result2.delay_ms,
                exec_ms=result2.exec_ms,
                tokens_in=result2.tokens_in,
                tokens_out=result2.tokens_out,
                worker_id=result2.worker_id,
            )

        except RunPodError as e:
            logger.error(f"RunPod error for match {match_id}: {e}")
            return NarrativeResult(status="error", error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error for match {match_id}: {e}")
            return NarrativeResult(status="error", error=str(e))
