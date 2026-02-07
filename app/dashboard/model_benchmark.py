"""
Model Benchmark endpoint for dashboard.

Returns historical accuracy data for selected models with dynamic date ranges.
Supports Market, Model A, Shadow, and Sensor B models.
"""

import logging
from collections import defaultdict
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_async_session
from app.security import verify_dashboard_token

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(verify_dashboard_token)],
)


# =============================================================================
# Configuration
# =============================================================================

# Model availability dates (when each model started producing reliable predictions)
MODEL_AVAILABILITY = {
    "Market": "2026-01-04",     # Odds available from the start
    "Model A": "2026-01-10",    # Post-important adjustments
    "Shadow": "2026-01-15",     # Shadow model start
    "Sensor B": "2026-01-15",   # Sensor B start
}

VALID_MODELS = set(MODEL_AVAILABILITY.keys())


# =============================================================================
# Pydantic Models
# =============================================================================


class DailyModelStats(BaseModel):
    """Daily aggregated stats for each model."""

    date: str  # ISO format: "2026-01-17"
    matches: int
    market_correct: int
    model_a_correct: int
    shadow_correct: int
    sensor_b_correct: int


class ModelSummary(BaseModel):
    """Summary stats for a single model."""

    name: str
    accuracy: float  # 0-100
    correct: int
    total: int
    days_won: float  # Fractional: ties split the day (2 tied = 0.5 each)


class ModelBenchmarkResponse(BaseModel):
    """Response for /dashboard/model-benchmark endpoint."""

    generated_at: str
    start_date: str  # Dynamic based on selected models
    selected_models: List[str]  # Models included in comparison
    total_matches: int
    daily_data: List[DailyModelStats]
    models: List[ModelSummary]


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_start_date(selected_models: List[str]) -> date:
    """
    Calculate the start date based on selected models.
    Returns the most recent availability date among selected models as a date object.
    """
    date_strs = [MODEL_AVAILABILITY[m] for m in selected_models if m in MODEL_AVAILABILITY]
    max_str = max(date_strs) if date_strs else "2026-01-10"
    return date.fromisoformat(max_str)


def parse_models_param(models_param: Optional[str]) -> List[str]:
    """
    Parse comma-separated models parameter.
    Returns list of valid model names.
    """
    if not models_param:
        return list(VALID_MODELS)  # Default: all models

    models = [m.strip() for m in models_param.split(",")]
    valid = [m for m in models if m in VALID_MODELS]
    return valid


# =============================================================================
# Co-pick Logic (matches frontend epsilon logic)
# =============================================================================

PROB_EPSILON = 0.005  # 0.5 percentage points for 0-1 scale


def get_top_outcomes(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    epsilon: float = PROB_EPSILON
) -> set:
    """
    Get all outcomes within epsilon of max probability (co-picks).

    Returns set of 'H', 'D', 'A' that are tied for maximum.
    """
    max_prob = max(home_prob, draw_prob, away_prob)
    top = set()
    if abs(home_prob - max_prob) <= epsilon:
        top.add('H')
    if abs(draw_prob - max_prob) <= epsilon:
        top.add('D')
    if abs(away_prob - max_prob) <= epsilon:
        top.add('A')
    return top


def is_prediction_correct(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    actual_outcome: str  # 'H', 'D', or 'A'
) -> bool:
    """
    Determine if prediction is correct using co-pick logic with epsilon.

    Returns True if actual_outcome is among the co-picks.

    Examples:
        - H=0.41, D=0.41, A=0.18, actual='D' → True (D is co-pick)
        - H=0.41, D=0.41, A=0.18, actual='A' → False (A not in co-picks)
        - H=0.52, D=0.28, A=0.20, actual='H' → True (single pick)
    """
    top_outcomes = get_top_outcomes(home_prob, draw_prob, away_prob)
    return actual_outcome in top_outcomes


# =============================================================================
# Endpoint
# =============================================================================


@router.get("/model-benchmark", response_model=ModelBenchmarkResponse)
async def get_model_benchmark(
    models: Optional[str] = Query(
        None,
        description="Comma-separated model names (Market, Model A, Shadow, Sensor B). Default: all models."
    ),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get historical model benchmark data with dynamic date range.

    Query params:
        models: Comma-separated list of models to compare.
                Examples: "Market,Model A" or "Market,Model A,Shadow,Sensor B"

    Rules:
        - Minimum 2 models required for comparison
        - Start date is the MOST RECENT availability date among selected models
        - Only includes matches where ALL selected models have predictions

    Examples:
        ?models=Market,Model%20A          -> desde 2026-01-10, ~800 matches
        ?models=Market,Model%20A,Shadow   -> desde 2026-01-15, ~400 matches
        ?models=Model%20A                 -> HTTP 400 (minimum 2 required)
    """
    try:
        # Parse and validate models
        selected = parse_models_param(models)

        if len(selected) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Selecciona al menos 2 modelos para comparar. Recibido: {selected}"
            )

        # Calculate dynamic start date
        start_date = calculate_start_date(selected)

        # Build inclusion flags
        include_market = "Market" in selected
        include_model_a = "Model A" in selected
        include_shadow = "Shadow" in selected
        include_sensor_b = "Sensor B" in selected

        # Build dynamic query - returns raw probabilities for Python co-pick processing
        # Use America/Los_Angeles timezone to match dashboard UI grouping
        query = text("""
            WITH match_data AS (
                SELECT
                    m.id as match_id,
                    (m.date AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date as match_date,
                    -- Actual outcome as H/D/A
                    CASE
                        WHEN m.home_goals > m.away_goals THEN 'H'
                        WHEN m.home_goals < m.away_goals THEN 'A'
                        ELSE 'D'
                    END as actual_outcome,

                    -- Market implied probabilities (normalized, with guardrails)
                    CASE
                        WHEN m.odds_home > 0 AND m.odds_draw > 0 AND m.odds_away > 0 THEN
                            (1.0 / m.odds_home) / (1.0/m.odds_home + 1.0/m.odds_draw + 1.0/m.odds_away)
                        ELSE NULL
                    END as market_home_prob,
                    CASE
                        WHEN m.odds_home > 0 AND m.odds_draw > 0 AND m.odds_away > 0 THEN
                            (1.0 / m.odds_draw) / (1.0/m.odds_home + 1.0/m.odds_draw + 1.0/m.odds_away)
                        ELSE NULL
                    END as market_draw_prob,
                    CASE
                        WHEN m.odds_home > 0 AND m.odds_draw > 0 AND m.odds_away > 0 THEN
                            (1.0 / m.odds_away) / (1.0/m.odds_home + 1.0/m.odds_draw + 1.0/m.odds_away)
                        ELSE NULL
                    END as market_away_prob,

                    -- Model A raw probabilities (active model version from config)
                    (SELECT p.home_prob FROM predictions p
                     WHERE p.match_id = m.id AND p.model_version = :model_a_version
                     ORDER BY p.created_at DESC LIMIT 1) as model_a_home,
                    (SELECT p.draw_prob FROM predictions p
                     WHERE p.match_id = m.id AND p.model_version = :model_a_version
                     ORDER BY p.created_at DESC LIMIT 1) as model_a_draw,
                    (SELECT p.away_prob FROM predictions p
                     WHERE p.match_id = m.id AND p.model_version = :model_a_version
                     ORDER BY p.created_at DESC LIMIT 1) as model_a_away,

                    -- Shadow raw probabilities
                    (SELECT shp.shadow_home_prob FROM shadow_predictions shp
                     WHERE shp.match_id = m.id
                     ORDER BY shp.created_at DESC LIMIT 1) as shadow_home_prob,
                    (SELECT shp.shadow_draw_prob FROM shadow_predictions shp
                     WHERE shp.match_id = m.id
                     ORDER BY shp.created_at DESC LIMIT 1) as shadow_draw_prob,
                    (SELECT shp.shadow_away_prob FROM shadow_predictions shp
                     WHERE shp.match_id = m.id
                     ORDER BY shp.created_at DESC LIMIT 1) as shadow_away_prob,

                    -- Sensor B raw probabilities
                    (SELECT sp.b_home_prob FROM sensor_predictions sp
                     WHERE sp.match_id = m.id
                     ORDER BY sp.created_at DESC LIMIT 1) as sensor_b_home_prob,
                    (SELECT sp.b_draw_prob FROM sensor_predictions sp
                     WHERE sp.match_id = m.id
                     ORDER BY sp.created_at DESC LIMIT 1) as sensor_b_draw_prob,
                    (SELECT sp.b_away_prob FROM sensor_predictions sp
                     WHERE sp.match_id = m.id
                     ORDER BY sp.created_at DESC LIMIT 1) as sensor_b_away_prob

                FROM matches m
                WHERE m.status IN ('FT', 'PEN')  -- Include penalties (90' result is draw)
                    AND m.date >= :start_date
                    AND m.home_goals IS NOT NULL
                    AND m.away_goals IS NOT NULL
            )
            SELECT * FROM match_data
            WHERE
                -- Filter based on selected models having data
                (market_home_prob IS NOT NULL OR NOT :include_market)
                AND (model_a_home IS NOT NULL OR NOT :include_model_a)
                AND (shadow_home_prob IS NOT NULL OR NOT :include_shadow)
                AND (sensor_b_home_prob IS NOT NULL OR NOT :include_sensor_b)
            ORDER BY match_date
        """)

        settings = get_settings()
        result = await db.execute(
            query,
            {
                "start_date": start_date,
                "model_a_version": settings.MODEL_VERSION,
                "include_market": include_market,
                "include_model_a": include_model_a,
                "include_shadow": include_shadow,
                "include_sensor_b": include_sensor_b,
            }
        )
        rows = result.fetchall()

        if not rows:
            return ModelBenchmarkResponse(
                generated_at=datetime.utcnow().isoformat(),
                start_date=start_date.isoformat(),
                selected_models=selected,
                total_matches=0,
                daily_data=[],
                models=[],
            )

        # Aggregate by day with co-pick logic
        daily_stats = defaultdict(lambda: {
            'matches': 0,
            'market_correct': 0,
            'model_a_correct': 0,
            'shadow_correct': 0,
            'sensor_b_correct': 0,
        })

        for row in rows:
            day = row.match_date.isoformat()
            daily_stats[day]['matches'] += 1
            actual = row.actual_outcome  # Already 'H'/'D'/'A' from SQL

            # Market (co-pick logic)
            if include_market and row.market_home_prob is not None:
                if is_prediction_correct(
                    row.market_home_prob, row.market_draw_prob, row.market_away_prob, actual
                ):
                    daily_stats[day]['market_correct'] += 1

            # Model A (co-pick logic)
            if include_model_a and row.model_a_home is not None:
                if is_prediction_correct(
                    row.model_a_home, row.model_a_draw, row.model_a_away, actual
                ):
                    daily_stats[day]['model_a_correct'] += 1

            # Shadow (co-pick logic)
            if include_shadow and row.shadow_home_prob is not None:
                if is_prediction_correct(
                    row.shadow_home_prob, row.shadow_draw_prob, row.shadow_away_prob, actual
                ):
                    daily_stats[day]['shadow_correct'] += 1

            # Sensor B (co-pick logic)
            if include_sensor_b and row.sensor_b_home_prob is not None:
                if is_prediction_correct(
                    row.sensor_b_home_prob, row.sensor_b_draw_prob, row.sensor_b_away_prob, actual
                ):
                    daily_stats[day]['sensor_b_correct'] += 1

        # Convert to list sorted by date
        daily_data = [
            DailyModelStats(
                date=day,
                matches=stats['matches'],
                market_correct=stats['market_correct'],
                model_a_correct=stats['model_a_correct'],
                shadow_correct=stats['shadow_correct'],
                sensor_b_correct=stats['sensor_b_correct'],
            )
            for day, stats in sorted(daily_stats.items())
        ]

        # Calculate cumulative stats
        total_matches = sum(d.matches for d in daily_data)
        total_market_correct = sum(d.market_correct for d in daily_data)
        total_model_a_correct = sum(d.model_a_correct for d in daily_data)
        total_shadow_correct = sum(d.shadow_correct for d in daily_data)
        total_sensor_b_correct = sum(d.sensor_b_correct for d in daily_data)

        # Calculate days won for each model (only among selected models)
        # Ties split the day: 2 tied = 0.5 each, 4 tied = 0.25 each
        def count_days_won(model_correct_fn, model_name: str) -> float:
            if model_name not in selected:
                return 0.0
            total = 0.0
            for d in daily_data:
                model_val = model_correct_fn(d)
                # Only compare against selected models
                candidates = []
                if include_market:
                    candidates.append(d.market_correct)
                if include_model_a:
                    candidates.append(d.model_a_correct)
                if include_shadow:
                    candidates.append(d.shadow_correct)
                if include_sensor_b:
                    candidates.append(d.sensor_b_correct)

                if not candidates:
                    continue

                max_val = max(candidates)
                if max_val <= 0:
                    continue

                # Count how many models tied for first place
                num_tied = sum(1 for c in candidates if c == max_val)

                # If this model is tied for first, add fractional day
                if model_val == max_val:
                    total += 1.0 / num_tied

            return round(total, 2)

        # Build models list (only selected models)
        models_list = []

        if include_market:
            models_list.append(ModelSummary(
                name="Market",
                accuracy=round((total_market_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_market_correct,
                total=total_matches,
                days_won=count_days_won(lambda d: d.market_correct, "Market"),
            ))

        if include_model_a:
            models_list.append(ModelSummary(
                name="Model A",
                accuracy=round((total_model_a_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_model_a_correct,
                total=total_matches,
                days_won=count_days_won(lambda d: d.model_a_correct, "Model A"),
            ))

        if include_shadow:
            models_list.append(ModelSummary(
                name="Shadow",
                accuracy=round((total_shadow_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_shadow_correct,
                total=total_matches,
                days_won=count_days_won(lambda d: d.shadow_correct, "Shadow"),
            ))

        if include_sensor_b:
            models_list.append(ModelSummary(
                name="Sensor B",
                accuracy=round((total_sensor_b_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_sensor_b_correct,
                total=total_matches,
                days_won=count_days_won(lambda d: d.sensor_b_correct, "Sensor B"),
            ))

        return ModelBenchmarkResponse(
            generated_at=datetime.utcnow().isoformat(),
            start_date=start_date.isoformat(),
            selected_models=selected,
            total_matches=total_matches,
            daily_data=daily_data,
            models=models_list,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch model benchmark data")
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmark: {str(e)}")
