"""
Model Benchmark endpoint for dashboard.

Returns historical accuracy data for selected models with dynamic date ranges.
Supports Market, Model A, Shadow, and Sensor B models.
"""

import logging
from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.logos.auth import verify_dashboard_token

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
    days_won: int


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

        # Build dynamic query
        query = text("""
            WITH match_predictions AS (
                SELECT
                    m.id as match_id,
                    DATE(m.date) as match_date,
                    m.home_goals,
                    m.away_goals,
                    -- Market prediction from odds (lowest odds = most probable)
                    CASE
                        WHEN m.odds_home < m.odds_draw AND m.odds_home < m.odds_away THEN 'H'
                        WHEN m.odds_draw < m.odds_home AND m.odds_draw < m.odds_away THEN 'D'
                        ELSE 'A'
                    END as market_pred,
                    -- Model A (v1.0.0) prediction
                    (SELECT CASE
                        WHEN p.home_prob > p.draw_prob AND p.home_prob > p.away_prob THEN 'H'
                        WHEN p.draw_prob > p.home_prob AND p.draw_prob > p.away_prob THEN 'D'
                        ELSE 'A'
                    END
                    FROM predictions p
                    WHERE p.match_id = m.id AND p.model_version = 'v1.0.0'
                    LIMIT 1) as model_a_pred,
                    -- Shadow (v1.1.0-two_stage) prediction
                    (SELECT CASE
                        WHEN p.home_prob > p.draw_prob AND p.home_prob > p.away_prob THEN 'H'
                        WHEN p.draw_prob > p.home_prob AND p.draw_prob > p.away_prob THEN 'D'
                        ELSE 'A'
                    END
                    FROM predictions p
                    WHERE p.match_id = m.id AND p.model_version = 'v1.1.0-two_stage'
                    LIMIT 1) as shadow_pred,
                    -- Sensor B prediction from sensor_predictions table (convert to H/D/A)
                    (SELECT CASE
                        WHEN sp.b_pick = 'home' THEN 'H'
                        WHEN sp.b_pick = 'draw' THEN 'D'
                        WHEN sp.b_pick = 'away' THEN 'A'
                        ELSE NULL
                    END
                    FROM sensor_predictions sp
                    WHERE sp.match_id = m.id
                    LIMIT 1) as sensor_b_pred
                FROM matches m
                WHERE m.status IN ('FT', 'PEN')  -- Include penalties (90' result is draw)
                    AND m.date >= :start_date
                    -- Must have odds (required for market)
                    AND m.odds_home IS NOT NULL
                    AND m.odds_draw IS NOT NULL
                    AND m.odds_away IS NOT NULL
            ),
            -- Only matches where SELECTED models have predictions
            complete_matches AS (
                SELECT *
                FROM match_predictions
                WHERE
                    -- Only validate presence of SELECTED models
                    (market_pred IS NOT NULL OR NOT :include_market)
                    AND (model_a_pred IS NOT NULL OR NOT :include_model_a)
                    AND (shadow_pred IS NOT NULL OR NOT :include_shadow)
                    AND (sensor_b_pred IS NOT NULL OR NOT :include_sensor_b)
            ),
            daily_results AS (
                SELECT
                    match_date,
                    COUNT(*) as total_matches,
                    SUM(CASE
                        WHEN (home_goals > away_goals AND market_pred = 'H') OR
                             (home_goals = away_goals AND market_pred = 'D') OR
                             (home_goals < away_goals AND market_pred = 'A') THEN 1
                        ELSE 0
                    END) as market_correct,
                    SUM(CASE
                        WHEN (home_goals > away_goals AND model_a_pred = 'H') OR
                             (home_goals = away_goals AND model_a_pred = 'D') OR
                             (home_goals < away_goals AND model_a_pred = 'A') THEN 1
                        ELSE 0
                    END) as model_a_correct,
                    SUM(CASE
                        WHEN (home_goals > away_goals AND shadow_pred = 'H') OR
                             (home_goals = away_goals AND shadow_pred = 'D') OR
                             (home_goals < away_goals AND shadow_pred = 'A') THEN 1
                        ELSE 0
                    END) as shadow_correct,
                    SUM(CASE
                        WHEN (home_goals > away_goals AND sensor_b_pred = 'H') OR
                             (home_goals = away_goals AND sensor_b_pred = 'D') OR
                             (home_goals < away_goals AND sensor_b_pred = 'A') THEN 1
                        ELSE 0
                    END) as sensor_b_correct
                FROM complete_matches
                GROUP BY match_date
                ORDER BY match_date
            )
            SELECT * FROM daily_results
        """)

        result = await db.execute(
            query,
            {
                "start_date": start_date,
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

        # Transform to response format
        daily_data = []
        for row in rows:
            daily_data.append(
                DailyModelStats(
                    date=row.match_date.isoformat(),
                    matches=row.total_matches,
                    market_correct=row.market_correct or 0,
                    model_a_correct=row.model_a_correct or 0,
                    shadow_correct=row.shadow_correct or 0,
                    sensor_b_correct=row.sensor_b_correct or 0,
                )
            )

        # Calculate cumulative stats
        total_matches = sum(d.matches for d in daily_data)
        total_market_correct = sum(d.market_correct for d in daily_data)
        total_model_a_correct = sum(d.model_a_correct for d in daily_data)
        total_shadow_correct = sum(d.shadow_correct for d in daily_data)
        total_sensor_b_correct = sum(d.sensor_b_correct for d in daily_data)

        # Calculate days won for each model (only among selected models)
        def count_days_won(model_correct_fn, model_name: str) -> int:
            if model_name not in selected:
                return 0
            count = 0
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

                max_val = max(candidates) if candidates else 0
                if model_val == max_val and max_val > 0:
                    count += 1
            return count

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
