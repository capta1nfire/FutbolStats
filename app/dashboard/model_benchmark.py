"""
Model Benchmark endpoint for dashboard.

Returns historical accuracy data for Market, Model A, Shadow, and Sensor B models
since 2026-01-17 for comparison visualization.
"""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
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
    start_date: str  # "2026-01-17"
    total_matches: int
    daily_data: List[DailyModelStats]
    models: List[ModelSummary]


# =============================================================================
# Endpoint
# =============================================================================


@router.get("/model-benchmark", response_model=ModelBenchmarkResponse)
async def get_model_benchmark(
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get historical model benchmark data.

    Returns daily accuracy for:
    - Market (from odds)
    - Model A (v1.0.0)
    - Shadow (v1.1.0-two_stage)
    - Sensor B (from sensor_predictions table)

    Only includes matches from 2026-01-17 onwards with odds and Model A predictions.

    Market prediction: outcome with lowest odds (most probable according to bookmakers).
    Model prediction: outcome with highest probability from model.
    Sensor B: b_pick from sensor_predictions table.
    """
    try:
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
                WHERE m.status = 'FT'
                    AND m.date >= '2026-01-17'
                    -- Must have odds (market)
                    AND m.odds_home IS NOT NULL
                    AND m.odds_draw IS NOT NULL
                    AND m.odds_away IS NOT NULL
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
                        WHEN model_a_pred IS NOT NULL AND (
                            (home_goals > away_goals AND model_a_pred = 'H') OR
                            (home_goals = away_goals AND model_a_pred = 'D') OR
                            (home_goals < away_goals AND model_a_pred = 'A')
                        ) THEN 1
                        ELSE 0
                    END) as model_a_correct,
                    SUM(CASE
                        WHEN shadow_pred IS NOT NULL AND (
                            (home_goals > away_goals AND shadow_pred = 'H') OR
                            (home_goals = away_goals AND shadow_pred = 'D') OR
                            (home_goals < away_goals AND shadow_pred = 'A')
                        ) THEN 1
                        ELSE 0
                    END) as shadow_correct,
                    SUM(CASE
                        WHEN sensor_b_pred IS NOT NULL AND (
                            (home_goals > away_goals AND sensor_b_pred = 'H') OR
                            (home_goals = away_goals AND sensor_b_pred = 'D') OR
                            (home_goals < away_goals AND sensor_b_pred = 'A')
                        ) THEN 1
                        ELSE 0
                    END) as sensor_b_correct
                FROM match_predictions
                WHERE model_a_pred IS NOT NULL  -- Only matches with Model A predictions
                GROUP BY match_date
                ORDER BY match_date
            )
            SELECT * FROM daily_results
        """)

        result = await db.execute(query)
        rows = result.fetchall()

        if not rows:
            return ModelBenchmarkResponse(
                generated_at=datetime.utcnow().isoformat(),
                start_date="2026-01-17",
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
                    market_correct=row.market_correct,
                    model_a_correct=row.model_a_correct,
                    shadow_correct=row.shadow_correct,
                    sensor_b_correct=row.sensor_b_correct,
                )
            )

        # Calculate cumulative stats
        total_matches = sum(d.matches for d in daily_data)
        total_market_correct = sum(d.market_correct for d in daily_data)
        total_model_a_correct = sum(d.model_a_correct for d in daily_data)
        total_shadow_correct = sum(d.shadow_correct for d in daily_data)
        total_sensor_b_correct = sum(d.sensor_b_correct for d in daily_data)

        # Calculate days won for each model
        def count_days_won(model_correct_fn):
            count = 0
            for d in daily_data:
                model_val = model_correct_fn(d)
                max_val = max(d.market_correct, d.model_a_correct, d.shadow_correct, d.sensor_b_correct)
                if model_val == max_val and max_val > 0:
                    count += 1
            return count

        market_days_won = count_days_won(lambda d: d.market_correct)
        model_a_days_won = count_days_won(lambda d: d.model_a_correct)
        shadow_days_won = count_days_won(lambda d: d.shadow_correct)
        sensor_b_days_won = count_days_won(lambda d: d.sensor_b_correct)

        models = [
            ModelSummary(
                name="Market",
                accuracy=round((total_market_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_market_correct,
                total=total_matches,
                days_won=market_days_won,
            ),
            ModelSummary(
                name="Model A",
                accuracy=round((total_model_a_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_model_a_correct,
                total=total_matches,
                days_won=model_a_days_won,
            ),
            ModelSummary(
                name="Shadow",
                accuracy=round((total_shadow_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_shadow_correct,
                total=total_matches,
                days_won=shadow_days_won,
            ),
            ModelSummary(
                name="Sensor B",
                accuracy=round((total_sensor_b_correct / total_matches) * 100, 1) if total_matches > 0 else 0,
                correct=total_sensor_b_correct,
                total=total_matches,
                days_won=sensor_b_days_won,
            ),
        ]

        return ModelBenchmarkResponse(
            generated_at=datetime.utcnow().isoformat(),
            start_date="2026-01-17",
            total_matches=total_matches,
            daily_data=daily_data,
            models=models,
        )

    except Exception as e:
        logger.exception("Failed to fetch model benchmark data")
        raise HTTPException(status_code=500, detail=f"Failed to fetch benchmark: {str(e)}")
