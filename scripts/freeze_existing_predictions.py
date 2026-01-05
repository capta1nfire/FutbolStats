#!/usr/bin/env python3
"""
One-time script to freeze existing predictions for matches that have already started/finished.

This backfills the frozen fields for predictions that existed before the freeze feature was added.
"""

import asyncio
import os
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

# Get database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


async def freeze_existing_predictions():
    """Freeze predictions for matches that have started or finished."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    # Import models
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.models import Match, Prediction

    # Convert postgres:// to postgresql+asyncpg://
    db_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        now = datetime.utcnow()

        # Find predictions that should be frozen
        result = await session.execute(
            select(Prediction)
            .options(selectinload(Prediction.match))
            .where(
                and_(
                    Prediction.is_frozen == False,  # noqa: E712
                )
            )
        )
        predictions = result.scalars().all()

        frozen_count = 0
        skipped_count = 0

        for pred in predictions:
            match = pred.match
            if not match:
                skipped_count += 1
                continue

            # Check if match has started or is in the past
            should_freeze = (
                match.status != "NS" or  # Match has started/finished
                match.date < now  # Match date is in the past
            )

            if not should_freeze:
                skipped_count += 1
                continue

            # Freeze the prediction
            pred.is_frozen = True
            pred.frozen_at = now

            # Capture bookmaker odds at freeze time
            pred.frozen_odds_home = match.odds_home
            pred.frozen_odds_draw = match.odds_draw
            pred.frozen_odds_away = match.odds_away

            # Calculate and freeze EV values
            if match.odds_home and pred.home_prob > 0:
                pred.frozen_ev_home = (pred.home_prob * match.odds_home) - 1
            if match.odds_draw and pred.draw_prob > 0:
                pred.frozen_ev_draw = (pred.draw_prob * match.odds_draw) - 1
            if match.odds_away and pred.away_prob > 0:
                pred.frozen_ev_away = (pred.away_prob * match.odds_away) - 1

            # Calculate and freeze confidence tier
            max_prob = max(pred.home_prob, pred.draw_prob, pred.away_prob)
            if max_prob >= 0.50:
                pred.frozen_confidence_tier = "gold"
            elif max_prob >= 0.40:
                pred.frozen_confidence_tier = "silver"
            else:
                pred.frozen_confidence_tier = "copper"

            # Calculate and freeze value bets
            value_bets = []
            ev_threshold = 0.05  # 5% EV minimum

            if pred.frozen_ev_home and pred.frozen_ev_home >= ev_threshold:
                value_bets.append({
                    "outcome": "home",
                    "odds": match.odds_home,
                    "model_prob": pred.home_prob,
                    "ev": pred.frozen_ev_home,
                })
            if pred.frozen_ev_draw and pred.frozen_ev_draw >= ev_threshold:
                value_bets.append({
                    "outcome": "draw",
                    "odds": match.odds_draw,
                    "model_prob": pred.draw_prob,
                    "ev": pred.frozen_ev_draw,
                })
            if pred.frozen_ev_away and pred.frozen_ev_away >= ev_threshold:
                value_bets.append({
                    "outcome": "away",
                    "odds": match.odds_away,
                    "model_prob": pred.away_prob,
                    "ev": pred.frozen_ev_away,
                })

            pred.frozen_value_bets = value_bets if value_bets else None

            frozen_count += 1
            print(f"  Frozen prediction {pred.id} for match {match.id} ({match.status})")

        await session.commit()

        print(f"\nFrozen {frozen_count} predictions")
        print(f"Skipped {skipped_count} predictions (NS matches or no match data)")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(freeze_existing_predictions())
