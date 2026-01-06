#!/usr/bin/env python3
"""
Backfill narrative insights for existing post_match_audits.

This script re-generates narrative insights for audits that don't have them yet.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, selectinload

from app.models import Match, Prediction, PredictionOutcome, PostMatchAudit
from app.audit.service import PostMatchAuditService
from app.etl.api_football import APIFootballProvider

# Get database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


async def backfill_insights():
    """Backfill narrative insights for existing audits."""
    db_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Find audits without narrative insights
        result = await session.execute(
            select(PostMatchAudit, PredictionOutcome, Prediction, Match)
            .join(PredictionOutcome, PostMatchAudit.outcome_id == PredictionOutcome.id)
            .join(Prediction, PredictionOutcome.prediction_id == Prediction.id)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .options(
                selectinload(Match.home_team),
                selectinload(Match.away_team),
            )
            .where(PostMatchAudit.narrative_insights.is_(None))
            .limit(50)  # Process in batches
        )
        rows = result.all()

        print(f"Found {len(rows)} audits to backfill")

        audit_service = PostMatchAuditService(session)

        updated = 0
        for audit, outcome, prediction, match in rows:
            try:
                # Fetch stats for this match
                stats = await audit_service.provider.get_fixture_statistics(match.external_id)

                # Generate narrative insights
                narrative_result = audit_service.generate_narrative_insights(
                    prediction=prediction,
                    actual_result=outcome.actual_result,
                    home_goals=match.home_goals,
                    away_goals=match.away_goals,
                    stats=stats or {},
                    home_team_name=match.home_team.name if match.home_team else "Local",
                    away_team_name=match.away_team.name if match.away_team else "Visitante",
                    home_position=None,
                    away_position=None,
                )

                # Update the audit
                audit.narrative_insights = narrative_result.get("insights")
                audit.momentum_analysis = narrative_result.get("momentum_analysis")
                updated += 1

                insights_count = len(narrative_result.get("insights", []))
                print(f"  Updated match {match.id}: {insights_count} insights")

            except Exception as e:
                print(f"  Error processing match {match.id}: {e}")
                continue

        await session.commit()
        await audit_service.close()

        print(f"\nBackfill complete: {updated} audits updated")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(backfill_insights())
