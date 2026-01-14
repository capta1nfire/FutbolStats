"""
Daily aggregates refresh job.

Computes league baselines and team profiles for all active leagues.
Runs as part of the scheduler or can be triggered manually.
"""

import logging
from datetime import datetime, date
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Match, LeagueSeasonBaseline, LeagueTeamProfile
from app.aggregates.service import AggregatesService

logger = logging.getLogger(__name__)


async def refresh_all_aggregates(
    session: AsyncSession,
    as_of_date: Optional[date] = None,
) -> dict:
    """
    Refresh baselines and profiles for all leagues with recent matches.

    Args:
        session: Database session
        as_of_date: Date to compute stats up to (default: today)

    Returns:
        Dict with refresh metrics
    """
    import time
    start_time = time.time()

    if as_of_date is None:
        as_of_date = datetime.utcnow().date()

    metrics = {
        "leagues_processed": 0,
        "baselines_created": 0,
        "profiles_created": 0,
        "errors": [],
        "started_at": datetime.utcnow().isoformat(),
    }

    service = AggregatesService(session)

    # Find all league/season combinations with finished matches
    result = await session.execute(
        select(Match.league_id, Match.season)
        .where(
            and_(
                Match.status.in_(["FT", "AET", "PEN"]),
                Match.home_goals.isnot(None),
            )
        )
        .group_by(Match.league_id, Match.season)
        .having(func.count(Match.id) >= 5)  # Only process with min matches
    )
    league_seasons = result.all()

    logger.info(f"[AGGREGATES] Found {len(league_seasons)} league/season combinations to process")

    for league_id, season in league_seasons:
        try:
            # Compute baseline
            baseline = await service.compute_league_baseline(league_id, season, as_of_date)
            if baseline:
                metrics["baselines_created"] += 1

            # Compute team profiles
            profiles = await service.compute_team_profiles(league_id, season, as_of_date)
            metrics["profiles_created"] += len(profiles)

            metrics["leagues_processed"] += 1

        except Exception as e:
            error_msg = f"League {league_id} season {season}: {str(e)}"
            logger.error(f"[AGGREGATES] Error processing: {error_msg}")
            metrics["errors"].append(error_msg)

    # Commit all changes
    await session.commit()

    duration_ms = (time.time() - start_time) * 1000
    metrics["completed_at"] = datetime.utcnow().isoformat()
    metrics["duration_ms"] = duration_ms

    # Get final counts for telemetry
    status = await get_aggregates_status(session)

    # Calculate min_sample_ok percentage
    min_sample_ok_pct = 0.0
    if status["profiles_count"] > 0:
        # Query count of profiles with min_sample_ok=True
        min_ok_result = await session.execute(
            select(func.count(LeagueTeamProfile.id))
            .where(LeagueTeamProfile.min_sample_ok == True)
        )
        min_ok_count = min_ok_result.scalar() or 0
        min_sample_ok_pct = (min_ok_count / status["profiles_count"]) * 100

    # Emit telemetry
    try:
        from app.telemetry.metrics import record_aggregates_refresh
        record_aggregates_refresh(
            status="ok" if not metrics["errors"] else "error",
            duration_ms=duration_ms,
            baselines_count=status["baselines_count"],
            profiles_count=status["profiles_count"],
            leagues_count=status["leagues_with_baselines"],
            min_sample_ok_pct=min_sample_ok_pct,
        )
    except Exception as e:
        logger.debug(f"[AGGREGATES] Failed to emit telemetry: {e}")

    logger.info(
        f"[AGGREGATES] Refresh complete: {metrics['leagues_processed']} leagues, "
        f"{metrics['baselines_created']} baselines, {metrics['profiles_created']} profiles, "
        f"duration={duration_ms:.0f}ms"
    )

    return metrics


async def refresh_single_league(
    session: AsyncSession,
    league_id: int,
    season: int,
    as_of_date: Optional[date] = None,
) -> dict:
    """
    Refresh aggregates for a single league/season.

    Args:
        session: Database session
        league_id: API-Football league ID
        season: Season year
        as_of_date: Date to compute stats up to (default: today)

    Returns:
        Dict with refresh results
    """
    if as_of_date is None:
        as_of_date = datetime.utcnow().date()

    service = AggregatesService(session)

    result = {
        "league_id": league_id,
        "season": season,
        "as_of_date": as_of_date.isoformat(),
        "baseline": None,
        "profiles_count": 0,
    }

    # Compute baseline
    baseline = await service.compute_league_baseline(league_id, season, as_of_date)
    if baseline:
        result["baseline"] = {
            "sample_n_matches": baseline.sample_n_matches,
            "goals_avg": baseline.goals_avg_per_match,
            "over_2_5_pct": baseline.over_2_5_pct,
            "btts_yes_pct": baseline.btts_yes_pct,
        }

    # Compute profiles
    profiles = await service.compute_team_profiles(league_id, season, as_of_date)
    result["profiles_count"] = len(profiles)

    await session.commit()

    return result


async def get_aggregates_status(session: AsyncSession) -> dict:
    """
    Get current status of aggregates tables.

    Returns counts and latest computation dates.
    """
    # Count baselines
    baselines_result = await session.execute(
        select(func.count(LeagueSeasonBaseline.id))
    )
    baselines_count = baselines_result.scalar() or 0

    # Count profiles
    profiles_result = await session.execute(
        select(func.count(LeagueTeamProfile.id))
    )
    profiles_count = profiles_result.scalar() or 0

    # Get latest computation date
    latest_baseline = await session.execute(
        select(LeagueSeasonBaseline.last_computed_at)
        .order_by(LeagueSeasonBaseline.last_computed_at.desc())
        .limit(1)
    )
    latest_baseline_date = latest_baseline.scalar()

    latest_profile = await session.execute(
        select(LeagueTeamProfile.last_computed_at)
        .order_by(LeagueTeamProfile.last_computed_at.desc())
        .limit(1)
    )
    latest_profile_date = latest_profile.scalar()

    # Get distinct leagues with baselines
    leagues_result = await session.execute(
        select(func.count(func.distinct(LeagueSeasonBaseline.league_id)))
    )
    leagues_count = leagues_result.scalar() or 0

    return {
        "baselines_count": baselines_count,
        "profiles_count": profiles_count,
        "leagues_with_baselines": leagues_count,
        "latest_baseline_at": latest_baseline_date.isoformat() if latest_baseline_date else None,
        "latest_profile_at": latest_profile_date.isoformat() if latest_profile_date else None,
    }
