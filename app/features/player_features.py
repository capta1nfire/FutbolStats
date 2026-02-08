"""
Player & Manager features (Phase 1 MVP).

PIT-strict: All queries filter by captured_at/detected_at < match_kickoff.
Backfilled data is NOT PIT-safe for training (only operational/coverage).

Reference: docs/PLAYERS_MANAGERS_PROPOSAL.md v2.1 §8
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def get_team_absences(
    session: AsyncSession,
    team_id: int,
    fixture_external_id: Optional[int],
    match_kickoff: datetime,
) -> dict:
    """
    Count known absences for a team in a specific fixture (PIT-strict).

    Args:
        session: Async DB session
        team_id: Internal team ID
        fixture_external_id: API-Football fixture ID (for exact match)
        match_kickoff: Kickoff timestamp (PIT boundary)

    Returns:
        n_missing: int       — injury_type = 'Missing Fixture'
        n_doubtful: int      — injury_type IN ('Questionable', 'Doubtful')
        _has_injury_data: bool — internal flag for missing-data detection
    """
    if not fixture_external_id:
        return {"n_missing": 0, "n_doubtful": 0, "_has_injury_data": False}

    result = await session.execute(
        text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE injury_type = 'Missing Fixture') AS n_missing,
                COUNT(*) FILTER (WHERE injury_type IN ('Questionable', 'Doubtful')) AS n_doubtful
            FROM player_injuries
            WHERE team_id = :team_id
              AND fixture_external_id = :fixture_ext_id
              AND captured_at < :match_kickoff
        """),
        {
            "team_id": team_id,
            "fixture_ext_id": fixture_external_id,
            "match_kickoff": match_kickoff,
        },
    )
    row = result.fetchone()
    if row:
        return {
            "n_missing": row.n_missing or 0,
            "n_doubtful": row.n_doubtful or 0,
            "_has_injury_data": (row.total or 0) > 0,
        }
    return {"n_missing": 0, "n_doubtful": 0, "_has_injury_data": False}


async def get_manager_context(
    session: AsyncSession,
    team_id: int,
    match_kickoff: datetime,
) -> dict:
    """
    Get manager context for a team at a specific point in time (PIT-strict).

    Args:
        session: Async DB session
        team_id: Internal team ID
        match_kickoff: Kickoff timestamp (PIT boundary, used as datetime not date)

    Returns:
        manager_tenure_days: int   — days since start_date
        is_new_manager: int        — 1 if tenure < 60 days, 0 otherwise
    """
    result = await session.execute(
        text("""
            SELECT manager_external_id, manager_name, start_date,
                   (CAST(:match_kickoff AS date) - start_date) AS tenure_days
            FROM team_manager_history
            WHERE team_id = :team_id
              AND start_date <= CAST(:match_kickoff AS date)
              AND (end_date IS NULL OR end_date > CAST(:match_kickoff AS date))
              AND detected_at < :match_kickoff
            ORDER BY start_date DESC
            LIMIT 1
        """),
        {
            "team_id": team_id,
            "match_kickoff": match_kickoff,
        },
    )
    row = result.fetchone()
    if row:
        tenure = row.tenure_days if row.tenure_days is not None else 365
        return {
            "manager_tenure_days": tenure,
            "is_new_manager": 1 if tenure < 60 else 0,
            "_has_manager_data": True,
        }
    # No manager data → assume stable manager (default imputation)
    return {
        "manager_tenure_days": 365,
        "is_new_manager": 0,
        "_has_manager_data": False,
    }


async def get_player_manager_features(
    session: AsyncSession,
    match_id: int,
    match_external_id: Optional[int],
    home_team_id: int,
    away_team_id: int,
    match_kickoff: datetime,
) -> dict:
    """
    Get all player/manager features for a match (entry point).

    Returns 9 features:
        home_n_missing, away_n_missing,
        home_n_doubtful, away_n_doubtful,
        home_manager_tenure_days, away_manager_tenure_days,
        home_is_new_manager, away_is_new_manager,
        player_manager_missing (flag: 1 if NO rows found in DB)
    """
    try:
        home_absences = await get_team_absences(
            session, home_team_id, match_external_id, match_kickoff
        )
        away_absences = await get_team_absences(
            session, away_team_id, match_external_id, match_kickoff
        )
        home_manager = await get_manager_context(session, home_team_id, match_kickoff)
        away_manager = await get_manager_context(session, away_team_id, match_kickoff)

        # Missing flag: 1 if we have NO data at all (no injury rows AND no manager rows)
        has_any_data = (
            home_absences["_has_injury_data"]
            or away_absences["_has_injury_data"]
            or home_manager["_has_manager_data"]
            or away_manager["_has_manager_data"]
        )

        return {
            "home_n_missing": home_absences["n_missing"],
            "away_n_missing": away_absences["n_missing"],
            "home_n_doubtful": home_absences["n_doubtful"],
            "away_n_doubtful": away_absences["n_doubtful"],
            "home_manager_tenure_days": home_manager["manager_tenure_days"],
            "away_manager_tenure_days": away_manager["manager_tenure_days"],
            "home_is_new_manager": home_manager["is_new_manager"],
            "away_is_new_manager": away_manager["is_new_manager"],
            "player_manager_missing": 0 if has_any_data else 1,
        }

    except Exception as e:
        logger.warning(
            f"Player/manager features failed for match {match_id}: {e}",
            exc_info=True,
        )
        # Avoid transaction poisoning → prevents InFailedSQLTransactionError cascade
        try:
            await session.rollback()
        except Exception:
            pass
        return get_player_manager_defaults()


def get_player_manager_defaults() -> dict:
    """Default values when player/manager data is unavailable."""
    return {
        "home_n_missing": 0,
        "away_n_missing": 0,
        "home_n_doubtful": 0,
        "away_n_doubtful": 0,
        "home_manager_tenure_days": 365,
        "away_manager_tenure_days": 365,
        "home_is_new_manager": 0,
        "away_is_new_manager": 0,
        "player_manager_missing": 1,
    }
