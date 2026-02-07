"""
Player & Manager ingestion jobs (Phase 1 MVP).

Jobs:
  sync_injuries  — Fetch injuries from API-Football for all tracked leagues
  sync_managers  — Fetch coaches, detect manager changes, update history

Reference: docs/PLAYERS_MANAGERS_PROPOSAL.md v2.1
PIT policy: captured_at < kickoff (strict). Backfills are NOT PIT-safe for training.
"""

import json
import logging
from datetime import datetime, date
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl.api_football import APIFootballProvider
from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES

logger = logging.getLogger(__name__)

# Reuse the same 28 leagues tracked by Sofascore for injuries
TRACKED_LEAGUES = sorted(SOFASCORE_SUPPORTED_LEAGUES)


def _current_season(league_id: int) -> int:
    """Determine API-Football season year for a league.

    European leagues: season = year when season starts (e.g. 2024 for 2024/25).
    South American leagues: calendar year.
    For simplicity, use current year. If league hasn't started yet, API returns
    data for the most recent season.
    """
    now = datetime.utcnow()
    month = now.month
    # European leagues start Aug/Sep, so before Aug use previous year
    # South American leagues are calendar year
    # UEFA competitions follow European schedule
    european = {39, 40, 140, 135, 78, 61, 94, 88, 144, 203, 2, 3, 848}
    if league_id in european and month < 8:
        return now.year - 1
    return now.year


# ---------------------------------------------------------------------------
# Job 1: sync_injuries
# ---------------------------------------------------------------------------

async def sync_injuries(
    session: AsyncSession,
    leagues: Optional[list[int]] = None,
    limit_leagues: int = 0,
) -> dict:
    """
    Fetch injuries from API-Football for all tracked leagues.

    One API call per league (returns all injuries for current season).
    Resolves team_id and match_id internally via external_id lookups.

    Args:
        session: Async DB session
        leagues: Override league list (default: TRACKED_LEAGUES)
        limit_leagues: If >0, only process this many leagues (for testing)

    Returns:
        Metrics dict with inserted, updated, errors, etc.
    """
    metrics = {
        "leagues_attempted": 0,
        "leagues_ok": 0,
        "injuries_inserted": 0,
        "injuries_updated": 0,
        "errors": 0,
        "error_details": [],
    }

    target_leagues = leagues or TRACKED_LEAGUES
    if limit_leagues > 0:
        target_leagues = target_leagues[:limit_leagues]

    provider = APIFootballProvider()

    # Pre-load team external_id → internal id mapping
    team_map = await _load_team_map(session)
    # Pre-load match external_id → internal id mapping (recent matches only)
    match_map = await _load_match_map(session)

    try:
        for league_id in target_leagues:
            metrics["leagues_attempted"] += 1
            season = _current_season(league_id)

            try:
                data = await provider._rate_limited_request(
                    "injuries",
                    {"league": league_id, "season": season},
                    entity="injury",
                )
                injuries = data.get("response", [])

                if not injuries:
                    logger.info(f"[INJURIES_SYNC] league={league_id} season={season}: 0 injuries returned")
                    metrics["leagues_ok"] += 1
                    continue

                # Savepoint per league: isolate DB errors so one bad league
                # doesn't poison the transaction for subsequent leagues
                async with session.begin_nested():
                    inserted, updated = await _upsert_injuries(
                        session, injuries, league_id, season, team_map, match_map
                    )
                metrics["injuries_inserted"] += inserted
                metrics["injuries_updated"] += updated
                metrics["leagues_ok"] += 1

                logger.info(
                    f"[INJURIES_SYNC] league={league_id}: "
                    f"total={len(injuries)}, inserted={inserted}, updated={updated}"
                )

            except Exception as e:
                metrics["errors"] += 1
                metrics["error_details"].append(f"league={league_id}: {e}")
                logger.error(f"[INJURIES_SYNC] Error league={league_id}: {e}")

        await session.commit()

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[INJURIES_SYNC] Fatal error: {e}", exc_info=True)
        try:
            await session.rollback()
        except Exception:
            pass
    finally:
        await provider.close()

    logger.info(
        f"[INJURIES_SYNC] Complete: leagues={metrics['leagues_ok']}/{metrics['leagues_attempted']}, "
        f"inserted={metrics['injuries_inserted']}, updated={metrics['injuries_updated']}, "
        f"errors={metrics['errors']}"
    )
    return metrics


async def _upsert_injuries(
    session: AsyncSession,
    injuries: list[dict],
    league_id: int,
    season: int,
    team_map: dict[int, int],
    match_map: dict[int, int],
) -> tuple[int, int]:
    """Parse and upsert injury records. Returns (inserted, updated)."""
    inserted = 0
    updated = 0

    for entry in injuries:
        player = entry.get("player", {})
        team = entry.get("team", {})
        fixture = entry.get("fixture", {})

        player_ext_id = player.get("id")
        fixture_ext_id = fixture.get("id")
        if not player_ext_id or not fixture_ext_id:
            continue

        team_ext_id = team.get("id")
        team_internal_id = team_map.get(team_ext_id)
        match_internal_id = match_map.get(fixture_ext_id)

        fixture_date_raw = fixture.get("date")
        fixture_date = None
        if fixture_date_raw:
            try:
                dt = datetime.fromisoformat(fixture_date_raw.replace("Z", "+00:00"))
                fixture_date = dt.replace(tzinfo=None)  # DB uses naive TIMESTAMP (UTC assumed)
            except (ValueError, AttributeError):
                pass

        player_name = player.get("name") or f"Player#{player_ext_id}"

        result = await session.execute(
            text("""
                INSERT INTO player_injuries (
                    player_external_id, player_name, team_id, league_id, season,
                    fixture_external_id, match_id, injury_type, injury_reason,
                    fixture_date, raw_json
                ) VALUES (
                    :player_ext_id, :player_name, :team_id, :league_id, :season,
                    :fixture_ext_id, :match_id, :injury_type, :injury_reason,
                    :fixture_date, :raw_json
                )
                ON CONFLICT (player_external_id, fixture_external_id)
                DO UPDATE SET
                    injury_type = EXCLUDED.injury_type,
                    injury_reason = EXCLUDED.injury_reason,
                    team_id = COALESCE(EXCLUDED.team_id, player_injuries.team_id),
                    match_id = COALESCE(EXCLUDED.match_id, player_injuries.match_id),
                    raw_json = EXCLUDED.raw_json
                RETURNING (xmax = 0) AS is_insert
            """),
            {
                "player_ext_id": player_ext_id,
                "player_name": player_name,
                "team_id": team_internal_id,
                "league_id": league_id,
                "season": season,
                "fixture_ext_id": fixture_ext_id,
                "match_id": match_internal_id,
                "injury_type": player.get("type") or "Unknown",
                "injury_reason": player.get("reason"),
                "fixture_date": fixture_date,
                "raw_json": json.dumps(entry),
            },
        )
        row = result.fetchone()
        if row and row.is_insert:
            inserted += 1
        else:
            updated += 1

    return inserted, updated


# ---------------------------------------------------------------------------
# Job 2: sync_managers
# ---------------------------------------------------------------------------

async def sync_managers(
    session: AsyncSession,
    leagues: Optional[list[int]] = None,
    batch_size: int = 50,
) -> dict:
    """
    Fetch coaches for all active teams, detect manager changes.

    For each team:
      1. GET /coachs?team={id} → current coach + career history
      2. UPSERT in managers table (catalog)
      3. Compare with latest team_manager_history entry
      4. If different → close old stint, open new stint

    Args:
        session: Async DB session
        leagues: Override league list (default: TRACKED_LEAGUES)
        batch_size: Teams per batch (with 1s delay between requests)

    Returns:
        Metrics dict
    """
    metrics = {
        "teams_attempted": 0,
        "teams_ok": 0,
        "managers_upserted": 0,
        "changes_detected": 0,
        "errors": 0,
        "error_details": [],
    }

    target_leagues = leagues or TRACKED_LEAGUES

    # Get all active teams for these leagues
    active_teams = await _get_active_teams(session, target_leagues)
    logger.info(f"[MANAGER_SYNC] Found {len(active_teams)} active teams across {len(target_leagues)} leagues")

    provider = APIFootballProvider()

    try:
        for team in active_teams:
            metrics["teams_attempted"] += 1
            team_id = team["id"]
            team_ext_id = team["external_id"]
            team_name = team["name"]

            try:
                data = await provider._rate_limited_request(
                    "coachs",
                    {"team": team_ext_id},
                    entity="coach",
                )
                coaches = data.get("response", [])

                if not coaches:
                    logger.debug(f"[MANAGER_SYNC] No coaches for {team_name} (ext={team_ext_id})")
                    metrics["teams_ok"] += 1
                    continue

                # The current coach is the one whose team.id matches and career has end=null
                current_coach = _find_current_coach(coaches, team_ext_id)
                if not current_coach:
                    logger.warning(f"[MANAGER_SYNC] No current coach found for {team_name}")
                    metrics["teams_ok"] += 1
                    continue

                # Savepoint per team: isolate DB errors so one bad team
                # doesn't poison the transaction for subsequent teams
                async with session.begin_nested():
                    # Upsert manager catalog
                    await _upsert_manager(session, current_coach)
                    metrics["managers_upserted"] += 1

                    # Check for change
                    changed = await _detect_and_record_change(
                        session, team_id, team_ext_id, team_name, current_coach
                    )
                    if changed:
                        metrics["changes_detected"] += 1

                metrics["teams_ok"] += 1

            except Exception as e:
                metrics["errors"] += 1
                metrics["error_details"].append(f"team={team_name}: {e}")
                logger.error(f"[MANAGER_SYNC] Error for {team_name}: {e}")

        await session.commit()

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"[MANAGER_SYNC] Fatal error: {e}", exc_info=True)
        try:
            await session.rollback()
        except Exception:
            pass
    finally:
        await provider.close()

    logger.info(
        f"[MANAGER_SYNC] Complete: teams={metrics['teams_ok']}/{metrics['teams_attempted']}, "
        f"upserted={metrics['managers_upserted']}, changes={metrics['changes_detected']}, "
        f"errors={metrics['errors']}"
    )
    return metrics


def _find_current_coach(coaches: list[dict], team_ext_id: int) -> Optional[dict]:
    """Find the coach currently managing the given team."""
    for coach in coaches:
        # Check if coach's current team matches
        coach_team = coach.get("team", {})
        if coach_team.get("id") == team_ext_id:
            return coach
        # Also check career for active stint
        for stint in coach.get("career", []):
            if stint.get("team", {}).get("id") == team_ext_id and stint.get("end") is None:
                return coach
    # Fallback: first coach in response (API usually returns current first)
    return coaches[0] if coaches else None


async def _upsert_manager(session: AsyncSession, coach: dict) -> None:
    """Upsert a manager into the catalog table."""
    birth = coach.get("birth", {})
    birth_date_raw = birth.get("date") if birth else None
    birth_date = None
    if birth_date_raw:
        try:
            birth_date = date.fromisoformat(birth_date_raw)
        except (ValueError, AttributeError):
            pass

    await session.execute(
        text("""
            INSERT INTO managers (external_id, name, first_name, last_name,
                                  birth_date, nationality, photo_url, career, raw_json, updated_at)
            VALUES (:ext_id, :name, :first_name, :last_name,
                    :birth_date, :nationality, :photo_url, :career, :raw_json, NOW())
            ON CONFLICT (external_id) DO UPDATE SET
                name = EXCLUDED.name,
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                career = EXCLUDED.career,
                raw_json = EXCLUDED.raw_json,
                updated_at = NOW()
        """),
        {
            "ext_id": coach.get("id"),
            "name": coach.get("name", "Unknown"),
            "first_name": coach.get("firstname"),
            "last_name": coach.get("lastname"),
            "birth_date": birth_date,
            "nationality": coach.get("nationality"),
            "photo_url": coach.get("photo"),
            "career": json.dumps(coach.get("career", [])),
            "raw_json": json.dumps(coach),
        },
    )


async def _detect_and_record_change(
    session: AsyncSession,
    team_id: int,
    team_ext_id: int,
    team_name: str,
    coach: dict,
) -> bool:
    """
    Compare current coach with latest team_manager_history entry.
    If different, close old stint and open new one.

    Uses career dates from API (not detection date) for start_date/end_date.
    Returns True if a change was detected.
    """
    manager_ext_id = coach.get("id")
    manager_name = coach.get("name", "Unknown")

    # Find the start_date from career (parse string to date for asyncpg)
    start_date = None
    for stint in coach.get("career", []):
        if stint.get("team", {}).get("id") == team_ext_id and stint.get("end") is None:
            raw = stint.get("start")
            if raw:
                try:
                    start_date = date.fromisoformat(raw)
                except (ValueError, AttributeError):
                    pass
            break

    if not start_date:
        start_date = date.today()

    # Get current manager for this team
    result = await session.execute(
        text("""
            SELECT id, manager_external_id, manager_name, start_date
            FROM team_manager_history
            WHERE team_id = :team_id AND end_date IS NULL
            ORDER BY start_date DESC
            LIMIT 1
        """),
        {"team_id": team_id},
    )
    current = result.fetchone()

    if current and current.manager_external_id == manager_ext_id:
        # Same manager, no change
        return False

    if current:
        # Close the old stint. Use the new manager's start_date as end_date
        # (the old manager ended when the new one started)
        await session.execute(
            text("""
                UPDATE team_manager_history
                SET end_date = :end_date
                WHERE id = :id
            """),
            {"id": current.id, "end_date": start_date},
        )
        logger.info(
            f"[MANAGER_CHANGE] {team_name}: {current.manager_name} → {manager_name} "
            f"(started {start_date})"
        )

    # Insert new stint (or first stint if no history)
    await session.execute(
        text("""
            INSERT INTO team_manager_history
                (team_id, manager_external_id, manager_name, start_date,
                 team_external_id, source, detected_at)
            VALUES
                (:team_id, :manager_ext_id, :manager_name, :start_date,
                 :team_ext_id, 'api-football', NOW())
            ON CONFLICT (team_id, manager_external_id, start_date) DO NOTHING
        """),
        {
            "team_id": team_id,
            "manager_ext_id": manager_ext_id,
            "manager_name": manager_name,
            "start_date": start_date,
            "team_ext_id": team_ext_id,
        },
    )

    return current is not None  # True only if we replaced someone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _load_team_map(session: AsyncSession) -> dict[int, int]:
    """Load external_id → internal id mapping for all teams."""
    result = await session.execute(text("SELECT id, external_id FROM teams WHERE external_id IS NOT NULL"))
    return {row.external_id: row.id for row in result.fetchall()}


async def _load_match_map(session: AsyncSession) -> dict[int, int]:
    """Load external_id → internal id for recent matches (last 90 days)."""
    result = await session.execute(
        text("""
            SELECT id, external_id FROM matches
            WHERE external_id IS NOT NULL
              AND date >= NOW() - INTERVAL '90 days'
        """)
    )
    return {row.external_id: row.id for row in result.fetchall()}


async def _get_active_teams(session: AsyncSession, league_ids: list[int]) -> list[dict]:
    """Get distinct teams that have played in the given leagues recently."""
    result = await session.execute(
        text("""
            SELECT DISTINCT t.id, t.external_id, t.name
            FROM teams t
            JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            WHERE m.league_id = ANY(:league_ids)
              AND m.date >= NOW() - INTERVAL '180 days'
              AND t.external_id IS NOT NULL
            ORDER BY t.name
        """),
        {"league_ids": league_ids},
    )
    return [{"id": row.id, "external_id": row.external_id, "name": row.name} for row in result.fetchall()]
