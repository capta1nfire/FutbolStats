"""Team identity override resolution.

Handles rebranding cases where API-Football hasn't updated team names/logos
but we need to show the correct identity to users.

Usage:
    # For single team
    display = await get_team_display_info(session, external_id=1134, match_date=datetime(2026, 1, 17))
    # Returns: TeamDisplayInfo(name="Internacional de Bogotá", logo_url="...", is_override=True)

    # For batch (efficient - single query)
    overrides = await preload_team_overrides(session, external_ids=[1134, 119, 127])
    display = resolve_team_display(overrides, external_id=1134, match_date=datetime(2026, 1, 17))
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import TeamOverride

logger = logging.getLogger(__name__)

# In-memory cache for overrides (refreshed per request batch)
_overrides_cache: dict[int, list[TeamOverride]] = {}


@dataclass
class TeamDisplayInfo:
    """Team display information after override resolution."""
    name: str
    logo_url: Optional[str]
    is_override: bool = False
    original_name: Optional[str] = None  # Original name if override applied


async def preload_team_overrides(
    session: AsyncSession,
    external_ids: list[int],
    provider: str = "api_football",
) -> dict[int, list[TeamOverride]]:
    """
    Batch preload team overrides for a list of external team IDs.

    This is the efficient way to resolve overrides for multiple teams
    without N+1 queries.

    Args:
        session: Database session.
        external_ids: List of provider team IDs to load overrides for.
        provider: Data provider name (default: api_football).

    Returns:
        Dict mapping external_team_id -> list of TeamOverride (sorted by effective_from desc).
    """
    if not external_ids:
        return {}

    result = await session.execute(
        select(TeamOverride)
        .where(
            and_(
                TeamOverride.provider == provider,
                TeamOverride.external_team_id.in_(external_ids),
            )
        )
        .order_by(TeamOverride.external_team_id, TeamOverride.effective_from.desc())
    )
    overrides = result.scalars().all()

    # Group by external_team_id
    grouped: dict[int, list[TeamOverride]] = {}
    for override in overrides:
        if override.external_team_id not in grouped:
            grouped[override.external_team_id] = []
        grouped[override.external_team_id].append(override)

    return grouped


def resolve_team_display(
    overrides: dict[int, list[TeamOverride]],
    external_id: int,
    match_date: datetime,
    original_name: str,
    original_logo_url: Optional[str] = None,
) -> TeamDisplayInfo:
    """
    Resolve display name/logo for a team given preloaded overrides.

    Args:
        overrides: Preloaded overrides from preload_team_overrides().
        external_id: Provider team ID.
        match_date: Match date to check against effective_from/to.
        original_name: Original team name from provider.
        original_logo_url: Original logo URL from provider.

    Returns:
        TeamDisplayInfo with resolved name/logo.
    """
    team_overrides = overrides.get(external_id, [])

    for override in team_overrides:
        # Check if match_date falls within override validity
        if match_date >= override.effective_from:
            # Check effective_to if set
            if override.effective_to is None or match_date < override.effective_to:
                return TeamDisplayInfo(
                    name=override.display_name,
                    logo_url=override.display_logo_url or original_logo_url,
                    is_override=True,
                    original_name=original_name,
                )

    # No override applies - return original
    return TeamDisplayInfo(
        name=original_name,
        logo_url=original_logo_url,
        is_override=False,
    )


async def get_team_display_info(
    session: AsyncSession,
    external_id: int,
    match_date: datetime,
    original_name: str,
    original_logo_url: Optional[str] = None,
    provider: str = "api_football",
) -> TeamDisplayInfo:
    """
    Get display info for a single team (convenience function).

    For multiple teams, use preload_team_overrides() + resolve_team_display().

    Args:
        session: Database session.
        external_id: Provider team ID.
        match_date: Match date to check against effective_from/to.
        original_name: Original team name from provider.
        original_logo_url: Original logo URL from provider.
        provider: Data provider name.

    Returns:
        TeamDisplayInfo with resolved name/logo.
    """
    overrides = await preload_team_overrides(session, [external_id], provider)
    return resolve_team_display(overrides, external_id, match_date, original_name, original_logo_url)


def apply_team_overrides_to_match(
    overrides: dict[int, list[TeamOverride]],
    match_data: dict,
    match_date: datetime,
) -> dict:
    """
    Apply team overrides to a match dictionary in-place.

    Expects match_data to have:
    - home_team_external_id, home_team, home_team_logo (or similar keys)
    - away_team_external_id, away_team, away_team_logo (or similar keys)

    Args:
        overrides: Preloaded overrides from preload_team_overrides().
        match_data: Match dictionary to modify.
        match_date: Match date for override resolution.

    Returns:
        Modified match_data dict.
    """
    # Home team
    home_ext_id = match_data.get("home_team_external_id") or match_data.get("home_external_id")
    if home_ext_id:
        home_display = resolve_team_display(
            overrides,
            home_ext_id,
            match_date,
            match_data.get("home_team", "Unknown"),
            match_data.get("home_team_logo"),
        )
        if home_display.is_override:
            match_data["home_team"] = home_display.name
            if home_display.logo_url:
                match_data["home_team_logo"] = home_display.logo_url

    # Away team
    away_ext_id = match_data.get("away_team_external_id") or match_data.get("away_external_id")
    if away_ext_id:
        away_display = resolve_team_display(
            overrides,
            away_ext_id,
            match_date,
            match_data.get("away_team", "Unknown"),
            match_data.get("away_team_logo"),
        )
        if away_display.is_override:
            match_data["away_team"] = away_display.name
            if away_display.logo_url:
                match_data["away_team_logo"] = away_display.logo_url

    return match_data


async def apply_team_overrides_to_standings(
    session: AsyncSession,
    standings: list[dict],
    league_id: int,
    season: int,
    provider: str = "api_football",
) -> list[dict]:
    """
    Apply team identity overrides to standings data (batch, no N+1).

    Resolves team_name and team_logo for each entry based on effective_from date.

    Args:
        session: Database session.
        standings: List of standings dicts with team_id, team_name, team_logo.
        league_id: League ID to determine season date logic.
        season: Season year (e.g., 2026).
        provider: Data provider name.

    Returns:
        Modified standings list with overrides applied.
    """
    if not standings:
        return standings

    # Determine as_of date based on league type
    # Calendar-year leagues (LATAM/MLS): season starts Jan 1
    # European leagues: season starts Jul 1
    # Using the same list as main.py for consistency
    CALENDAR_YEAR_SEASON_LEAGUES = {
        71,   # Brazil - Serie A
        128,  # Argentina - Primera División
        239,  # Colombia Primera A
        242,  # Ecuador Liga Pro
        253,  # USA - MLS (calendar)
        265,  # Chile Primera Division
        268,  # Uruguay Primera - Apertura
        270,  # Uruguay Primera - Clausura
        281,  # Peru Primera Division
        299,  # Venezuela Primera Division
        344,  # Bolivia Primera Division
    }

    is_calendar_year = league_id in CALENDAR_YEAR_SEASON_LEAGUES

    if is_calendar_year:
        # Calendar year (LATAM): 2026 season starts Jan 2026
        as_of = datetime(season, 1, 1)
    else:
        # European: 2025-26 season means it started Jul 2025
        as_of = datetime(season, 7, 1)

    # Extract all team_ids for batch preload
    team_ids = [s.get("team_id") for s in standings if s.get("team_id")]
    if not team_ids:
        return standings

    # Preload overrides in batch (no N+1)
    overrides = await preload_team_overrides(session, team_ids, provider)

    if not overrides:
        return standings

    # Apply overrides to each standing entry
    for entry in standings:
        team_id = entry.get("team_id")
        if not team_id:
            continue

        team_name = entry.get("team_name", "Unknown")
        team_logo = entry.get("team_logo")

        display = resolve_team_display(
            overrides,
            team_id,
            as_of,
            team_name,
            team_logo,
        )

        if display.is_override:
            entry["team_name"] = display.name
            if display.logo_url:
                entry["team_logo"] = display.logo_url
            # Optional: mark as overridden for debugging
            entry["_identity_override"] = True

    return standings


async def enrich_standings_with_display_names(
    session: AsyncSession,
    standings: list[dict],
) -> list[dict]:
    """
    Enrich standings with display_name for use_short_names toggle.

    For each team, calculates:
        COALESCE(override.short_name, wikidata.short_name, team.name) AS display_name

    This enables the frontend to show shortened team names when the league's
    use_short_names setting is enabled.

    Args:
        session: Database session.
        standings: List of standings dicts with team_id (internal, not external).

    Returns:
        Modified standings list with display_name added to each entry.
    """
    if not standings:
        return standings

    # Extract team_ids (internal IDs after translation)
    team_ids = [s.get("team_id") for s in standings if s.get("team_id")]
    if not team_ids:
        return standings

    # Query display_name for all teams in one query
    from sqlalchemy import text
    result = await session.execute(
        text("""
            SELECT
                t.id AS team_id,
                COALESCE(
                    teo.short_name,
                    twe.short_name,
                    t.name
                ) AS display_name
            FROM teams t
            LEFT JOIN team_enrichment_overrides teo ON t.id = teo.team_id
            LEFT JOIN team_wikidata_enrichment twe ON t.id = twe.team_id
            WHERE t.id = ANY(:team_ids)
        """),
        {"team_ids": team_ids}
    )
    rows = result.fetchall()

    # Build lookup map
    display_name_map = {row.team_id: row.display_name for row in rows}

    # Enrich standings
    for entry in standings:
        team_id = entry.get("team_id")
        if team_id and team_id in display_name_map:
            entry["display_name"] = display_name_map[team_id]
        else:
            # Fallback to team_name if no display_name found
            entry["display_name"] = entry.get("team_name", "Unknown")

    return standings
