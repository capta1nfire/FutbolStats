"""
Admin Leagues Seed/Sync Module

Provides idempotent seed/sync functionality for admin_leagues table.
- Seeds from COMPETITIONS dict with source='seed', is_active=true
- Discovers observed leagues from matches with source='observed', is_active=false
- Creates paired league groups for Apertura/Clausura
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.etl.competitions import COMPETITIONS, Competition

logger = logging.getLogger(__name__)

# Country mapping for leagues (derived from teams in matches)
# For international competitions, country is None
LEAGUE_COUNTRIES = {
    # Top 5 European
    39: "England",      # Premier League
    140: "Spain",       # La Liga
    135: "Italy",       # Serie A
    78: "Germany",      # Bundesliga
    61: "France",       # Ligue 1
    # Secondary European
    40: "England",      # Championship
    88: "Netherlands",  # Eredivisie
    94: "Portugal",     # Primeira Liga
    203: "Turkey",      # Süper Lig
    144: "Belgium",     # Belgian Pro League
    # Domestic Cups
    45: "England",      # FA Cup
    143: "Spain",       # Copa del Rey
    # LATAM
    71: "Brazil",       # Serie A
    128: "Argentina",   # Primera División
    239: "Colombia",    # Primera A
    242: "Ecuador",     # Liga Pro
    250: "Paraguay",    # Primera - Apertura
    252: "Paraguay",    # Primera - Clausura
    253: "USA",         # MLS
    262: "Mexico",      # Liga MX
    265: "Chile",       # Primera División
    268: "Uruguay",     # Primera - Apertura
    270: "Uruguay",     # Primera - Clausura
    281: "Peru",        # Liga 1
    299: "Venezuela",   # Primera División
    344: "Bolivia",     # Primera División
    # Middle East
    307: "Saudi-Arabia",  # Pro League
    # International (no country)
    1: None,    # World Cup
    2: None,    # Champions League
    3: None,    # Europa League
    4: None,    # Euro
    5: None,    # Nations League
    6: None,    # AFCON
    7: None,    # Asian Cup
    9: None,    # Copa América
    10: None,   # Friendlies
    11: None,   # Sudamericana
    13: None,   # Libertadores
    22: None,   # Gold Cup
    29: None,   # WCQ CAF
    30: None,   # WCQ AFC
    31: None,   # WCQ CONCACAF
    32: None,   # WCQ UEFA
    33: None,   # WCQ OFC
    34: None,   # WCQ CONMEBOL
    37: None,   # WCQ Intercontinental
    848: None,  # Conference League
}

# Kind classification for leagues
LEAGUE_KINDS = {
    # Cups
    45: "cup",      # FA Cup
    143: "cup",     # Copa del Rey
    # International
    1: "international",   # World Cup
    2: "international",   # Champions League
    3: "international",   # Europa League
    4: "international",   # Euro
    5: "international",   # Nations League
    6: "international",   # AFCON
    7: "international",   # Asian Cup
    9: "international",   # Copa América
    11: "international",  # Sudamericana
    13: "international",  # Libertadores
    22: "international",  # Gold Cup
    29: "international",  # WCQ CAF
    30: "international",  # WCQ AFC
    31: "international",  # WCQ CONCACAF
    32: "international",  # WCQ UEFA
    33: "international",  # WCQ OFC
    34: "international",  # WCQ CONMEBOL
    37: "international",  # WCQ Intercontinental
    848: "international", # Conference League
    # Friendlies
    10: "friendly",  # International Friendlies
}

# Paired leagues (Apertura/Clausura)
PAIRED_LEAGUES = {
    # Paraguay
    "PAR_PRIMERA": {
        "name": "Paraguay Primera División",
        "country": "Paraguay",
        "leagues": [250, 252],  # Apertura, Clausura
    },
    # Uruguay
    "URY_PRIMERA": {
        "name": "Uruguay Primera División",
        "country": "Uruguay",
        "leagues": [268, 270],  # Apertura, Clausura
    },
}


def _get_kind(comp: Competition) -> str:
    """Determine kind from competition."""
    if comp.league_id in LEAGUE_KINDS:
        return LEAGUE_KINDS[comp.league_id]
    if comp.match_type == "friendly":
        return "friendly"
    return "league"


async def seed_admin_leagues(session: AsyncSession, dry_run: bool = False) -> dict:
    """
    Seed admin_leagues from COMPETITIONS dict.

    This is idempotent:
    - INSERT ... ON CONFLICT DO NOTHING for new leagues
    - Does NOT update existing leagues (preserves overrides)

    Args:
        session: Database session
        dry_run: If True, don't commit changes

    Returns:
        Summary dict with counts
    """
    logger.info("Starting admin_leagues seed from COMPETITIONS...")

    # First, create paired league groups
    groups_created = 0
    group_ids = {}

    for group_key, group_info in PAIRED_LEAGUES.items():
        if not dry_run:
            result = await session.execute(
                text("""
                    INSERT INTO admin_league_groups (group_key, name, country)
                    VALUES (:key, :name, :country)
                    ON CONFLICT (group_key) DO NOTHING
                    RETURNING group_id
                """),
                {"key": group_key, "name": group_info["name"], "country": group_info["country"]}
            )
            row = result.fetchone()
            if row:
                group_ids[group_key] = row[0]
                groups_created += 1
            else:
                # Already exists, fetch ID
                result = await session.execute(
                    text("SELECT group_id FROM admin_league_groups WHERE group_key = :key"),
                    {"key": group_key}
                )
                row = result.fetchone()
                if row:
                    group_ids[group_key] = row[0]

    # Build reverse lookup: league_id -> group_id
    league_to_group = {}
    for group_key, group_info in PAIRED_LEAGUES.items():
        if group_key in group_ids:
            for lid in group_info["leagues"]:
                league_to_group[lid] = group_ids[group_key]

    # Seed leagues from COMPETITIONS
    inserted = 0
    skipped = 0

    for league_id, comp in COMPETITIONS.items():
        country = LEAGUE_COUNTRIES.get(league_id)
        kind = _get_kind(comp)
        group_id = league_to_group.get(league_id)

        if not dry_run:
            result = await session.execute(
                text("""
                    INSERT INTO admin_leagues (
                        league_id, name, country, kind, is_active,
                        priority, match_type, match_weight, group_id, source
                    )
                    VALUES (
                        :league_id, :name, :country, :kind, TRUE,
                        :priority, :match_type, :match_weight, :group_id, 'seed'
                    )
                    ON CONFLICT (league_id) DO NOTHING
                    RETURNING league_id
                """),
                {
                    "league_id": league_id,
                    "name": comp.name,
                    "country": country,
                    "kind": kind,
                    "priority": comp.priority.value,
                    "match_type": comp.match_type,
                    "match_weight": comp.match_weight,
                    "group_id": group_id,
                }
            )
            if result.fetchone():
                inserted += 1
                logger.debug(f"Inserted league {league_id}: {comp.name}")
            else:
                skipped += 1
                logger.debug(f"Skipped league {league_id}: {comp.name} (already exists)")
        else:
            inserted += 1

    if not dry_run:
        await session.commit()

    logger.info(f"Seed complete: {inserted} inserted, {skipped} skipped, {groups_created} groups created")

    return {
        "inserted": inserted,
        "skipped": skipped,
        "groups_created": groups_created,
        "dry_run": dry_run,
    }


async def sync_observed_leagues(session: AsyncSession, dry_run: bool = False) -> dict:
    """
    Discover leagues in matches that are not in admin_leagues.

    Inserts them with source='observed', is_active=false.

    Args:
        session: Database session
        dry_run: If True, don't commit changes

    Returns:
        Summary dict with counts
    """
    logger.info("Starting observed leagues discovery...")

    # Find leagues in matches not in admin_leagues
    result = await session.execute(
        text("""
            SELECT DISTINCT m.league_id
            FROM matches m
            WHERE NOT EXISTS (
                SELECT 1 FROM admin_leagues al WHERE al.league_id = m.league_id
            )
            ORDER BY m.league_id
        """)
    )
    observed_ids = [r[0] for r in result.fetchall()]

    logger.info(f"Found {len(observed_ids)} observed leagues not in admin_leagues")

    inserted = 0
    for league_id in observed_ids:
        # Try to infer country from teams
        country_result = await session.execute(
            text("""
                SELECT t.country, COUNT(*) as cnt
                FROM matches m
                JOIN teams t ON m.home_team_id = t.id
                WHERE m.league_id = :league_id AND t.country IS NOT NULL AND t.country != ''
                GROUP BY t.country
                ORDER BY cnt DESC
                LIMIT 1
            """),
            {"league_id": league_id}
        )
        country_row = country_result.fetchone()
        country = country_row[0] if country_row else None

        if not dry_run:
            result = await session.execute(
                text("""
                    INSERT INTO admin_leagues (
                        league_id, name, country, kind, is_active, source
                    )
                    VALUES (
                        :league_id, :name, :country, 'league', FALSE, 'observed'
                    )
                    ON CONFLICT (league_id) DO NOTHING
                    RETURNING league_id
                """),
                {
                    "league_id": league_id,
                    "name": f"League {league_id}",
                    "country": country,
                }
            )
            if result.fetchone():
                inserted += 1
                logger.debug(f"Inserted observed league {league_id} (country: {country})")

    if not dry_run:
        await session.commit()

    logger.info(f"Observed sync complete: {inserted} inserted from {len(observed_ids)} discovered")

    return {
        "discovered": len(observed_ids),
        "inserted": inserted,
        "dry_run": dry_run,
    }


async def full_sync(session: AsyncSession, dry_run: bool = False) -> dict:
    """
    Full sync: seed from COMPETITIONS + discover observed.

    Args:
        session: Database session
        dry_run: If True, don't commit changes

    Returns:
        Combined summary dict
    """
    seed_result = await seed_admin_leagues(session, dry_run=dry_run)
    observed_result = await sync_observed_leagues(session, dry_run=dry_run)

    return {
        "seed": seed_result,
        "observed": observed_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def get_sync_status(session: AsyncSession) -> dict:
    """
    Get current sync status of admin_leagues.

    Returns:
        Status dict with counts by source and is_active
    """
    result = await session.execute(
        text("""
            SELECT
                source,
                is_active,
                COUNT(*) as count
            FROM admin_leagues
            GROUP BY source, is_active
            ORDER BY source, is_active DESC
        """)
    )
    rows = result.fetchall()

    by_source = {}
    for row in rows:
        source = row[0]
        is_active = row[1]
        count = row[2]
        if source not in by_source:
            by_source[source] = {"active": 0, "inactive": 0}
        if is_active:
            by_source[source]["active"] = count
        else:
            by_source[source]["inactive"] = count

    total_result = await session.execute(
        text("SELECT COUNT(*) FROM admin_leagues")
    )
    total = total_result.scalar()

    competitions_count = len(COMPETITIONS)

    return {
        "total_in_db": total,
        "competitions_dict_count": competitions_count,
        "by_source": by_source,
        "synced": total >= competitions_count,
    }
