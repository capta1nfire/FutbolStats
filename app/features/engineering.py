"""Feature engineering for match prediction.

This module implements the baseline features and SOTA extensions:
- Baseline: rolling goals/shots/corners averages (API-Football)
- SOTA Understat: xG rolling, justice regression
- SOTA Weather/Bio: temperature, humidity, thermal shock, circadian disruption

Point-in-time enforcement:
- All features use only data with Match.date < t0 (kickoff)
- Snapshot features (weather, understat) require captured_at < t0

Imputations and flags:
- When data is missing, impute with reasonable defaults and set *_missing=1
- understat_missing: set if insufficient xG history
- weather_missing: set if no valid weather snapshot
- thermal_shock defaults to 0 if away team profile missing
- circadian_disruption defaults to 0 if insufficient history
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models import Match, Team

logger = logging.getLogger(__name__)
settings = get_settings()

# ============================================================================
# SOTA Constants
# ============================================================================

# Justice shrinkage constant (k in rho = n/(n+k))
JUSTICE_SHRINKAGE_K = 10

# Epsilon for sqrt in justice calculation
JUSTICE_EPSILON = 0.01

# Bio disruption weights
BIO_CIRCADIAN_WEIGHT = 0.6
BIO_TZ_WEIGHT = 0.4

# Number of historical kickoffs to use for circadian baseline
CIRCADIAN_HISTORY_MATCHES = 20

# Sofascore XI position weights (per ARCHITECTURE_SOTA.md)
# GK=1.0, DEF=0.9, MID=1.0, FWD=1.1
XI_POSITION_WEIGHTS = {
    "GK": 1.0,
    "DEF": 0.9,
    "MID": 1.0,
    "FWD": 1.1,
}


# ============================================================================
# SOTA Data Loading Helpers (point-in-time safe)
# ============================================================================

async def load_match_understat(
    session: AsyncSession,
    match_id: int,
    t0: datetime,
) -> Optional[dict]:
    """
    Load Understat xG data for a match, validating point-in-time.

    Args:
        session: Database session.
        match_id: Match ID.
        t0: Kickoff time (only use data with captured_at < t0).

    Returns:
        Dict with xg_home, xg_away, xpts_home, xpts_away or None if not available.
    """
    result = await session.execute(
        text("""
            SELECT xg_home, xg_away, xpts_home, xpts_away, captured_at
            FROM match_understat_team
            WHERE match_id = :match_id AND captured_at < :t0
            ORDER BY captured_at DESC
            LIMIT 1
        """),
        {"match_id": match_id, "t0": t0}
    )
    row = result.fetchone()
    if row:
        return {
            "xg_home": row.xg_home,
            "xg_away": row.xg_away,
            "xpts_home": row.xpts_home,
            "xpts_away": row.xpts_away,
        }
    return None


async def load_match_weather(
    session: AsyncSession,
    match_id: int,
    t0: datetime,
    preferred_horizon: int = 24,
) -> Optional[dict]:
    """
    Load weather data for a match, validating point-in-time.

    Prefers the snapshot with forecast_horizon closest to preferred_horizon,
    but only if captured_at < t0.

    Args:
        session: Database session.
        match_id: Match ID.
        t0: Kickoff time.
        preferred_horizon: Preferred forecast horizon (default 24h).

    Returns:
        Dict with weather fields or None if not available.
    """
    # Get the snapshot with captured_at < t0, prefer horizon=24, then closest to t0
    result = await session.execute(
        text("""
            SELECT temp_c, humidity, wind_ms, precip_mm, is_daylight,
                   forecast_horizon_hours, captured_at
            FROM match_weather
            WHERE match_id = :match_id AND captured_at < :t0
            ORDER BY
                CASE WHEN forecast_horizon_hours = :horizon THEN 0 ELSE 1 END,
                captured_at DESC
            LIMIT 1
        """),
        {"match_id": match_id, "t0": t0, "horizon": preferred_horizon}
    )
    row = result.fetchone()
    if row:
        return {
            "weather_temp_c": row.temp_c,
            "weather_humidity": row.humidity,
            "weather_wind_ms": row.wind_ms,
            "weather_precip_mm": row.precip_mm,
            "is_daylight": row.is_daylight,
            "weather_forecast_horizon_hours": row.forecast_horizon_hours,
        }
    return None


async def load_team_profile(
    session: AsyncSession,
    team_id: int,
) -> Optional[dict]:
    """
    Load team home city profile (timezone, climate normals).

    Args:
        session: Database session.
        team_id: Team ID.

    Returns:
        Dict with timezone and climate_normals_by_month or None.
    """
    result = await session.execute(
        text("""
            SELECT home_city, timezone, climate_normals_by_month
            FROM team_home_city_profile
            WHERE team_id = :team_id
        """),
        {"team_id": team_id}
    )
    row = result.fetchone()
    if row:
        return {
            "home_city": row.home_city,
            "timezone": row.timezone,
            "climate_normals_by_month": row.climate_normals_by_month or {},
        }
    return None


async def load_team_understat_history(
    session: AsyncSession,
    team_id: int,
    before_date: datetime,
    limit: int = 20,
) -> list[dict]:
    """
    Load Understat xG history for a team's recent matches.

    Only loads matches where:
    - Team played (home or away)
    - Match finished before before_date (the target match kickoff)
    - Understat data captured before before_date (point-in-time safe for target match)

    Note: Understat xG is post-match data, so captured_at will always be AFTER
    the historical match ended. The PIT constraint is that the snapshot must
    exist before the TARGET match kickoff (before_date), not before the
    historical match kickoff.

    Args:
        session: Database session.
        team_id: Team ID.
        before_date: Target match kickoff (only use data captured before this).
        limit: Max matches to return.

    Returns:
        List of dicts with match info + xG data.
    """
    result = await session.execute(
        text("""
            SELECT
                m.id AS match_id,
                m.date AS match_date,
                m.home_team_id,
                m.away_team_id,
                m.home_goals,
                m.away_goals,
                m.match_weight,
                mut.xg_home,
                mut.xg_away,
                mut.xpts_home,
                mut.xpts_away
            FROM matches m
            JOIN match_understat_team mut ON mut.match_id = m.id
            WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
              AND m.status IN ('FT', 'AET', 'PEN')
              AND m.date < :before_date
              AND m.home_goals IS NOT NULL
              AND m.tainted = FALSE
              AND mut.captured_at < :before_date
            ORDER BY m.date DESC
            LIMIT :limit
        """),
        {"team_id": team_id, "before_date": before_date, "limit": limit}
    )
    rows = result.fetchall()
    return [
        {
            "match_id": row.match_id,
            "match_date": row.match_date,
            "is_home": row.home_team_id == team_id,
            "home_goals": row.home_goals,
            "away_goals": row.away_goals,
            "match_weight": row.match_weight or 1.0,
            "xg_home": row.xg_home,
            "xg_away": row.xg_away,
            "xpts_home": row.xpts_home,
            "xpts_away": row.xpts_away,
        }
        for row in rows
    ]


async def load_match_sofascore_xi(
    session: AsyncSession,
    match_id: int,
    t0: datetime,
) -> Optional[dict]:
    """
    Load Sofascore XI data for a match, validating point-in-time.

    Returns lineup and player data for both teams if available.

    Args:
        session: Database session.
        match_id: Match ID.
        t0: Kickoff time (only use data with captured_at < t0).

    Returns:
        Dict with home/away lineup data or None if not available.
    """
    # Load lineups
    lineup_result = await session.execute(
        text("""
            SELECT team_side, formation, captured_at
            FROM match_sofascore_lineup
            WHERE match_id = :match_id AND captured_at < :t0
        """),
        {"match_id": match_id, "t0": t0}
    )
    lineup_rows = lineup_result.fetchall()

    if not lineup_rows:
        return None

    # Load players
    player_result = await session.execute(
        text("""
            SELECT team_side, player_id_ext, position, is_starter,
                   rating_pre_match, rating_recent_form, captured_at
            FROM match_sofascore_player
            WHERE match_id = :match_id AND captured_at < :t0
        """),
        {"match_id": match_id, "t0": t0}
    )
    player_rows = player_result.fetchall()

    # Organize by team side
    data = {
        "home": {"formation": None, "captured_at": None, "players": []},
        "away": {"formation": None, "captured_at": None, "players": []},
    }

    for row in lineup_rows:
        side = row.team_side
        if side in data:
            data[side]["formation"] = row.formation
            data[side]["captured_at"] = row.captured_at

    for row in player_rows:
        side = row.team_side
        if side in data:
            data[side]["players"].append({
                "player_id_ext": row.player_id_ext,
                "position": row.position,
                "is_starter": row.is_starter,
                "rating_pre_match": row.rating_pre_match,
                "rating_recent_form": row.rating_recent_form,
            })

    # Validate we have at least some data
    has_home = data["home"]["formation"] or data["home"]["players"]
    has_away = data["away"]["formation"] or data["away"]["players"]

    if not has_home and not has_away:
        return None

    return data


def calculate_xi_features(
    players: list[dict],
    suffix: str,
) -> dict:
    """
    Calculate XI features from player lineup data.

    Features per FEATURE_DICTIONARY_SOTA.md (suffix convention):
    - xi_weighted_{suffix}: Position-weighted average rating
    - xi_p10_{suffix}, xi_p50_{suffix}, xi_p90_{suffix}: Percentiles
    - xi_weaklink_{suffix}: Minimum rating (weakest starter)
    - xi_std_{suffix}: Standard deviation of ratings

    Args:
        players: List of player dicts with position, is_starter, rating_*.
        suffix: Feature suffix (home/away).

    Returns:
        Dict of XI features.
    """
    features = {}

    # Filter to starters only
    starters = [p for p in players if p.get("is_starter")]

    # Get ratings (prefer pre_match, fallback to recent_form)
    ratings = []
    weighted_sum = 0.0
    weight_sum = 0.0

    for player in starters:
        rating = player.get("rating_pre_match") or player.get("rating_recent_form")
        if rating is not None and rating > 0:
            ratings.append(rating)

            # Get position weight
            position = player.get("position", "MID")
            pos_weight = XI_POSITION_WEIGHTS.get(position, 1.0)

            weighted_sum += rating * pos_weight
            weight_sum += pos_weight

    # Calculate features (suffix convention per FEATURE_DICTIONARY_SOTA.md)
    if ratings:
        ratings_arr = np.array(ratings)

        # Weighted average
        if weight_sum > 0:
            features[f"xi_weighted_{suffix}"] = round(weighted_sum / weight_sum, 3)
        else:
            features[f"xi_weighted_{suffix}"] = round(np.mean(ratings_arr), 3)

        # Percentiles
        features[f"xi_p10_{suffix}"] = round(np.percentile(ratings_arr, 10), 3)
        features[f"xi_p50_{suffix}"] = round(np.percentile(ratings_arr, 50), 3)
        features[f"xi_p90_{suffix}"] = round(np.percentile(ratings_arr, 90), 3)

        # Weaklink (minimum)
        features[f"xi_weaklink_{suffix}"] = round(np.min(ratings_arr), 3)

        # Standard deviation
        features[f"xi_std_{suffix}"] = round(np.std(ratings_arr), 3) if len(ratings_arr) > 1 else 0.0
    else:
        # No valid ratings - use defaults
        features[f"xi_weighted_{suffix}"] = 6.5  # League average approximation
        features[f"xi_p10_{suffix}"] = 6.0
        features[f"xi_p50_{suffix}"] = 6.5
        features[f"xi_p90_{suffix}"] = 7.0
        features[f"xi_weaklink_{suffix}"] = 6.0
        features[f"xi_std_{suffix}"] = 0.0

    return features


async def load_team_kickoff_history(
    session: AsyncSession,
    team_id: int,
    before_date: datetime,
    limit: int = CIRCADIAN_HISTORY_MATCHES,
) -> list[datetime]:
    """
    Load historical kickoff times for a team (for circadian baseline).

    Args:
        session: Database session.
        team_id: Team ID.
        before_date: Only include matches before this date.
        limit: Max matches.

    Returns:
        List of kickoff datetimes.
    """
    result = await session.execute(
        text("""
            SELECT m.date
            FROM matches m
            WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
              AND m.status IN ('FT', 'AET', 'PEN')
              AND m.date < :before_date
            ORDER BY m.date DESC
            LIMIT :limit
        """),
        {"team_id": team_id, "before_date": before_date, "limit": limit}
    )
    return [row.date for row in result.fetchall()]


def calculate_circular_mean_hour(kickoffs: list[datetime]) -> Optional[float]:
    """
    Calculate circular mean of kickoff hours.

    Uses circular statistics to handle wraparound (23:00 -> 01:00).

    Args:
        kickoffs: List of kickoff datetimes.

    Returns:
        Mean hour (0-24) or None if no kickoffs.
    """
    if not kickoffs:
        return None

    # Convert hours to radians (24h = 2π)
    angles = [2 * math.pi * dt.hour / 24 for dt in kickoffs]

    # Circular mean using unit vectors
    x = sum(math.cos(a) for a in angles) / len(angles)
    y = sum(math.sin(a) for a in angles) / len(angles)

    # Convert back to hours
    mean_angle = math.atan2(y, x)
    if mean_angle < 0:
        mean_angle += 2 * math.pi

    return mean_angle * 24 / (2 * math.pi)


def get_local_hour(dt: datetime, tz_name: Optional[str]) -> float:
    """
    Get local hour from UTC datetime and timezone name.

    Args:
        dt: Datetime in UTC.
        tz_name: IANA timezone name (e.g., 'Europe/London').

    Returns:
        Local hour (0-24). Falls back to UTC hour if timezone unavailable.
    """
    if not tz_name:
        return dt.hour

    try:
        import pytz
        tz = pytz.timezone(tz_name)
        local_dt = dt.astimezone(tz)
        return local_dt.hour + local_dt.minute / 60
    except Exception:
        return dt.hour


def get_tz_offset_hours(tz_name: Optional[str], reference_dt: datetime) -> float:
    """
    Get timezone offset in hours from UTC.

    Args:
        tz_name: IANA timezone name.
        reference_dt: Reference datetime for DST calculation.

    Returns:
        Offset in hours (e.g., 1.0 for Europe/Paris in winter).
    """
    if not tz_name:
        return 0.0

    try:
        import pytz
        tz = pytz.timezone(tz_name)
        offset = tz.utcoffset(reference_dt)
        if offset:
            return offset.total_seconds() / 3600
        return 0.0
    except Exception:
        return 0.0


class TeamMatchCache:
    """
    In-memory cache for team matches to avoid N+1 queries.

    Preloads all matches for relevant teams once, then serves
    queries from memory using binary search on dates.
    """

    def __init__(self):
        # {team_id: [(match_date, match), ...]} sorted by date desc
        self._cache: dict[int, list[tuple[datetime, Match]]] = {}
        self._loaded = False

    async def preload(self, session: AsyncSession, team_ids: set[int]) -> None:
        """Preload all matches for the given teams in a single query."""
        if not team_ids:
            return

        logger.info(f"Preloading match history for {len(team_ids)} teams...")

        # Single query to get all relevant matches (excluding tainted data)
        query = (
            select(Match)
            .where(
                or_(
                    Match.home_team_id.in_(team_ids),
                    Match.away_team_id.in_(team_ids),
                ),
                Match.status == "FT",
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
                Match.tainted == False,  # Exclude data quality issues
            )
            .order_by(Match.date.desc())
        )

        result = await session.execute(query)
        all_matches = list(result.scalars().all())

        # Index matches by team
        for match in all_matches:
            for team_id in [match.home_team_id, match.away_team_id]:
                if team_id in team_ids:
                    if team_id not in self._cache:
                        self._cache[team_id] = []
                    self._cache[team_id].append((match.date, match))

        # Sort each team's matches by date descending (most recent first)
        for team_id in self._cache:
            self._cache[team_id].sort(key=lambda x: x[0], reverse=True)

        self._loaded = True
        logger.info(f"Preloaded {len(all_matches)} matches into cache")

    def get_matches_before(
        self,
        team_id: int,
        before_date: datetime,
        limit: int = 10,
    ) -> list[Match]:
        """Get matches for a team before a given date from cache."""
        if team_id not in self._cache:
            return []

        # Filter matches before the date and return up to limit
        matches = []
        for match_date, match in self._cache[team_id]:
            if match_date < before_date:
                matches.append(match)
                if len(matches) >= limit:
                    break

        return matches

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._loaded = False


class FeatureEngineer:
    """
    Feature engineering for football match prediction.

    Implements rolling averages, time decay, and match weighting
    specifically designed for national team matches.
    """

    def __init__(
        self,
        session: AsyncSession,
        rolling_window: int = None,
        time_decay_lambda: float = None,
    ):
        self.session = session
        self.rolling_window = rolling_window or settings.ROLLING_WINDOW
        self.time_decay_lambda = time_decay_lambda or settings.TIME_DECAY_LAMBDA
        self._cache: TeamMatchCache | None = None

    @staticmethod
    def calculate_time_decay(days_since_match: int, lambda_decay: float = 0.01) -> float:
        """
        Calculate time decay weight for a match.

        Returns a weight between 0 and 1:
        - 7 days ago → ~0.93
        - 30 days ago → ~0.74
        - 90 days ago → ~0.41
        - 180 days ago → ~0.17

        Args:
            days_since_match: Number of days since the match.
            lambda_decay: Decay rate (default 0.01).

        Returns:
            Decay weight between 0 and 1.
        """
        return np.exp(-lambda_decay * days_since_match)

    async def _get_team_matches(
        self,
        team_id: int,
        before_date: datetime,
        limit: int = None,
        league_only: bool = False,
    ) -> list[Match]:
        """Get completed matches for a team before a given date.

        Args:
            team_id: Team to get matches for.
            before_date: Only matches before this date (PIT-safe).
            limit: Max matches to return.
            league_only: If True, only include matches from competitions where
                         admin_leagues.kind = 'league' (excludes cups/international).
                         Used to prevent "Exeter mode" where cup matches against
                         amateur teams inflate rolling averages.

        Returns:
            List of Match objects ordered by date descending.
        """
        limit = limit or self.rolling_window * 2  # Get more for decay calculation

        # Use cache if available (for batch operations like training)
        if self._cache is not None:
            # Note: cache does not support league_only filter yet
            # For league_only queries, we fall through to direct query
            if not league_only:
                return self._cache.get_matches_before(team_id, before_date, limit)

        # Build base query with PIT-safe guardrails:
        # 1. status='FT' (finished matches only)
        # 2. date < before_date (PIT-safe)
        # 3. Exclude tainted data
        stmt = (
            select(Match)
            .where(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
                Match.date < before_date,
                Match.status == "FT",
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
                Match.tainted == False,  # Exclude data quality issues
            )
        )

        # FASE 0 FIX: Filter to league matches only (exclude cups/international)
        # This prevents "Exeter mode" where cup matches against amateur teams
        # inflate rolling averages for lower-division teams
        if league_only:
            stmt = stmt.where(
                Match.league_id.in_(
                    text("SELECT league_id FROM admin_leagues WHERE kind = 'league'")
                )
            )

        stmt = stmt.order_by(Match.date.desc()).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    def _extract_match_stats(
        self,
        match: Match,
        team_id: int,
        reference_date: datetime,
    ) -> dict:
        """Extract stats from a match for a specific team."""
        is_home = match.home_team_id == team_id

        goals_scored = match.home_goals if is_home else match.away_goals
        goals_conceded = match.away_goals if is_home else match.home_goals

        # Calculate days since match for decay
        days_since = (reference_date - match.date).days
        decay_weight = self.calculate_time_decay(days_since, self.time_decay_lambda)

        # Extract stats from JSON if available
        shots = 0
        corners = 0
        if match.stats:
            side = "home" if is_home else "away"
            side_stats = match.stats.get(side, {})
            shots = side_stats.get("total_shots", side_stats.get("shots_on_goal", 0)) or 0
            corners = side_stats.get("corner_kicks", 0) or 0

        return {
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "shots": shots,
            "corners": corners,
            "match_weight": match.match_weight,
            "decay_weight": decay_weight,
            "combined_weight": match.match_weight * decay_weight,
        }

    def _calculate_weighted_average(
        self,
        values: list[float],
        weights: list[float],
    ) -> float:
        """Calculate weighted average with match and decay weights."""
        if not values or not weights or sum(weights) == 0:
            return 0.0

        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    async def get_team_features(
        self,
        team_id: int,
        match_date: datetime,
        is_home: bool,
        league_only: bool = False,
    ) -> dict:
        """
        Calculate rolling average features for a team.

        Args:
            team_id: The team's internal ID.
            match_date: The reference date (features use only data before this).
            is_home: Whether this is for the home or away team.
            league_only: If True, only use matches from leagues (not cups).
                         Prevents "Exeter mode" inflation.

        Returns:
            Dictionary of feature values.
        """
        prefix = "home" if is_home else "away"
        matches = await self._get_team_matches(team_id, match_date, league_only=league_only)

        if not matches:
            # No history - return defaults
            return {
                f"{prefix}_goals_scored_avg": 1.0,
                f"{prefix}_goals_conceded_avg": 1.0,
                f"{prefix}_shots_avg": 10.0,
                f"{prefix}_corners_avg": 4.0,
                f"{prefix}_rest_days": 30,
                f"{prefix}_matches_played": 0,
            }

        # Extract stats from each match
        match_stats = [
            self._extract_match_stats(m, team_id, match_date)
            for m in matches[: self.rolling_window]
        ]

        # Calculate weighted averages
        weights = [s["combined_weight"] for s in match_stats]

        goals_scored_avg = self._calculate_weighted_average(
            [s["goals_scored"] for s in match_stats], weights
        )
        goals_conceded_avg = self._calculate_weighted_average(
            [s["goals_conceded"] for s in match_stats], weights
        )
        shots_avg = self._calculate_weighted_average(
            [s["shots"] for s in match_stats], weights
        )
        corners_avg = self._calculate_weighted_average(
            [s["corners"] for s in match_stats], weights
        )

        # Rest days since last match
        rest_days = (match_date - matches[0].date).days

        return {
            f"{prefix}_goals_scored_avg": round(goals_scored_avg, 3),
            f"{prefix}_goals_conceded_avg": round(goals_conceded_avg, 3),
            f"{prefix}_shots_avg": round(shots_avg, 3),
            f"{prefix}_corners_avg": round(corners_avg, 3),
            f"{prefix}_rest_days": rest_days,
            f"{prefix}_matches_played": len(matches),
        }

    # ========================================================================
    # SOTA Feature Methods: Understat (xG, Justice)
    # ========================================================================

    async def get_understat_features(
        self,
        match: Match,
    ) -> dict:
        """
        Calculate Understat-based features for a match.

        Features:
        - home_xg_for_avg, home_xg_against_avg: Rolling xG averages for home team
        - away_xg_for_avg, away_xg_against_avg: Rolling xG averages for away team
        - xg_diff_avg: home_xg_for_avg - away_xg_for_avg
        - xpts_diff_avg: Rolling xPTS difference (if available)
        - home_justice_shrunk, away_justice_shrunk: Regression to mean indicator
        - justice_diff: Difference in justice

        Point-in-time: Only uses understat data with captured_at < match.date.
        """
        t0 = match.date
        features = {}

        # Load history for both teams
        home_history = await load_team_understat_history(
            self.session, match.home_team_id, t0, limit=self.rolling_window
        )
        away_history = await load_team_understat_history(
            self.session, match.away_team_id, t0, limit=self.rolling_window
        )

        # Process home team
        home_xg_for, home_xg_against, home_xpts = [], [], []
        home_goals_total, home_xg_total = 0.0, 0.0
        home_weights = []

        for h in home_history:
            days_since = (t0 - h["match_date"]).days
            decay = self.calculate_time_decay(days_since, self.time_decay_lambda)
            weight = h["match_weight"] * decay
            home_weights.append(weight)

            if h["is_home"]:
                xg_for = h["xg_home"] or 0
                xg_against = h["xg_away"] or 0
                goals = h["home_goals"]
                xpts = h["xpts_home"]
            else:
                xg_for = h["xg_away"] or 0
                xg_against = h["xg_home"] or 0
                goals = h["away_goals"]
                xpts = h["xpts_away"]

            home_xg_for.append(xg_for)
            home_xg_against.append(xg_against)
            home_goals_total += goals
            home_xg_total += xg_for
            if xpts is not None:
                home_xpts.append(xpts)

        # Process away team
        away_xg_for, away_xg_against, away_xpts = [], [], []
        away_goals_total, away_xg_total = 0.0, 0.0
        away_weights = []

        for h in away_history:
            days_since = (t0 - h["match_date"]).days
            decay = self.calculate_time_decay(days_since, self.time_decay_lambda)
            weight = h["match_weight"] * decay
            away_weights.append(weight)

            if h["is_home"]:
                xg_for = h["xg_home"] or 0
                xg_against = h["xg_away"] or 0
                goals = h["home_goals"]
                xpts = h["xpts_home"]
            else:
                xg_for = h["xg_away"] or 0
                xg_against = h["xg_home"] or 0
                goals = h["away_goals"]
                xpts = h["xpts_away"]

            away_xg_for.append(xg_for)
            away_xg_against.append(xg_against)
            away_goals_total += goals
            away_xg_total += xg_for
            if xpts is not None:
                away_xpts.append(xpts)

        # Calculate weighted averages
        features["home_xg_for_avg"] = round(
            self._calculate_weighted_average(home_xg_for, home_weights), 3
        )
        features["home_xg_against_avg"] = round(
            self._calculate_weighted_average(home_xg_against, home_weights), 3
        )
        features["away_xg_for_avg"] = round(
            self._calculate_weighted_average(away_xg_for, away_weights), 3
        )
        features["away_xg_against_avg"] = round(
            self._calculate_weighted_average(away_xg_against, away_weights), 3
        )

        # Derived: xg_diff_avg
        features["xg_diff_avg"] = round(
            features["home_xg_for_avg"] - features["away_xg_for_avg"], 3
        )

        # xPTS diff (if available)
        home_xpts_avg = self._calculate_weighted_average(home_xpts, home_weights[:len(home_xpts)]) if home_xpts else 0
        away_xpts_avg = self._calculate_weighted_average(away_xpts, away_weights[:len(away_xpts)]) if away_xpts else 0
        features["xpts_diff_avg"] = round(home_xpts_avg - away_xpts_avg, 3)

        # Justice calculation: (G - XG) / sqrt(XG + eps)
        # Then shrinkage: rho = n / (n + k), justice_shrunk = rho * justice
        n_home = len(home_history)
        n_away = len(away_history)

        if n_home > 0 and home_xg_total > 0:
            justice_home = (home_goals_total - home_xg_total) / math.sqrt(home_xg_total + JUSTICE_EPSILON)
            rho_home = n_home / (n_home + JUSTICE_SHRINKAGE_K)
            features["home_justice_shrunk"] = round(rho_home * justice_home, 3)
        else:
            features["home_justice_shrunk"] = 0.0

        if n_away > 0 and away_xg_total > 0:
            justice_away = (away_goals_total - away_xg_total) / math.sqrt(away_xg_total + JUSTICE_EPSILON)
            rho_away = n_away / (n_away + JUSTICE_SHRINKAGE_K)
            features["away_justice_shrunk"] = round(rho_away * justice_away, 3)
        else:
            features["away_justice_shrunk"] = 0.0

        features["justice_diff"] = round(
            features["home_justice_shrunk"] - features["away_justice_shrunk"], 3
        )

        # Flags
        features["understat_missing"] = 1 if (n_home == 0 or n_away == 0) else 0
        features["understat_samples_home"] = n_home
        features["understat_samples_away"] = n_away

        return features

    # ========================================================================
    # SOTA Feature Methods: Weather + Bio-adaptability
    # ========================================================================

    async def get_weather_bio_features(
        self,
        match: Match,
    ) -> dict:
        """
        Calculate weather and bio-adaptability features for a match.

        Weather features:
        - weather_temp_c, weather_humidity, weather_wind_ms, weather_precip_mm
        - is_daylight, weather_forecast_horizon_hours, weather_missing

        Bio features:
        - thermal_shock: T_stadium - T_away_home_month_mean
        - thermal_shock_abs: abs(thermal_shock)
        - tz_shift: |tz_match - tz_away_base|
        - circadian_disruption: distance to typical kickoff hour
        - bio_disruption: weighted combination

        Point-in-time: Only uses weather data with captured_at < match.date.
        """
        t0 = match.date
        features = {}

        # Load weather data
        weather = await load_match_weather(self.session, match.id, t0)

        if weather:
            features["weather_temp_c"] = weather["weather_temp_c"]
            features["weather_humidity"] = weather["weather_humidity"]
            features["weather_wind_ms"] = weather["weather_wind_ms"]
            features["weather_precip_mm"] = weather["weather_precip_mm"]
            features["is_daylight"] = 1 if weather["is_daylight"] else 0
            features["weather_forecast_horizon_hours"] = weather["weather_forecast_horizon_hours"]
            features["weather_missing"] = 0
        else:
            # Impute defaults
            features["weather_temp_c"] = 15.0  # Mild default
            features["weather_humidity"] = 60.0
            features["weather_wind_ms"] = 3.0
            features["weather_precip_mm"] = 0.0
            features["is_daylight"] = 1 if 6 <= t0.hour < 20 else 0
            features["weather_forecast_horizon_hours"] = 24
            features["weather_missing"] = 1

        # Bio features: need away team profile
        away_profile = await load_team_profile(self.session, match.away_team_id)

        # Thermal shock: T_stadium - T_away_home_month_mean
        if away_profile and away_profile["climate_normals_by_month"]:
            month_key = f"{t0.month:02d}"
            climate = away_profile["climate_normals_by_month"].get(month_key, {})
            away_temp_mean = climate.get("temp_c_mean")
            if away_temp_mean is not None:
                features["thermal_shock"] = round(
                    features["weather_temp_c"] - away_temp_mean, 2
                )
            else:
                features["thermal_shock"] = 0.0
        else:
            features["thermal_shock"] = 0.0

        features["thermal_shock_abs"] = abs(features["thermal_shock"])

        # Timezone shift
        match_profile = await load_team_profile(self.session, match.home_team_id)
        match_tz = match_profile["timezone"] if match_profile else None
        away_tz = away_profile["timezone"] if away_profile else None

        if match_tz and away_tz:
            match_offset = get_tz_offset_hours(match_tz, t0)
            away_offset = get_tz_offset_hours(away_tz, t0)
            features["tz_shift"] = abs(match_offset - away_offset)
        else:
            features["tz_shift"] = 0.0

        # Circadian disruption
        # Get away team's typical kickoff hour from history
        kickoff_history = await load_team_kickoff_history(
            self.session, match.away_team_id, t0
        )
        typical_hour = calculate_circular_mean_hour(kickoff_history)

        if typical_hour is not None and away_tz:
            # Get local hour of this match for away team
            local_hour = get_local_hour(t0, away_tz)

            # Circular distance (0-12 hours)
            diff = abs(local_hour - typical_hour)
            if diff > 12:
                diff = 24 - diff

            # Normalize to [0, 1]: 12h difference = 1.0
            features["circadian_disruption"] = round(diff / 12, 3)
        else:
            features["circadian_disruption"] = 0.0

        # Combined bio disruption
        # bio_disruption = a*circadian + b*min(tz_shift,6)/6
        features["bio_disruption"] = round(
            BIO_CIRCADIAN_WEIGHT * features["circadian_disruption"]
            + BIO_TZ_WEIGHT * min(features["tz_shift"], 6) / 6,
            3
        )

        return features

    # ========================================================================
    # SOTA Feature Methods: Sofascore XI (lineup/ratings)
    # ========================================================================

    async def get_sofascore_xi_features(
        self,
        match: Match,
    ) -> dict:
        """
        Calculate Sofascore XI features for a match.

        Features (per FEATURE_DICTIONARY_SOTA.md):
        - home_xi_weighted, away_xi_weighted: Position-weighted average rating
        - xi_weighted_diff: home - away difference
        - home_xi_p10, home_xi_p50, home_xi_p90: Percentiles (same for away)
        - home_xi_weaklink, away_xi_weaklink: Minimum rating
        - home_xi_std, away_xi_std: Standard deviation
        - formation_home, formation_away: Formation strings
        - xi_missing: Flag (0/1) if data unavailable
        - xi_captured_horizon_minutes: Minutes before kickoff when captured

        Point-in-time: Only uses data with captured_at < match.date.
        """
        t0 = match.date
        features = {}

        # Load XI data
        xi_data = await load_match_sofascore_xi(self.session, match.id, t0)

        if xi_data is None:
            # No XI data - return defaults with missing flag
            return self._get_xi_defaults()

        # Calculate features for home team
        home_players = xi_data["home"]["players"]
        home_features = calculate_xi_features(home_players, "home")
        features.update(home_features)

        # Calculate features for away team
        away_players = xi_data["away"]["players"]
        away_features = calculate_xi_features(away_players, "away")
        features.update(away_features)

        # Derived: weighted diff (per FEATURE_DICTIONARY_SOTA.md)
        features["xi_weighted_diff"] = round(
            features["xi_weighted_home"] - features["xi_weighted_away"], 3
        )

        # Formations
        features["formation_home"] = xi_data["home"]["formation"] or "unknown"
        features["formation_away"] = xi_data["away"]["formation"] or "unknown"

        # Calculate capture horizon (minutes before kickoff)
        captured_at = xi_data["home"].get("captured_at") or xi_data["away"].get("captured_at")
        if captured_at and t0:
            horizon_seconds = (t0 - captured_at).total_seconds()
            features["xi_captured_horizon_minutes"] = max(0, int(horizon_seconds / 60))
        else:
            features["xi_captured_horizon_minutes"] = 0

        # Check data completeness
        home_starters = [p for p in home_players if p.get("is_starter")]
        away_starters = [p for p in away_players if p.get("is_starter")]

        # Missing flag: set if we have <11 starters per side (incomplete XI)
        if len(home_starters) < 11 or len(away_starters) < 11:
            features["xi_missing"] = 1
        else:
            features["xi_missing"] = 0

        return features

    def _get_xi_defaults(self) -> dict:
        """Return default XI features when data is missing."""
        return {
            "xi_weighted_home": 6.5,
            "xi_p10_home": 6.0,
            "xi_p50_home": 6.5,
            "xi_p90_home": 7.0,
            "xi_weaklink_home": 6.0,
            "xi_std_home": 0.0,
            "xi_weighted_away": 6.5,
            "xi_p10_away": 6.0,
            "xi_p50_away": 6.5,
            "xi_p90_away": 7.0,
            "xi_weaklink_away": 6.0,
            "xi_std_away": 0.0,
            "xi_weighted_diff": 0.0,
            "formation_home": "unknown",
            "formation_away": "unknown",
            "xi_missing": 1,
            "xi_captured_horizon_minutes": 0,
        }

    async def get_match_features(
        self, match: Match, league_only: bool = False
    ) -> dict:
        """
        Calculate all features for a match.

        Args:
            match: The Match object to calculate features for.
            league_only: If True, rolling averages use only league matches
                         (excludes cups/international). Prevents "Exeter mode"
                         where cup matches against amateur teams inflate stats.

        Returns:
            Dictionary with all features for the match.
        """
        home_features = await self.get_team_features(
            match.home_team_id, match.date, is_home=True, league_only=league_only
        )
        away_features = await self.get_team_features(
            match.away_team_id, match.date, is_home=False, league_only=league_only
        )

        # Combine features
        features = {
            "match_id": match.id,
            "match_external_id": match.external_id,
            "date": match.date,
            "league_id": match.league_id,
            # FASE 0 FIX: Include team IDs for kill-switch router
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            **home_features,
            **away_features,
        }

        # Add derived features
        features["goal_diff_avg"] = (
            features["home_goals_scored_avg"]
            - features["away_goals_scored_avg"]
        )
        features["rest_diff"] = (
            features["home_rest_days"] - features["away_rest_days"]
        )

        # === FASE 1: Competitiveness features for draw prediction ===
        # These capture how evenly matched teams are (low values → more likely draw)

        # Attack strength difference (absolute)
        features["abs_attack_diff"] = abs(
            features["home_goals_scored_avg"] - features["away_goals_scored_avg"]
        )

        # Defense strength difference (absolute)
        features["abs_defense_diff"] = abs(
            features["home_goals_conceded_avg"] - features["away_goals_conceded_avg"]
        )

        # Combined strength gap: overall team quality difference
        # Low values indicate evenly matched teams → higher draw probability
        home_net = features["home_goals_scored_avg"] - features["home_goals_conceded_avg"]
        away_net = features["away_goals_scored_avg"] - features["away_goals_conceded_avg"]
        features["abs_strength_gap"] = abs(home_net - away_net)

        # === SOTA Features: Understat (xG, Justice) ===
        try:
            understat_features = await self.get_understat_features(match)
            features.update(understat_features)
        except Exception as e:
            logger.warning(f"Understat features failed for match {match.id}: {e}")
            # Set defaults with missing flag
            features.update({
                "home_xg_for_avg": 0.0,
                "home_xg_against_avg": 0.0,
                "away_xg_for_avg": 0.0,
                "away_xg_against_avg": 0.0,
                "xg_diff_avg": 0.0,
                "xpts_diff_avg": 0.0,
                "home_justice_shrunk": 0.0,
                "away_justice_shrunk": 0.0,
                "justice_diff": 0.0,
                "understat_missing": 1,
                "understat_samples_home": 0,
                "understat_samples_away": 0,
            })

        # === SOTA Features: Weather + Bio-adaptability ===
        try:
            weather_bio_features = await self.get_weather_bio_features(match)
            features.update(weather_bio_features)
        except Exception as e:
            logger.warning(f"Weather/Bio features failed for match {match.id}: {e}")
            # Set defaults with missing flag
            features.update({
                "weather_temp_c": 15.0,
                "weather_humidity": 60.0,
                "weather_wind_ms": 3.0,
                "weather_precip_mm": 0.0,
                "is_daylight": 1,
                "weather_forecast_horizon_hours": 24,
                "weather_missing": 1,
                "thermal_shock": 0.0,
                "thermal_shock_abs": 0.0,
                "tz_shift": 0.0,
                "circadian_disruption": 0.0,
                "bio_disruption": 0.0,
            })

        # === SOTA Features: Sofascore XI (lineup/ratings) ===
        try:
            xi_features = await self.get_sofascore_xi_features(match)
            features.update(xi_features)
        except Exception as e:
            logger.warning(f"Sofascore XI features failed for match {match.id}: {e}")
            # Set defaults with missing flag
            features.update(self._get_xi_defaults())

        return features

    async def get_match_features_asof(
        self,
        match: Match,
        asof_dt: datetime,
        league_only: bool = False,
    ) -> dict:
        """
        PIT-strict: Calculate features as-of a specific datetime.

        Used for experimental predictions where we need features
        calculated at snapshot_at (pre-kickoff) rather than match.date.

        This ensures that rest_days, goal averages, and other temporal features
        are computed using only data available at asof_dt, preventing information
        leakage from the time between snapshot and kickoff.

        Args:
            match: Match object
            asof_dt: Point-in-time for feature calculation (typically snapshot_at)
            league_only: If True, rolling averages use only league matches
                         (excludes cups/international). Prevents "Exeter mode".

        Returns:
            dict with features calculated as-of asof_dt
        """
        # Guardar fecha original
        orig_date = match.date

        try:
            # Prevenir flush accidental mientras modificamos el objeto
            with self.session.no_autoflush:
                # Override temporal para cálculos
                match.date = asof_dt

                # Reutilizar lógica existente con fecha "falsa"
                features = await self.get_match_features(match, league_only=league_only)

            return features
        finally:
            # Restaurar fecha original (importante si match está en session)
            match.date = orig_date

    async def build_training_dataset(
        self,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        league_ids: Optional[list[int]] = None,
        league_only: bool = False,
    ) -> pd.DataFrame:
        """
        Build a training dataset from historical matches.

        Only includes completed matches with results.
        Calculates features using only data before each match date (no data leakage).

        Args:
            min_date: Minimum match date to include.
            max_date: Maximum match date to include.
            league_ids: Optional list of league IDs to filter.
            league_only: If True, rolling averages use only league matches
                         (admin_leagues.kind='league'). FASE 1: Eliminates
                         training-serving skew when serving uses league_only=True.

        Returns:
            DataFrame with features and target variable.
        """
        # Build query for completed matches (excluding tainted data)
        query = select(Match).where(
            Match.status == "FT",
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
            Match.tainted == False,  # Exclude data quality issues
        )

        if min_date:
            query = query.where(Match.date >= min_date)
        if max_date:
            query = query.where(Match.date <= max_date)
        if league_ids:
            query = query.where(Match.league_id.in_(league_ids))

        query = query.order_by(Match.date)

        result = await self.session.execute(query)
        matches = list(result.scalars().all())

        logger.info(f"Building features for {len(matches)} matches... (league_only={league_only})")

        # Preload match history cache to avoid N+1 queries
        # This reduces ~2000 queries to 1 query for 1000 matches
        team_ids = set()
        for match in matches:
            team_ids.add(match.home_team_id)
            team_ids.add(match.away_team_id)

        self._cache = TeamMatchCache()
        await self._cache.preload(self.session, team_ids)

        # Build features for each match
        rows = []
        for i, match in enumerate(matches):
            if (i + 1) % 500 == 0:
                logger.info(f"Processing match {i + 1}/{len(matches)}")

            try:
                features = await self.get_match_features(match, league_only=league_only)

                # Add target variable
                if match.home_goals > match.away_goals:
                    target = 0  # Home win
                elif match.home_goals == match.away_goals:
                    target = 1  # Draw
                else:
                    target = 2  # Away win

                features["result"] = target
                features["home_goals"] = match.home_goals
                features["away_goals"] = match.away_goals

                # Add odds if available
                features["odds_home"] = match.odds_home
                features["odds_draw"] = match.odds_draw
                features["odds_away"] = match.odds_away

                rows.append(features)

            except Exception as e:
                logger.error(f"Error processing match {match.id}: {e}")
                continue

        # Clear cache to free memory
        if self._cache is not None:
            self._cache.clear()
            self._cache = None

        df = pd.DataFrame(rows)
        logger.info(f"Built dataset with {len(df)} samples and {len(df.columns)} features")

        return df

    async def get_matches_features_by_ids(
        self,
        match_ids: list[int],
        league_only: bool = False,
    ) -> pd.DataFrame:
        """
        Build features for specific matches by their IDs.

        Used by Sensor B to build training data from recent finished matches.
        Only processes completed matches (FT, AET, PEN) with valid scores.

        Args:
            match_ids: List of match IDs to process.
            league_only: If True, rolling averages use only league matches.
                         FASE 0 FIX: Prevents "Exeter mode" inflation.

        Returns:
            DataFrame with features for each match, including home_goals/away_goals.
        """
        if not match_ids:
            return pd.DataFrame()

        # Query matches by ID (only completed with valid scores)
        result = await self.session.execute(
            select(Match)
            .where(
                Match.id.in_(match_ids),
                Match.status.in_(["FT", "AET", "PEN"]),
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
                Match.tainted.is_(False),
            )
            .order_by(Match.date.desc())
        )
        matches = list(result.scalars().all())

        if not matches:
            return pd.DataFrame()

        logger.info(f"Building features for {len(matches)} matches by ID...")

        # Preload match history cache to avoid N+1 queries
        team_ids = set()
        for match in matches:
            team_ids.add(match.home_team_id)
            team_ids.add(match.away_team_id)

        self._cache = TeamMatchCache()
        await self._cache.preload(self.session, team_ids)

        # Build features for each match
        rows = []
        for match in matches:
            try:
                features = await self.get_match_features(match, league_only=league_only)

                # Add scores for label computation
                features["home_goals"] = match.home_goals
                features["away_goals"] = match.away_goals

                # Add odds if available
                features["odds_home"] = match.odds_home
                features["odds_draw"] = match.odds_draw
                features["odds_away"] = match.odds_away

                rows.append(features)

            except Exception as e:
                logger.error(f"Error processing match {match.id}: {e}")
                continue

        # Clear cache to free memory
        if self._cache is not None:
            self._cache.clear()
            self._cache = None

        df = pd.DataFrame(rows)
        logger.info(f"Built features for {len(df)} matches by ID")

        return df

    async def get_upcoming_matches_features(
        self,
        league_ids: Optional[list[int]] = None,
        include_recent_days: int = 7,
        days_ahead: Optional[int] = None,
        league_only: bool = False,
    ) -> pd.DataFrame:
        """
        Get features for upcoming and recent matches.

        Args:
            league_ids: Optional list of league IDs to filter.
            include_recent_days: Include finished matches from last N days (for showing scores).
                                 Default 7 to preserve history for iOS date selector.
            days_ahead: Limit upcoming matches to next N days. None = no limit (all upcoming).
            league_only: If True, rolling averages use only league matches (not cups).
                         FASE 0 FIX: Prevents "Exeter mode" inflation.

        Returns:
            DataFrame with features for matches (upcoming + recent finished).
        """
        from datetime import timedelta

        logger.info(f"get_upcoming_matches_features: include_recent_days={include_recent_days}, days_ahead={days_ahead}")

        # Calculate date range using calendar days (start of day in UTC)
        # days_back=1 means yesterday at 00:00, days_ahead=1 means tomorrow at 23:59:59
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        recent_cutoff = today_start - timedelta(days=include_recent_days)

        # Calculate future cutoff (end of the target day)
        future_cutoff = None
        if days_ahead is not None:
            future_cutoff = today_start + timedelta(days=days_ahead + 1)  # Start of day after

        logger.info(f"Date cutoffs: today={today_start.isoformat()}, recent={recent_cutoff.isoformat()}, future={future_cutoff.isoformat() if future_cutoff else 'None'}")

        # Live match statuses (in-progress games that should remain visible)
        # These are matches that transitioned from NS and are currently being played
        LIVE_STATUSES = ["1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"]

        # Guardrail: don't include "stuck" live matches older than 24h
        # (these are likely data errors or abandoned matches)
        live_guardrail_cutoff = today_start - timedelta(days=1)

        # Build upcoming condition
        # Note: NS matches should only include those from recent_cutoff forward
        # (older NS matches are stale/postponed and shouldn't be shown)
        if future_cutoff is not None:
            upcoming_condition = and_(
                Match.status == "NS",
                Match.date >= recent_cutoff,  # Don't show stale NS matches
                Match.date <= future_cutoff,
            )
            live_condition = and_(
                Match.status.in_(LIVE_STATUSES),
                Match.date >= live_guardrail_cutoff,  # Guardrail: no stuck live > 24h
                Match.date <= future_cutoff,
            )
        else:
            upcoming_condition = and_(
                Match.status == "NS",
                Match.date >= recent_cutoff,  # Don't show stale NS matches
            )
            live_condition = and_(
                Match.status.in_(LIVE_STATUSES),
                Match.date >= live_guardrail_cutoff,  # Guardrail: no stuck live > 24h
            )

        # Get upcoming matches (NS), live matches, AND recent finished matches (FT, AET, PEN)
        query = (
            select(Match)
            .where(
                or_(
                    upcoming_condition,  # Upcoming (with optional future limit)
                    live_condition,  # Live/in-progress matches
                    and_(
                        Match.status.in_(["FT", "AET", "PEN"]),  # Finished
                        Match.date >= recent_cutoff,  # Recent
                    ),
                )
            )
            .options(selectinload(Match.home_team), selectinload(Match.away_team))
        )

        if league_ids:
            query = query.where(Match.league_id.in_(league_ids))

        query = query.order_by(Match.date)

        result = await self.session.execute(query)
        matches = list(result.scalars().all())

        logger.info(f"Building features for {len(matches)} matches (upcoming + recent)...")

        # Preload match history cache to avoid N+1 queries
        # Without this: ~480 queries for 240 matches (2 per match for home/away)
        # With this: 1 query to preload all team history
        team_ids = set()
        for match in matches:
            if match.home_team_id:
                team_ids.add(match.home_team_id)
            if match.away_team_id:
                team_ids.add(match.away_team_id)

        if team_ids:
            self._cache = TeamMatchCache()
            await self._cache.preload(self.session, team_ids)

        rows = []
        for match in matches:
            try:
                features = await self.get_match_features(match, league_only=league_only)
                features["home_team_name"] = match.home_team.name if match.home_team else "Unknown"
                features["away_team_name"] = match.away_team.name if match.away_team else "Unknown"
                features["home_team_logo"] = match.home_team.logo_url if match.home_team else None
                features["away_team_logo"] = match.away_team.logo_url if match.away_team else None
                # External IDs for team override resolution
                features["home_team_external_id"] = match.home_team.external_id if match.home_team else None
                features["away_team_external_id"] = match.away_team.external_id if match.away_team else None
                features["odds_home"] = match.odds_home
                features["odds_draw"] = match.odds_draw
                features["odds_away"] = match.odds_away
                # Include match status and score for iOS display
                features["status"] = match.status
                features["elapsed"] = match.elapsed  # Current minute for live matches
                features["elapsed_extra"] = match.elapsed_extra  # Added/injury time (e.g., 3 for 90+3)
                features["home_goals"] = match.home_goals
                features["away_goals"] = match.away_goals
                # Include events for live match timeline (goals with minute/team info)
                features["events"] = match.events
                # Include venue for LLM narrative enrichment
                features["venue_name"] = match.venue_name
                features["venue_city"] = match.venue_city
                rows.append(features)
            except Exception as e:
                logger.error(f"Error processing match {match.id}: {e}")
                continue

        # Clear cache to free memory
        if self._cache is not None:
            self._cache.clear()
            self._cache = None

        return pd.DataFrame(rows)
