"""Feature engineering for match prediction."""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models import Match, Team

logger = logging.getLogger(__name__)
settings = get_settings()


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
    ) -> list[Match]:
        """Get completed matches for a team before a given date."""
        limit = limit or self.rolling_window * 2  # Get more for decay calculation

        # Use cache if available (for batch operations like training)
        if self._cache is not None:
            return self._cache.get_matches_before(team_id, before_date, limit)

        # Fallback to direct query (for single predictions, excluding tainted data)
        result = await self.session.execute(
            select(Match)
            .where(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
                Match.date < before_date,
                Match.status == "FT",
                Match.home_goals.isnot(None),
                Match.away_goals.isnot(None),
                Match.tainted == False,  # Exclude data quality issues
            )
            .order_by(Match.date.desc())
            .limit(limit)
        )
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
    ) -> dict:
        """
        Calculate rolling average features for a team.

        Args:
            team_id: The team's internal ID.
            match_date: The reference date (features use only data before this).
            is_home: Whether this is for the home or away team.

        Returns:
            Dictionary of feature values.
        """
        prefix = "home" if is_home else "away"
        matches = await self._get_team_matches(team_id, match_date)

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

    async def get_match_features(self, match: Match) -> dict:
        """
        Calculate all features for a match.

        Args:
            match: The Match object to calculate features for.

        Returns:
            Dictionary with all features for the match.
        """
        home_features = await self.get_team_features(
            match.home_team_id, match.date, is_home=True
        )
        away_features = await self.get_team_features(
            match.away_team_id, match.date, is_home=False
        )

        # Combine features
        features = {
            "match_id": match.id,
            "match_external_id": match.external_id,
            "date": match.date,
            "league_id": match.league_id,
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

        return features

    async def build_training_dataset(
        self,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        league_ids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Build a training dataset from historical matches.

        Only includes completed matches with results.
        Calculates features using only data before each match date (no data leakage).

        Args:
            min_date: Minimum match date to include.
            max_date: Maximum match date to include.
            league_ids: Optional list of league IDs to filter.

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

        logger.info(f"Building features for {len(matches)} matches...")

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
                features = await self.get_match_features(match)

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

    async def get_upcoming_matches_features(
        self,
        league_ids: Optional[list[int]] = None,
        include_recent_days: int = 7,
        days_ahead: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get features for upcoming and recent matches.

        Args:
            league_ids: Optional list of league IDs to filter.
            include_recent_days: Include finished matches from last N days (for showing scores).
                                 Default 7 to preserve history for iOS date selector.
            days_ahead: Limit upcoming matches to next N days. None = no limit (all upcoming).

        Returns:
            DataFrame with features for matches (upcoming + recent finished).
        """
        from datetime import timedelta

        logger.info(f"get_upcoming_matches_features: include_recent_days={include_recent_days}, days_ahead={days_ahead}")

        # Calculate date range for recent matches
        recent_cutoff = datetime.utcnow() - timedelta(days=include_recent_days)

        # Calculate date range for upcoming matches (if limited)
        future_cutoff = None
        if days_ahead is not None:
            future_cutoff = datetime.utcnow() + timedelta(days=days_ahead)

        logger.info(f"Date cutoffs: recent={recent_cutoff.isoformat()}, future={future_cutoff.isoformat() if future_cutoff else 'None'}")

        # Build upcoming condition
        # Note: NS matches should only include those from recent_cutoff forward
        # (older NS matches are stale/postponed and shouldn't be shown)
        if future_cutoff is not None:
            upcoming_condition = and_(
                Match.status == "NS",
                Match.date >= recent_cutoff,  # Don't show stale NS matches
                Match.date <= future_cutoff,
            )
        else:
            upcoming_condition = and_(
                Match.status == "NS",
                Match.date >= recent_cutoff,  # Don't show stale NS matches
            )

        # Get upcoming matches (NS) AND recent finished matches (FT, AET, PEN)
        query = (
            select(Match)
            .where(
                or_(
                    upcoming_condition,  # Upcoming (with optional future limit)
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

        rows = []
        for match in matches:
            try:
                features = await self.get_match_features(match)
                features["home_team_name"] = match.home_team.name if match.home_team else "Unknown"
                features["away_team_name"] = match.away_team.name if match.away_team else "Unknown"
                features["home_team_logo"] = match.home_team.logo_url if match.home_team else None
                features["away_team_logo"] = match.away_team.logo_url if match.away_team else None
                features["odds_home"] = match.odds_home
                features["odds_draw"] = match.odds_draw
                features["odds_away"] = match.odds_away
                # Include match status and score for iOS display
                features["status"] = match.status
                features["home_goals"] = match.home_goals
                features["away_goals"] = match.away_goals
                rows.append(features)
            except Exception as e:
                logger.error(f"Error processing match {match.id}: {e}")
                continue

        return pd.DataFrame(rows)
