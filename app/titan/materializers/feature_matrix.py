"""TITAN Feature Matrix Materializer.

Builds feature_matrix rows from extracted data with strict policies:
1. PIT Compliance: pit_max_captured_at < kickoff_utc
2. Insertion Policy: Tier 1 (odds) required, Tier 2/3 optional with NULLs

Per plan zazzy-jingling-pudding.md v1.1:
- REGLA 1: Sin odds -> NO insertar (Tier 1 es gate obligatorio)
- REGLA 2: Con odds, sin Tier 2/3 -> SI insertar con NULLs
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional


def _utc_now() -> datetime:
    """Get current UTC timestamp (timezone-aware) for TIMESTAMPTZ compatibility."""
    return datetime.now(timezone.utc)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.titan.config import get_titan_settings

logger = logging.getLogger(__name__)
titan_settings = get_titan_settings()


class PITViolationError(Exception):
    """Raised when data would violate PIT constraint."""

    pass


class InsertionPolicyViolation(Exception):
    """Raised when insertion policy requirements not met."""

    pass


@dataclass
class OddsFeatures:
    """Tier 1: Odds features."""

    odds_home_close: Decimal
    odds_draw_close: Decimal
    odds_away_close: Decimal
    implied_prob_home: Decimal
    implied_prob_draw: Decimal
    implied_prob_away: Decimal
    captured_at: datetime


@dataclass
class FormFeatures:
    """Tier 2: Form features for ONE team.

    Called twice: once for home team, once for away team.
    The caller (runner) passes form_home and form_away separately.
    """

    form_last5: str  # e.g., "WWDLW"
    goals_scored_last5: int
    goals_conceded_last5: int
    points_last5: int
    captured_at: datetime


@dataclass
class H2HFeatures:
    """Tier 3: Head-to-Head features."""

    h2h_total_matches: int
    h2h_home_wins: int
    h2h_draws: int
    h2h_away_wins: int
    h2h_home_goals: int
    h2h_away_goals: int
    captured_at: datetime


def should_insert_feature_row(
    odds: Optional[OddsFeatures],
    form: Optional[FormFeatures],
    h2h: Optional[H2HFeatures],
) -> tuple[bool, str]:
    """
    Insertion policy for feature_matrix.

    RULE 1: No odds -> DON'T insert (Tier 1 is mandatory gate)
    RULE 2: With odds, missing Tier 2/3 -> DO insert with NULLs

    Args:
        odds: Tier 1 odds features (or None)
        form: Tier 2 form features (or None)
        h2h: Tier 3 H2H features (or None)

    Returns:
        (should_insert, reason)
    """
    # RULE 1: Without odds -> NO insert
    if odds is None:
        return False, "Missing Tier 1 (odds) - skipping"

    # RULE 2: With odds, missing Tier 2/3 -> YES insert with NULLs
    return True, "Tier 1 complete, inserting (Tier 2/3 may be NULL)"


def compute_pit_max(
    odds: Optional[OddsFeatures],
    form: Optional[FormFeatures],
    h2h: Optional[H2HFeatures],
) -> datetime:
    """
    Compute pit_max_captured_at.

    If only odds: pit_max = odds.captured_at
    If multiple: pit_max = max(all valid captured_at)

    Args:
        odds: Tier 1 features (MUST have at least this per insertion policy)
        form: Tier 2 features (optional)
        h2h: Tier 3 features (optional)

    Returns:
        Maximum captured_at timestamp

    Raises:
        ValueError: If no timestamps available (should never happen if policy enforced)
    """
    timestamps = []

    if odds and odds.captured_at:
        timestamps.append(odds.captured_at)
    if form and form.captured_at:
        timestamps.append(form.captured_at)
    if h2h and h2h.captured_at:
        timestamps.append(h2h.captured_at)

    if not timestamps:
        raise ValueError("No captured_at timestamps available")

    return max(timestamps)


def compute_implied_probabilities(
    odds_home: Decimal,
    odds_draw: Decimal,
    odds_away: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    """
    Compute normalized implied probabilities from odds.

    Normalization removes bookmaker margin (overround).

    Formula:
        raw_prob_X = 1 / odds_X
        sum = raw_prob_home + raw_prob_draw + raw_prob_away
        implied_prob_X = raw_prob_X / sum

    Args:
        odds_home: Home win decimal odds
        odds_draw: Draw decimal odds
        odds_away: Away win decimal odds

    Returns:
        (implied_prob_home, implied_prob_draw, implied_prob_away)
    """
    raw_home = Decimal(1) / odds_home
    raw_draw = Decimal(1) / odds_draw
    raw_away = Decimal(1) / odds_away

    total = raw_home + raw_draw + raw_away

    return (
        round(raw_home / total, 4),
        round(raw_draw / total, 4),
        round(raw_away / total, 4),
    )


class FeatureMatrixMaterializer:
    """Materializes feature_matrix rows with PIT compliance.

    Usage:
        materializer = FeatureMatrixMaterializer(session)

        # Build features from extracted data
        odds = materializer.build_odds_features(extraction_result)
        form = await materializer.compute_form_features(match_id, kickoff)
        h2h = await materializer.compute_h2h_features(home_id, away_id, kickoff)

        # Insert with policy enforcement
        await materializer.insert_row(match_id, kickoff, odds, form, h2h)
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.schema = titan_settings.TITAN_SCHEMA

    def build_odds_features(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        captured_at: datetime,
    ) -> OddsFeatures:
        """Build Tier 1 odds features from raw odds.

        Args:
            odds_home: Home win decimal odds
            odds_draw: Draw decimal odds
            odds_away: Away win decimal odds
            captured_at: When odds were captured

        Returns:
            OddsFeatures dataclass
        """
        home = Decimal(str(odds_home))
        draw = Decimal(str(odds_draw))
        away = Decimal(str(odds_away))

        prob_home, prob_draw, prob_away = compute_implied_probabilities(home, draw, away)

        return OddsFeatures(
            odds_home_close=home,
            odds_draw_close=draw,
            odds_away_close=away,
            implied_prob_home=prob_home,
            implied_prob_draw=prob_draw,
            implied_prob_away=prob_away,
            captured_at=captured_at,
        )

    async def compute_form_features(
        self,
        team_id: int,
        kickoff_utc: datetime,
        limit: int = 5,
    ) -> Optional[FormFeatures]:
        """Compute Tier 2 form features from public.matches.

        PIT-SAFE: Only uses matches with date < kickoff_utc.

        Args:
            team_id: Team to compute form for
            kickoff_utc: Target match kickoff (for PIT filter)
            limit: Number of recent matches (default 5)

        Returns:
            FormFeatures or None if insufficient data
        """
        query = text("""
            SELECT
                id,
                home_team_id,
                away_team_id,
                home_goals,
                away_goals,
                date
            FROM public.matches
            WHERE (home_team_id = :team_id OR away_team_id = :team_id)
              AND date < :kickoff
              AND status IN ('FT', 'AET', 'PEN')
            ORDER BY date DESC
            LIMIT :limit
        """)

        result = await self.session.execute(query, {
            "team_id": team_id,
            "kickoff": kickoff_utc,
            "limit": limit,
        })
        rows = result.fetchall()

        if len(rows) < limit:
            return None  # Insufficient data

        form = []
        goals_for = 0
        goals_against = 0

        for row in rows:
            is_home = row[1] == team_id
            home_goals, away_goals = row[3], row[4]

            if is_home:
                goals_for += home_goals
                goals_against += away_goals
                if home_goals > away_goals:
                    form.append("W")
                elif home_goals < away_goals:
                    form.append("L")
                else:
                    form.append("D")
            else:
                goals_for += away_goals
                goals_against += home_goals
                if away_goals > home_goals:
                    form.append("W")
                elif away_goals < home_goals:
                    form.append("L")
                else:
                    form.append("D")

        form_str = "".join(form)
        points = form.count("W") * 3 + form.count("D")

        return FormFeatures(
            form_last5=form_str,
            goals_scored_last5=goals_for,
            goals_conceded_last5=goals_against,
            points_last5=points,
            captured_at=_utc_now(),
        )

    async def compute_h2h_features(
        self,
        home_team_id: int,
        away_team_id: int,
        kickoff_utc: datetime,
        limit: int = 10,
    ) -> Optional[H2HFeatures]:
        """Compute Tier 3 H2H features from public.matches.

        PIT-SAFE: Only uses matches with date < kickoff_utc.

        Args:
            home_team_id: Home team ID for this match
            away_team_id: Away team ID for this match
            kickoff_utc: Target match kickoff (for PIT filter)
            limit: Max H2H matches to analyze

        Returns:
            H2HFeatures or None if no H2H history
        """
        query = text("""
            SELECT
                home_team_id,
                away_team_id,
                home_goals,
                away_goals
            FROM public.matches
            WHERE ((home_team_id = :home_id AND away_team_id = :away_id)
                OR (home_team_id = :away_id AND away_team_id = :home_id))
              AND date < :kickoff
              AND status IN ('FT', 'AET', 'PEN')
            ORDER BY date DESC
            LIMIT :limit
        """)

        result = await self.session.execute(query, {
            "home_id": home_team_id,
            "away_id": away_team_id,
            "kickoff": kickoff_utc,
            "limit": limit,
        })
        rows = result.fetchall()

        if not rows:
            return None

        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = 0
        away_goals = 0

        for row in rows:
            match_home_id, match_away_id = row[0], row[1]
            match_home_goals, match_away_goals = row[2], row[3]

            if match_home_id == home_team_id:
                # Normal orientation
                home_goals += match_home_goals
                away_goals += match_away_goals
                if match_home_goals > match_away_goals:
                    home_wins += 1
                elif match_home_goals < match_away_goals:
                    away_wins += 1
                else:
                    draws += 1
            else:
                # Reversed orientation
                home_goals += match_away_goals
                away_goals += match_home_goals
                if match_away_goals > match_home_goals:
                    home_wins += 1
                elif match_away_goals < match_home_goals:
                    away_wins += 1
                else:
                    draws += 1

        return H2HFeatures(
            h2h_total_matches=len(rows),
            h2h_home_wins=home_wins,
            h2h_draws=draws,
            h2h_away_wins=away_wins,
            h2h_home_goals=home_goals,
            h2h_away_goals=away_goals,
            captured_at=_utc_now(),
        )

    async def insert_row(
        self,
        match_id: int,
        kickoff_utc: datetime,
        competition_id: int,
        season: int,
        home_team_id: int,
        away_team_id: int,
        odds: Optional[OddsFeatures],
        form_home: Optional[FormFeatures] = None,
        form_away: Optional[FormFeatures] = None,
        h2h: Optional[H2HFeatures] = None,
    ) -> bool:
        """Insert or update feature_matrix row with policy enforcement.

        Args:
            match_id: API-Football match ID
            kickoff_utc: Match kickoff time
            competition_id: Competition ID
            season: Season year
            home_team_id: Home team ID
            away_team_id: Away team ID
            odds: Tier 1 odds features
            form_home: Tier 2 form for home team
            form_away: Tier 2 form for away team
            h2h: Tier 3 H2H features

        Returns:
            True if inserted, False if skipped

        Raises:
            PITViolationError: If data violates PIT constraint
            InsertionPolicyViolation: If policy requirements not met
        """
        # Check insertion policy
        should_insert, reason = should_insert_feature_row(odds, form_home, h2h)
        if not should_insert:
            logger.info(f"Skipping match {match_id}: {reason}")
            return False

        # Compute pit_max_captured_at
        pit_max = compute_pit_max(odds, form_home, h2h)

        # Validate PIT constraint
        if pit_max >= kickoff_utc:
            raise PITViolationError(
                f"PIT violation for match {match_id}: "
                f"pit_max={pit_max} >= kickoff={kickoff_utc}"
            )

        # Build insert values
        values = {
            "match_id": match_id,
            "kickoff_utc": kickoff_utc,
            "competition_id": competition_id,
            "season": season,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "pit_max_captured_at": pit_max,
            "tier1_complete": odds is not None,
            "tier2_complete": form_home is not None and form_away is not None,
            "tier3_complete": h2h is not None,
        }

        # Tier 1: Odds
        if odds:
            values.update({
                "odds_home_close": odds.odds_home_close,
                "odds_draw_close": odds.odds_draw_close,
                "odds_away_close": odds.odds_away_close,
                "implied_prob_home": odds.implied_prob_home,
                "implied_prob_draw": odds.implied_prob_draw,
                "implied_prob_away": odds.implied_prob_away,
                "odds_captured_at": odds.captured_at,
            })

        # Tier 2: Form (form_home = home team's form, form_away = away team's form)
        if form_home and form_away:
            values.update({
                "form_home_last5": form_home.form_last5,
                "form_away_last5": form_away.form_last5,
                "goals_home_last5": form_home.goals_scored_last5,
                "goals_away_last5": form_away.goals_scored_last5,
                "goals_against_home_last5": form_home.goals_conceded_last5,
                "goals_against_away_last5": form_away.goals_conceded_last5,
                "points_home_last5": form_home.points_last5,
                "points_away_last5": form_away.points_last5,
                "form_captured_at": max(form_home.captured_at, form_away.captured_at),
            })

        # Tier 3: H2H
        if h2h:
            values.update({
                "h2h_total_matches": h2h.h2h_total_matches,
                "h2h_home_wins": h2h.h2h_home_wins,
                "h2h_draws": h2h.h2h_draws,
                "h2h_away_wins": h2h.h2h_away_wins,
                "h2h_home_goals": h2h.h2h_home_goals,
                "h2h_away_goals": h2h.h2h_away_goals,
                "h2h_captured_at": h2h.captured_at,
            })

        # UPSERT
        query = text(f"""
            INSERT INTO {self.schema}.feature_matrix (
                match_id, kickoff_utc, competition_id, season, home_team_id, away_team_id,
                odds_home_close, odds_draw_close, odds_away_close,
                implied_prob_home, implied_prob_draw, implied_prob_away, odds_captured_at,
                form_home_last5, form_away_last5, goals_home_last5, goals_away_last5,
                goals_against_home_last5, goals_against_away_last5,
                points_home_last5, points_away_last5, form_captured_at,
                h2h_total_matches, h2h_home_wins, h2h_draws, h2h_away_wins,
                h2h_home_goals, h2h_away_goals, h2h_captured_at,
                pit_max_captured_at, tier1_complete, tier2_complete, tier3_complete
            ) VALUES (
                :match_id, :kickoff_utc, :competition_id, :season, :home_team_id, :away_team_id,
                :odds_home_close, :odds_draw_close, :odds_away_close,
                :implied_prob_home, :implied_prob_draw, :implied_prob_away, :odds_captured_at,
                :form_home_last5, :form_away_last5, :goals_home_last5, :goals_away_last5,
                :goals_against_home_last5, :goals_against_away_last5,
                :points_home_last5, :points_away_last5, :form_captured_at,
                :h2h_total_matches, :h2h_home_wins, :h2h_draws, :h2h_away_wins,
                :h2h_home_goals, :h2h_away_goals, :h2h_captured_at,
                :pit_max_captured_at, :tier1_complete, :tier2_complete, :tier3_complete
            )
            ON CONFLICT (match_id) DO UPDATE SET
                odds_home_close = COALESCE(EXCLUDED.odds_home_close, {self.schema}.feature_matrix.odds_home_close),
                odds_draw_close = COALESCE(EXCLUDED.odds_draw_close, {self.schema}.feature_matrix.odds_draw_close),
                odds_away_close = COALESCE(EXCLUDED.odds_away_close, {self.schema}.feature_matrix.odds_away_close),
                implied_prob_home = COALESCE(EXCLUDED.implied_prob_home, {self.schema}.feature_matrix.implied_prob_home),
                implied_prob_draw = COALESCE(EXCLUDED.implied_prob_draw, {self.schema}.feature_matrix.implied_prob_draw),
                implied_prob_away = COALESCE(EXCLUDED.implied_prob_away, {self.schema}.feature_matrix.implied_prob_away),
                odds_captured_at = COALESCE(EXCLUDED.odds_captured_at, {self.schema}.feature_matrix.odds_captured_at),
                form_home_last5 = COALESCE(EXCLUDED.form_home_last5, {self.schema}.feature_matrix.form_home_last5),
                form_away_last5 = COALESCE(EXCLUDED.form_away_last5, {self.schema}.feature_matrix.form_away_last5),
                goals_home_last5 = COALESCE(EXCLUDED.goals_home_last5, {self.schema}.feature_matrix.goals_home_last5),
                goals_away_last5 = COALESCE(EXCLUDED.goals_away_last5, {self.schema}.feature_matrix.goals_away_last5),
                goals_against_home_last5 = COALESCE(EXCLUDED.goals_against_home_last5, {self.schema}.feature_matrix.goals_against_home_last5),
                goals_against_away_last5 = COALESCE(EXCLUDED.goals_against_away_last5, {self.schema}.feature_matrix.goals_against_away_last5),
                points_home_last5 = COALESCE(EXCLUDED.points_home_last5, {self.schema}.feature_matrix.points_home_last5),
                points_away_last5 = COALESCE(EXCLUDED.points_away_last5, {self.schema}.feature_matrix.points_away_last5),
                form_captured_at = COALESCE(EXCLUDED.form_captured_at, {self.schema}.feature_matrix.form_captured_at),
                h2h_total_matches = COALESCE(EXCLUDED.h2h_total_matches, {self.schema}.feature_matrix.h2h_total_matches),
                h2h_home_wins = COALESCE(EXCLUDED.h2h_home_wins, {self.schema}.feature_matrix.h2h_home_wins),
                h2h_draws = COALESCE(EXCLUDED.h2h_draws, {self.schema}.feature_matrix.h2h_draws),
                h2h_away_wins = COALESCE(EXCLUDED.h2h_away_wins, {self.schema}.feature_matrix.h2h_away_wins),
                h2h_home_goals = COALESCE(EXCLUDED.h2h_home_goals, {self.schema}.feature_matrix.h2h_home_goals),
                h2h_away_goals = COALESCE(EXCLUDED.h2h_away_goals, {self.schema}.feature_matrix.h2h_away_goals),
                h2h_captured_at = COALESCE(EXCLUDED.h2h_captured_at, {self.schema}.feature_matrix.h2h_captured_at),
                pit_max_captured_at = GREATEST(EXCLUDED.pit_max_captured_at, {self.schema}.feature_matrix.pit_max_captured_at),
                tier1_complete = EXCLUDED.tier1_complete OR {self.schema}.feature_matrix.tier1_complete,
                tier2_complete = EXCLUDED.tier2_complete OR {self.schema}.feature_matrix.tier2_complete,
                tier3_complete = EXCLUDED.tier3_complete OR {self.schema}.feature_matrix.tier3_complete
        """)

        # Set NULL for missing optional values
        for key in ["odds_home_close", "odds_draw_close", "odds_away_close",
                    "implied_prob_home", "implied_prob_draw", "implied_prob_away",
                    "odds_captured_at", "form_home_last5", "form_away_last5",
                    "goals_home_last5", "goals_away_last5", "goals_against_home_last5",
                    "goals_against_away_last5", "points_home_last5", "points_away_last5",
                    "form_captured_at", "h2h_total_matches", "h2h_home_wins", "h2h_draws",
                    "h2h_away_wins", "h2h_home_goals", "h2h_away_goals", "h2h_captured_at"]:
            if key not in values:
                values[key] = None

        await self.session.execute(query, values)
        await self.session.commit()

        logger.info(f"Inserted feature_matrix row for match {match_id}")
        return True

    async def get_pit_stats(self) -> dict:
        """Get PIT compliance statistics for dashboard.

        Returns:
            Dict with violation count, coverage stats, etc.
        """
        query = text(f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(*) FILTER (WHERE tier1_complete) as tier1_count,
                COUNT(*) FILTER (WHERE tier2_complete) as tier2_count,
                COUNT(*) FILTER (WHERE tier3_complete) as tier3_count,
                COUNT(*) FILTER (WHERE pit_max_captured_at >= kickoff_utc) as pit_violations,
                COUNT(*) FILTER (WHERE outcome IS NOT NULL) as with_outcome,
                MIN(kickoff_utc) as earliest_match,
                MAX(kickoff_utc) as latest_match
            FROM {self.schema}.feature_matrix
        """)

        result = await self.session.execute(query)
        row = result.fetchone()

        return {
            "total_rows": row[0] or 0,
            "tier1_complete": row[1] or 0,
            "tier2_complete": row[2] or 0,
            "tier3_complete": row[3] or 0,
            "pit_violations": row[4] or 0,
            "with_outcome": row[5] or 0,
            "earliest_match": row[6].isoformat() if row[6] else None,
            "latest_match": row[7].isoformat() if row[7] else None,
            "tier1_coverage_pct": round((row[1] or 0) / row[0] * 100, 1) if row[0] else 0,
            "tier2_coverage_pct": round((row[2] or 0) / row[0] * 100, 1) if row[0] else 0,
            "tier3_coverage_pct": round((row[3] or 0) / row[0] * 100, 1) if row[0] else 0,
        }
