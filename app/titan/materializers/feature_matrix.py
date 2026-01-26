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
class XGFeatures:
    """Tier 1b: xG features (Understat).

    Rolling averages over last N matches per team.
    Naming convention: *_last5 clarifies these are rolling windows.
    """

    xg_home_last5: Decimal  # Home team avg xG
    xg_away_last5: Decimal  # Away team avg xG
    xga_home_last5: Decimal  # Home team avg xG Against
    xga_away_last5: Decimal  # Away team avg xG Against
    npxg_home_last5: Decimal  # Home team avg non-penalty xG
    npxg_away_last5: Decimal  # Away team avg non-penalty xG
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


@dataclass
class SofaScoreLineupFeatures:
    """Tier 1c: SofaScore lineup features (via SOTA tables).

    FASE 3A: Read from public.match_sofascore_lineup and match_sofascore_player.
    Integrity score is derived from: formation_present + starters==11.
    """

    sofascore_lineup_available: bool
    sofascore_home_formation: Optional[str]
    sofascore_away_formation: Optional[str]
    lineup_home_starters_count: Optional[int]
    lineup_away_starters_count: Optional[int]
    sofascore_lineup_integrity_score: Optional[Decimal]
    captured_at: Optional[datetime]


@dataclass
class XIDepthFeatures:
    """Tier 1d: XI depth features derived from lineup positions.

    FASE 3B-1: Read from public.match_sofascore_player.
    Position counts for DEF/MID/FWD per team (starters only).
    """

    xi_home_def_count: int
    xi_home_mid_count: int
    xi_home_fwd_count: int
    xi_away_def_count: int
    xi_away_mid_count: int
    xi_away_fwd_count: int
    xi_formation_mismatch_flag: bool
    captured_at: Optional[datetime]  # PIT timestamp, None if no player data


def detect_formation_mismatch(
    formation: Optional[str],
    def_count: int,
    mid_count: int,
    fwd_count: int,
) -> bool:
    """
    Detect if XI position counts differ from declared formation.

    Example: formation="4-3-3" expects DEF=4, MID=3, FWD=3
    If actual XI has DEF=5, MID=4, FWD=1 → mismatch=TRUE

    Tolerance: ±1 per line (hybrid formations like 4-2-3-1 can vary).

    Args:
        formation: Declared formation string (e.g., "4-3-3", "4-2-3-1")
        def_count: Actual defender starters count
        mid_count: Actual midfielder starters count
        fwd_count: Actual forward starters count

    Returns:
        True if mismatch detected, False otherwise (including when no formation)
    """
    if not formation:
        return False  # No formation → no mismatch detectable

    # Parse formation (e.g., "4-3-3" → [4, 3, 3], "4-2-3-1" → [4, 2, 3, 1])
    parts = formation.replace("-", "").strip()
    if len(parts) < 3:
        return False  # Invalid formation format

    try:
        expected_def = int(parts[0])
        # Middle sections are midfielders (can be multiple in formations like 4-2-3-1)
        expected_mid = sum(int(p) for p in parts[1:-1]) if len(parts) > 3 else int(parts[1])
        expected_fwd = int(parts[-1])
    except (ValueError, IndexError):
        return False  # Can't parse → no mismatch detectable

    # Check with tolerance of ±1 per line
    def_ok = abs(def_count - expected_def) <= 1
    mid_ok = abs(mid_count - expected_mid) <= 1
    fwd_ok = abs(fwd_count - expected_fwd) <= 1

    return not (def_ok and mid_ok and fwd_ok)


def should_insert_feature_row(
    odds: Optional[OddsFeatures],
    form: Optional[FormFeatures],
    h2h: Optional[H2HFeatures],
    xg: Optional[XGFeatures] = None,
    lineup: Optional["SofaScoreLineupFeatures"] = None,
    xi_depth: Optional["XIDepthFeatures"] = None,
) -> tuple[bool, str]:
    """
    Insertion policy for feature_matrix.

    RULE 1: No odds -> DON'T insert (Tier 1 is mandatory gate)
    RULE 2: With odds, missing Tier 1b/1c/1d/2/3 -> DO insert with NULLs

    Args:
        odds: Tier 1 odds features (or None)
        form: Tier 2 form features (or None)
        h2h: Tier 3 H2H features (or None)
        xg: Tier 1b xG features (or None) - optional enrichment
        lineup: Tier 1c SofaScore lineup features (or None) - optional enrichment
        xi_depth: Tier 1d XI depth features (or None) - optional enrichment

    Returns:
        (should_insert, reason)
    """
    # RULE 1: Without odds -> NO insert
    if odds is None:
        return False, "Missing Tier 1 (odds) - skipping"

    # RULE 2: With odds, missing Tier 1b/1c/1d/2/3 -> YES insert with NULLs
    return True, "Tier 1 complete, inserting (Tier 1b/1c/1d/2/3 may be NULL)"


def compute_pit_max(
    odds: Optional[OddsFeatures],
    form: Optional[FormFeatures],
    h2h: Optional[H2HFeatures],
    xg: Optional[XGFeatures] = None,
    lineup: Optional["SofaScoreLineupFeatures"] = None,
    xi_depth: Optional["XIDepthFeatures"] = None,
) -> datetime:
    """
    Compute pit_max_captured_at.

    If only odds: pit_max = odds.captured_at
    If multiple: pit_max = max(all valid captured_at)

    Args:
        odds: Tier 1 features (MUST have at least this per insertion policy)
        form: Tier 2 features (optional)
        h2h: Tier 3 features (optional)
        xg: Tier 1b xG features (optional)
        lineup: Tier 1c SofaScore lineup features (optional)
        xi_depth: Tier 1d XI depth features (optional)

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
    if xg and xg.captured_at:
        timestamps.append(xg.captured_at)
    if lineup and lineup.captured_at:
        timestamps.append(lineup.captured_at)
    if xi_depth and xi_depth.captured_at:
        timestamps.append(xi_depth.captured_at)

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

        # public.matches.date is TIMESTAMP (naive), so strip timezone for comparison
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        result = await self.session.execute(query, {
            "team_id": team_id,
            "kickoff": kickoff_naive,
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

        # public.matches.date is TIMESTAMP (naive), so strip timezone for comparison
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        result = await self.session.execute(query, {
            "home_id": home_team_id,
            "away_id": away_team_id,
            "kickoff": kickoff_naive,
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

    async def compute_xg_last5_features(
        self,
        home_team_id: int,
        away_team_id: int,
        kickoff_utc: datetime,
        limit: int = None,
    ) -> Optional[XGFeatures]:
        """Compute Tier 1b xG features from public.match_understat_team.

        Uses proper subquery pattern per auditor requirement:
        SELECT AVG(...) FROM (SELECT ... ORDER BY date DESC LIMIT N) t

        PIT-SAFE: Only uses matches with date < kickoff_utc.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            kickoff_utc: Target match kickoff (for PIT filter)
            limit: Number of recent matches (default from config.XG_ROLLING_WINDOW)

        Returns:
            XGFeatures or None if insufficient data
        """
        if limit is None:
            limit = titan_settings.XG_ROLLING_WINDOW

        # Query for home team xG (using subquery for proper AVG with ORDER BY/LIMIT)
        home_query = text("""
            SELECT
                AVG(t.xg) as avg_xg,
                AVG(t.xga) as avg_xga,
                AVG(t.npxg) as avg_npxg,
                COUNT(*) as match_count
            FROM (
                SELECT
                    CASE WHEN m.home_team_id = :team_id THEN mut.xg_home ELSE mut.xg_away END as xg,
                    CASE WHEN m.home_team_id = :team_id THEN mut.xga_home ELSE mut.xga_away END as xga,
                    CASE WHEN m.home_team_id = :team_id THEN mut.npxg_home ELSE mut.npxg_away END as npxg
                FROM public.matches m
                JOIN public.match_understat_team mut ON m.id = mut.match_id
                WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
                  AND m.date < :kickoff
                  AND m.status IN ('FT', 'AET', 'PEN')
                ORDER BY m.date DESC
                LIMIT :limit
            ) t
        """)

        # Query for away team xG
        away_query = text("""
            SELECT
                AVG(t.xg) as avg_xg,
                AVG(t.xga) as avg_xga,
                AVG(t.npxg) as avg_npxg,
                COUNT(*) as match_count
            FROM (
                SELECT
                    CASE WHEN m.home_team_id = :team_id THEN mut.xg_home ELSE mut.xg_away END as xg,
                    CASE WHEN m.home_team_id = :team_id THEN mut.xga_home ELSE mut.xga_away END as xga,
                    CASE WHEN m.home_team_id = :team_id THEN mut.npxg_home ELSE mut.npxg_away END as npxg
                FROM public.matches m
                JOIN public.match_understat_team mut ON m.id = mut.match_id
                WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
                  AND m.date < :kickoff
                  AND m.status IN ('FT', 'AET', 'PEN')
                ORDER BY m.date DESC
                LIMIT :limit
            ) t
        """)

        # public.matches.date is TIMESTAMP (naive), so strip timezone for comparison
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        # Execute both queries
        home_result = await self.session.execute(home_query, {
            "team_id": home_team_id,
            "kickoff": kickoff_naive,
            "limit": limit,
        })
        home_row = home_result.fetchone()

        away_result = await self.session.execute(away_query, {
            "team_id": away_team_id,
            "kickoff": kickoff_naive,
            "limit": limit,
        })
        away_row = away_result.fetchone()

        # Check if we have enough data for both teams
        home_count = home_row[3] if home_row else 0
        away_count = away_row[3] if away_row else 0

        if home_count < limit or away_count < limit:
            logger.debug(
                f"Insufficient xG data for match: home={home_count}/{limit}, away={away_count}/{limit}"
            )
            return None

        # Check for NULL values (shouldn't happen but defensive)
        if home_row[0] is None or away_row[0] is None:
            logger.warning("xG query returned NULL values despite having enough matches")
            return None

        return XGFeatures(
            xg_home_last5=Decimal(str(round(home_row[0], 2))),
            xg_away_last5=Decimal(str(round(away_row[0], 2))),
            xga_home_last5=Decimal(str(round(home_row[1], 2))),
            xga_away_last5=Decimal(str(round(away_row[1], 2))),
            npxg_home_last5=Decimal(str(round(home_row[2], 2))),
            npxg_away_last5=Decimal(str(round(away_row[2], 2))),
            captured_at=_utc_now(),
        )

    async def compute_lineup_features(
        self,
        match_id: int,
        kickoff_utc: datetime,
    ) -> Optional[SofaScoreLineupFeatures]:
        """Compute Tier 1c SofaScore lineup features from SOTA tables.

        Reads from public.match_sofascore_lineup and public.match_sofascore_player.
        PIT-safe: Only returns data where captured_at < kickoff_utc.
        Fail-open: Returns None if no lineup found (normal pre-KO).

        Integrity score calculation:
        - formation_present: 1.0 if both formations exist, 0.0 otherwise
        - starters_complete: (home_starters==11 + away_starters==11) / 2
        - integrity_score = (formation_present + starters_complete) / 2

        Args:
            match_id: Internal match ID (public.matches.id, NOT external_id)
            kickoff_utc: Match kickoff time (aware UTC)

        Returns:
            SofaScoreLineupFeatures if found and PIT-compliant, None otherwise.
        """
        # public.* tables use TIMESTAMP (naive) - apply timezone normalization
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        # Query lineup + player counts in single query
        query = text("""
            WITH lineup_data AS (
                SELECT
                    msl.team_side,
                    msl.formation,
                    msl.captured_at,
                    COUNT(*) FILTER (WHERE msp.is_starter = TRUE) as starters_count
                FROM public.match_sofascore_lineup msl
                LEFT JOIN public.match_sofascore_player msp
                    ON msl.match_id = msp.match_id
                   AND msl.team_side = msp.team_side
                WHERE msl.match_id = :match_id
                  AND msl.captured_at < :kickoff
                GROUP BY msl.match_id, msl.team_side, msl.formation, msl.captured_at
            )
            SELECT
                home.formation as home_formation,
                away.formation as away_formation,
                home.starters_count as home_starters,
                away.starters_count as away_starters,
                GREATEST(home.captured_at, away.captured_at) as captured_at
            FROM lineup_data home
            JOIN lineup_data away ON TRUE
            WHERE home.team_side = 'home'
              AND away.team_side = 'away'
            LIMIT 1
        """)

        result = await self.session.execute(query, {
            "match_id": match_id,
            "kickoff": kickoff_naive,
        })
        row = result.fetchone()

        if not row:
            return None  # Fail-open: no lineup yet (normal pre-KO)

        home_formation, away_formation, home_starters, away_starters, captured_at = row

        # Calculate integrity score (0.000-1.000)
        # Component 1: formation_present (both formations exist)
        formation_present = 1.0 if (home_formation and away_formation) else 0.0

        # Component 2: starters_complete (both have 11 starters)
        home_complete = 1.0 if home_starters == 11 else 0.0
        away_complete = 1.0 if away_starters == 11 else 0.0
        starters_complete = (home_complete + away_complete) / 2

        # Final score: average of both components
        integrity_score = Decimal(str((formation_present + starters_complete) / 2)).quantize(
            Decimal("0.001")
        )

        # Convert captured_at to aware (for titan.* storage)
        captured_at_aware = captured_at.replace(tzinfo=timezone.utc) if captured_at else None

        return SofaScoreLineupFeatures(
            sofascore_lineup_available=True,
            sofascore_home_formation=home_formation,
            sofascore_away_formation=away_formation,
            lineup_home_starters_count=home_starters,
            lineup_away_starters_count=away_starters,
            sofascore_lineup_integrity_score=integrity_score,
            captured_at=captured_at_aware,
        )

    async def compute_xi_depth_features(
        self,
        match_id: int,
        kickoff_utc: datetime,
        home_formation: Optional[str] = None,
        away_formation: Optional[str] = None,
    ) -> Optional[XIDepthFeatures]:
        """Compute Tier 1d XI depth features from SOTA player positions.

        Reads from public.match_sofascore_player.
        PIT-safe: Only uses data where captured_at < kickoff_utc.
        Fail-open: Returns None if no player data found.

        Args:
            match_id: Internal match ID (public.matches.id)
            kickoff_utc: Match kickoff time (aware UTC)
            home_formation: Home formation from Tier 1c (optional, for mismatch detection)
            away_formation: Away formation from Tier 1c (optional)

        Returns:
            XIDepthFeatures if found and PIT-compliant, None otherwise.
        """
        # public.* tables use TIMESTAMP (naive) - apply timezone normalization
        kickoff_naive = kickoff_utc.replace(tzinfo=None) if kickoff_utc.tzinfo else kickoff_utc

        query = text("""
            SELECT
                team_side,
                position,
                COUNT(*) as count,
                MAX(captured_at) as captured_at
            FROM public.match_sofascore_player
            WHERE match_id = :match_id
              AND is_starter = TRUE
              AND captured_at < :kickoff
            GROUP BY team_side, position
        """)

        result = await self.session.execute(query, {
            "match_id": match_id,
            "kickoff": kickoff_naive,
        })
        rows = result.fetchall()

        if not rows:
            return None  # Fail-open: no player data

        # Aggregate counts by team and position
        counts = {"home": {"DEF": 0, "MID": 0, "FWD": 0}, "away": {"DEF": 0, "MID": 0, "FWD": 0}}
        latest_captured = None

        for row in rows:
            team_side, position, count, captured_at = row
            if team_side in counts and position in counts[team_side]:
                counts[team_side][position] = count
            if captured_at and (latest_captured is None or captured_at > latest_captured):
                latest_captured = captured_at

        # Detect formation mismatch
        home_mismatch = detect_formation_mismatch(
            home_formation,
            counts["home"]["DEF"],
            counts["home"]["MID"],
            counts["home"]["FWD"],
        )
        away_mismatch = detect_formation_mismatch(
            away_formation,
            counts["away"]["DEF"],
            counts["away"]["MID"],
            counts["away"]["FWD"],
        )

        # Convert captured_at to aware (for titan.* storage)
        captured_aware = latest_captured.replace(tzinfo=timezone.utc) if latest_captured else None

        return XIDepthFeatures(
            xi_home_def_count=counts["home"]["DEF"],
            xi_home_mid_count=counts["home"]["MID"],
            xi_home_fwd_count=counts["home"]["FWD"],
            xi_away_def_count=counts["away"]["DEF"],
            xi_away_mid_count=counts["away"]["MID"],
            xi_away_fwd_count=counts["away"]["FWD"],
            xi_formation_mismatch_flag=home_mismatch or away_mismatch,
            captured_at=captured_aware,
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
        xg: Optional[XGFeatures] = None,
        lineup: Optional[SofaScoreLineupFeatures] = None,
        xi_depth: Optional[XIDepthFeatures] = None,
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
            xg: Tier 1b xG features (optional enrichment)
            lineup: Tier 1c SofaScore lineup features (optional enrichment)
            xi_depth: Tier 1d XI depth features (optional enrichment)

        Returns:
            True if inserted, False if skipped

        Raises:
            PITViolationError: If data violates PIT constraint
            InsertionPolicyViolation: If policy requirements not met
        """
        # Check insertion policy
        should_insert, reason = should_insert_feature_row(odds, form_home, h2h, xg, lineup, xi_depth)
        if not should_insert:
            logger.info(f"Skipping match {match_id}: {reason}")
            return False

        # Compute pit_max_captured_at
        pit_max = compute_pit_max(odds, form_home, h2h, xg, lineup, xi_depth)

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
            "tier1b_complete": xg is not None,
            "tier1c_complete": lineup is not None and lineup.captured_at is not None,
            "tier1d_complete": xi_depth is not None and xi_depth.captured_at is not None,
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

        # Tier 1b: xG (Understat)
        if xg:
            values.update({
                "xg_home_last5": xg.xg_home_last5,
                "xg_away_last5": xg.xg_away_last5,
                "xga_home_last5": xg.xga_home_last5,
                "xga_away_last5": xg.xga_away_last5,
                "npxg_home_last5": xg.npxg_home_last5,
                "npxg_away_last5": xg.npxg_away_last5,
                "xg_captured_at": xg.captured_at,
            })

        # Tier 1c: SofaScore Lineup (via SOTA)
        if lineup:
            values.update({
                "sofascore_lineup_available": lineup.sofascore_lineup_available,
                "sofascore_home_formation": lineup.sofascore_home_formation,
                "sofascore_away_formation": lineup.sofascore_away_formation,
                "lineup_home_starters_count": lineup.lineup_home_starters_count,
                "lineup_away_starters_count": lineup.lineup_away_starters_count,
                "sofascore_lineup_integrity_score": lineup.sofascore_lineup_integrity_score,
                "sofascore_lineup_captured_at": lineup.captured_at,
            })
        else:
            values.update({
                "sofascore_lineup_available": False,
                "sofascore_home_formation": None,
                "sofascore_away_formation": None,
                "lineup_home_starters_count": None,
                "lineup_away_starters_count": None,
                "sofascore_lineup_integrity_score": None,
                "sofascore_lineup_captured_at": None,
            })

        # Tier 1d: XI Depth (via SOTA)
        if xi_depth:
            values.update({
                "xi_home_def_count": xi_depth.xi_home_def_count,
                "xi_home_mid_count": xi_depth.xi_home_mid_count,
                "xi_home_fwd_count": xi_depth.xi_home_fwd_count,
                "xi_away_def_count": xi_depth.xi_away_def_count,
                "xi_away_mid_count": xi_depth.xi_away_mid_count,
                "xi_away_fwd_count": xi_depth.xi_away_fwd_count,
                "xi_formation_mismatch_flag": xi_depth.xi_formation_mismatch_flag,
                "xi_depth_captured_at": xi_depth.captured_at,
            })
        else:
            values.update({
                "xi_home_def_count": None,
                "xi_home_mid_count": None,
                "xi_home_fwd_count": None,
                "xi_away_def_count": None,
                "xi_away_mid_count": None,
                "xi_away_fwd_count": None,
                "xi_formation_mismatch_flag": False,
                "xi_depth_captured_at": None,
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
                xg_home_last5, xg_away_last5, xga_home_last5, xga_away_last5,
                npxg_home_last5, npxg_away_last5, xg_captured_at,
                sofascore_lineup_available, sofascore_home_formation, sofascore_away_formation,
                lineup_home_starters_count, lineup_away_starters_count,
                sofascore_lineup_integrity_score, sofascore_lineup_captured_at,
                xi_home_def_count, xi_home_mid_count, xi_home_fwd_count,
                xi_away_def_count, xi_away_mid_count, xi_away_fwd_count,
                xi_formation_mismatch_flag, xi_depth_captured_at,
                form_home_last5, form_away_last5, goals_home_last5, goals_away_last5,
                goals_against_home_last5, goals_against_away_last5,
                points_home_last5, points_away_last5, form_captured_at,
                h2h_total_matches, h2h_home_wins, h2h_draws, h2h_away_wins,
                h2h_home_goals, h2h_away_goals, h2h_captured_at,
                pit_max_captured_at, tier1_complete, tier1b_complete, tier1c_complete, tier1d_complete, tier2_complete, tier3_complete
            ) VALUES (
                :match_id, :kickoff_utc, :competition_id, :season, :home_team_id, :away_team_id,
                :odds_home_close, :odds_draw_close, :odds_away_close,
                :implied_prob_home, :implied_prob_draw, :implied_prob_away, :odds_captured_at,
                :xg_home_last5, :xg_away_last5, :xga_home_last5, :xga_away_last5,
                :npxg_home_last5, :npxg_away_last5, :xg_captured_at,
                :sofascore_lineup_available, :sofascore_home_formation, :sofascore_away_formation,
                :lineup_home_starters_count, :lineup_away_starters_count,
                :sofascore_lineup_integrity_score, :sofascore_lineup_captured_at,
                :xi_home_def_count, :xi_home_mid_count, :xi_home_fwd_count,
                :xi_away_def_count, :xi_away_mid_count, :xi_away_fwd_count,
                :xi_formation_mismatch_flag, :xi_depth_captured_at,
                :form_home_last5, :form_away_last5, :goals_home_last5, :goals_away_last5,
                :goals_against_home_last5, :goals_against_away_last5,
                :points_home_last5, :points_away_last5, :form_captured_at,
                :h2h_total_matches, :h2h_home_wins, :h2h_draws, :h2h_away_wins,
                :h2h_home_goals, :h2h_away_goals, :h2h_captured_at,
                :pit_max_captured_at, :tier1_complete, :tier1b_complete, :tier1c_complete, :tier1d_complete, :tier2_complete, :tier3_complete
            )
            ON CONFLICT (match_id) DO UPDATE SET
                odds_home_close = COALESCE(EXCLUDED.odds_home_close, {self.schema}.feature_matrix.odds_home_close),
                odds_draw_close = COALESCE(EXCLUDED.odds_draw_close, {self.schema}.feature_matrix.odds_draw_close),
                odds_away_close = COALESCE(EXCLUDED.odds_away_close, {self.schema}.feature_matrix.odds_away_close),
                implied_prob_home = COALESCE(EXCLUDED.implied_prob_home, {self.schema}.feature_matrix.implied_prob_home),
                implied_prob_draw = COALESCE(EXCLUDED.implied_prob_draw, {self.schema}.feature_matrix.implied_prob_draw),
                implied_prob_away = COALESCE(EXCLUDED.implied_prob_away, {self.schema}.feature_matrix.implied_prob_away),
                odds_captured_at = COALESCE(EXCLUDED.odds_captured_at, {self.schema}.feature_matrix.odds_captured_at),
                xg_home_last5 = COALESCE(EXCLUDED.xg_home_last5, {self.schema}.feature_matrix.xg_home_last5),
                xg_away_last5 = COALESCE(EXCLUDED.xg_away_last5, {self.schema}.feature_matrix.xg_away_last5),
                xga_home_last5 = COALESCE(EXCLUDED.xga_home_last5, {self.schema}.feature_matrix.xga_home_last5),
                xga_away_last5 = COALESCE(EXCLUDED.xga_away_last5, {self.schema}.feature_matrix.xga_away_last5),
                npxg_home_last5 = COALESCE(EXCLUDED.npxg_home_last5, {self.schema}.feature_matrix.npxg_home_last5),
                npxg_away_last5 = COALESCE(EXCLUDED.npxg_away_last5, {self.schema}.feature_matrix.npxg_away_last5),
                xg_captured_at = COALESCE(EXCLUDED.xg_captured_at, {self.schema}.feature_matrix.xg_captured_at),
                sofascore_lineup_available = COALESCE(EXCLUDED.sofascore_lineup_available, {self.schema}.feature_matrix.sofascore_lineup_available),
                sofascore_home_formation = COALESCE(EXCLUDED.sofascore_home_formation, {self.schema}.feature_matrix.sofascore_home_formation),
                sofascore_away_formation = COALESCE(EXCLUDED.sofascore_away_formation, {self.schema}.feature_matrix.sofascore_away_formation),
                lineup_home_starters_count = COALESCE(EXCLUDED.lineup_home_starters_count, {self.schema}.feature_matrix.lineup_home_starters_count),
                lineup_away_starters_count = COALESCE(EXCLUDED.lineup_away_starters_count, {self.schema}.feature_matrix.lineup_away_starters_count),
                sofascore_lineup_integrity_score = COALESCE(EXCLUDED.sofascore_lineup_integrity_score, {self.schema}.feature_matrix.sofascore_lineup_integrity_score),
                sofascore_lineup_captured_at = COALESCE(EXCLUDED.sofascore_lineup_captured_at, {self.schema}.feature_matrix.sofascore_lineup_captured_at),
                xi_home_def_count = COALESCE(EXCLUDED.xi_home_def_count, {self.schema}.feature_matrix.xi_home_def_count),
                xi_home_mid_count = COALESCE(EXCLUDED.xi_home_mid_count, {self.schema}.feature_matrix.xi_home_mid_count),
                xi_home_fwd_count = COALESCE(EXCLUDED.xi_home_fwd_count, {self.schema}.feature_matrix.xi_home_fwd_count),
                xi_away_def_count = COALESCE(EXCLUDED.xi_away_def_count, {self.schema}.feature_matrix.xi_away_def_count),
                xi_away_mid_count = COALESCE(EXCLUDED.xi_away_mid_count, {self.schema}.feature_matrix.xi_away_mid_count),
                xi_away_fwd_count = COALESCE(EXCLUDED.xi_away_fwd_count, {self.schema}.feature_matrix.xi_away_fwd_count),
                xi_formation_mismatch_flag = COALESCE(EXCLUDED.xi_formation_mismatch_flag, {self.schema}.feature_matrix.xi_formation_mismatch_flag),
                xi_depth_captured_at = COALESCE(EXCLUDED.xi_depth_captured_at, {self.schema}.feature_matrix.xi_depth_captured_at),
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
                tier1b_complete = EXCLUDED.tier1b_complete OR {self.schema}.feature_matrix.tier1b_complete,
                tier1c_complete = EXCLUDED.tier1c_complete OR {self.schema}.feature_matrix.tier1c_complete,
                tier1d_complete = EXCLUDED.tier1d_complete OR {self.schema}.feature_matrix.tier1d_complete,
                tier2_complete = EXCLUDED.tier2_complete OR {self.schema}.feature_matrix.tier2_complete,
                tier3_complete = EXCLUDED.tier3_complete OR {self.schema}.feature_matrix.tier3_complete
        """)

        # Set NULL for missing optional values
        for key in ["odds_home_close", "odds_draw_close", "odds_away_close",
                    "implied_prob_home", "implied_prob_draw", "implied_prob_away",
                    "odds_captured_at",
                    "xg_home_last5", "xg_away_last5", "xga_home_last5", "xga_away_last5",
                    "npxg_home_last5", "npxg_away_last5", "xg_captured_at",
                    "sofascore_home_formation", "sofascore_away_formation",
                    "lineup_home_starters_count", "lineup_away_starters_count",
                    "sofascore_lineup_integrity_score", "sofascore_lineup_captured_at",
                    "xi_home_def_count", "xi_home_mid_count", "xi_home_fwd_count",
                    "xi_away_def_count", "xi_away_mid_count", "xi_away_fwd_count",
                    "xi_formation_mismatch_flag", "xi_depth_captured_at",
                    "form_home_last5", "form_away_last5",
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
                COUNT(*) FILTER (WHERE tier1b_complete) as tier1b_count,
                COUNT(*) FILTER (WHERE tier1c_complete) as tier1c_count,
                COUNT(*) FILTER (WHERE tier1d_complete) as tier1d_count,
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

        total = row[0] or 0
        return {
            "total_rows": total,
            "tier1_complete": row[1] or 0,
            "tier1b_complete": row[2] or 0,
            "tier1c_complete": row[3] or 0,
            "tier1d_complete": row[4] or 0,
            "tier2_complete": row[5] or 0,
            "tier3_complete": row[6] or 0,
            "pit_violations": row[7] or 0,
            "with_outcome": row[8] or 0,
            "earliest_match": row[9].isoformat() if row[9] else None,
            "latest_match": row[10].isoformat() if row[10] else None,
            "tier1_coverage_pct": round((row[1] or 0) / total * 100, 1) if total else 0,
            "tier1b_coverage_pct": round((row[2] or 0) / total * 100, 1) if total else 0,
            "tier1c_coverage_pct": round((row[3] or 0) / total * 100, 1) if total else 0,
            "tier1d_coverage_pct": round((row[4] or 0) / total * 100, 1) if total else 0,
            "tier2_coverage_pct": round((row[5] or 0) / total * 100, 1) if total else 0,
            "tier3_coverage_pct": round((row[6] or 0) / total * 100, 1) if total else 0,
        }
