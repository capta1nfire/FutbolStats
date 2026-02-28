"""
Aggregates service for computing league baselines and team profiles.

Calculates derived statistics from match data for narrative context.
All metrics are deterministic and verifiable.
"""

import logging
from datetime import datetime, date, timezone
from typing import Optional

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Match, Team, LeagueSeasonBaseline, LeagueTeamProfile

logger = logging.getLogger(__name__)

# Configuration
MIN_SAMPLE_MATCHES = 5  # Minimum matches for valid stats
HIGH_CONFIDENCE_MATCHES = 10  # Matches for "high" rank confidence


class AggregatesService:
    """Service for computing and retrieving league aggregates."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def compute_league_baseline(
        self,
        league_id: int,
        season: int,
        as_of_date: Optional[date] = None,
    ) -> Optional[LeagueSeasonBaseline]:
        """
        Compute baseline statistics for a league/season.

        Args:
            league_id: API-Football league ID
            season: Season year
            as_of_date: Date to compute stats up to (default: today)

        Returns:
            LeagueSeasonBaseline or None if insufficient data
        """
        if as_of_date is None:
            as_of_date = datetime.now(timezone.utc).date()

        as_of_datetime = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)

        # Query finished matches
        result = await self.session.execute(
            select(Match)
            .where(
                and_(
                    Match.league_id == league_id,
                    Match.season == season,
                    Match.status.in_(["FT", "AET", "PEN"]),
                    Match.date <= as_of_datetime,
                    Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None),
                )
            )
        )
        matches = result.scalars().all()

        if len(matches) < MIN_SAMPLE_MATCHES:
            logger.info(
                f"[AGGREGATES] Insufficient matches for league {league_id} season {season}: "
                f"{len(matches)} < {MIN_SAMPLE_MATCHES}"
            )
            return None

        # Calculate baseline metrics
        n = len(matches)
        total_goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in matches)
        goals_avg = total_goals / n

        # Over/Under percentages
        over_1_5 = sum(1 for m in matches if (m.home_goals or 0) + (m.away_goals or 0) > 1.5) / n
        over_2_5 = sum(1 for m in matches if (m.home_goals or 0) + (m.away_goals or 0) > 2.5) / n
        over_3_5 = sum(1 for m in matches if (m.home_goals or 0) + (m.away_goals or 0) > 3.5) / n

        # BTTS
        btts_yes = sum(1 for m in matches if (m.home_goals or 0) > 0 and (m.away_goals or 0) > 0) / n

        # Clean sheets
        clean_sheet_home = sum(1 for m in matches if (m.away_goals or 0) == 0) / n
        clean_sheet_away = sum(1 for m in matches if (m.home_goals or 0) == 0) / n

        # Failed to score
        fts_home = sum(1 for m in matches if (m.home_goals or 0) == 0) / n
        fts_away = sum(1 for m in matches if (m.away_goals or 0) == 0) / n

        # Cards and corners (from stats JSON if available)
        corners_total = 0
        corners_count = 0
        yellow_total = 0
        yellow_count = 0
        red_total = 0
        red_count = 0

        for m in matches:
            stats = m.stats or {}
            if not isinstance(stats, dict):
                stats = {}
            home_stats = stats.get("home", {})
            away_stats = stats.get("away", {})

            # Corners
            home_corners = home_stats.get("corner_kicks") or home_stats.get("corners")
            away_corners = away_stats.get("corner_kicks") or away_stats.get("corners")
            if home_corners is not None and away_corners is not None:
                corners_total += home_corners + away_corners
                corners_count += 1

            # Yellow cards
            home_yellow = home_stats.get("yellow_cards")
            away_yellow = away_stats.get("yellow_cards")
            if home_yellow is not None and away_yellow is not None:
                yellow_total += home_yellow + away_yellow
                yellow_count += 1

            # Red cards
            home_red = home_stats.get("red_cards")
            away_red = away_stats.get("red_cards")
            if home_red is not None and away_red is not None:
                red_total += home_red + away_red
                red_count += 1

        # Create or update baseline
        baseline = LeagueSeasonBaseline(
            league_id=league_id,
            season=season,
            as_of_date=as_of_datetime,
            sample_n_matches=n,
            goals_avg_per_match=round(goals_avg, 2),
            over_1_5_pct=round(over_1_5 * 100, 1),
            over_2_5_pct=round(over_2_5 * 100, 1),
            over_3_5_pct=round(over_3_5 * 100, 1),
            btts_yes_pct=round(btts_yes * 100, 1),
            clean_sheet_pct_home=round(clean_sheet_home * 100, 1),
            clean_sheet_pct_away=round(clean_sheet_away * 100, 1),
            failed_to_score_pct_home=round(fts_home * 100, 1),
            failed_to_score_pct_away=round(fts_away * 100, 1),
            corners_avg_per_match=round(corners_total / corners_count, 1) if corners_count > 0 else None,
            yellow_cards_avg_per_match=round(yellow_total / yellow_count, 1) if yellow_count > 0 else None,
            red_cards_avg_per_match=round(red_total / red_count, 2) if red_count > 0 else None,
            last_computed_at=datetime.now(timezone.utc),
        )

        # Upsert
        existing = await self.session.execute(
            select(LeagueSeasonBaseline)
            .where(
                and_(
                    LeagueSeasonBaseline.league_id == league_id,
                    LeagueSeasonBaseline.season == season,
                    func.date(LeagueSeasonBaseline.as_of_date) == as_of_date,
                )
            )
        )
        existing_baseline = existing.scalar_one_or_none()

        if existing_baseline:
            # Update existing
            for key, value in baseline.__dict__.items():
                if not key.startswith("_") and key != "id":
                    setattr(existing_baseline, key, value)
            baseline = existing_baseline
        else:
            self.session.add(baseline)

        await self.session.flush()

        logger.info(
            f"[AGGREGATES] Computed baseline for league {league_id} season {season}: "
            f"n={n}, goals_avg={goals_avg:.2f}, over_2_5={over_2_5*100:.1f}%"
        )

        return baseline

    async def compute_team_profiles(
        self,
        league_id: int,
        season: int,
        as_of_date: Optional[date] = None,
    ) -> list[LeagueTeamProfile]:
        """
        Compute team profiles for all teams in a league/season.

        Args:
            league_id: API-Football league ID
            season: Season year
            as_of_date: Date to compute stats up to (default: today)

        Returns:
            List of LeagueTeamProfile objects
        """
        if as_of_date is None:
            as_of_date = datetime.now(timezone.utc).date()

        as_of_datetime = datetime.combine(as_of_date, datetime.max.time(), tzinfo=timezone.utc)

        # Query finished matches
        result = await self.session.execute(
            select(Match)
            .where(
                and_(
                    Match.league_id == league_id,
                    Match.season == season,
                    Match.status.in_(["FT", "AET", "PEN"]),
                    Match.date <= as_of_datetime,
                    Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None),
                )
            )
        )
        matches = result.scalars().all()

        if not matches:
            return []

        # Group matches by team
        team_matches: dict[int, list[dict]] = {}

        for m in matches:
            # Home team perspective
            if m.home_team_id not in team_matches:
                team_matches[m.home_team_id] = []
            team_matches[m.home_team_id].append({
                "match": m,
                "is_home": True,
                "goals_for": m.home_goals or 0,
                "goals_against": m.away_goals or 0,
            })

            # Away team perspective
            if m.away_team_id not in team_matches:
                team_matches[m.away_team_id] = []
            team_matches[m.away_team_id].append({
                "match": m,
                "is_home": False,
                "goals_for": m.away_goals or 0,
                "goals_against": m.home_goals or 0,
            })

        # Calculate profiles
        profiles = []
        team_stats = []  # For ranking

        for team_id, team_data in team_matches.items():
            n = len(team_data)
            if n == 0:
                continue

            # Basic metrics
            goals_for = sum(d["goals_for"] for d in team_data)
            goals_against = sum(d["goals_against"] for d in team_data)

            goals_for_pm = goals_for / n
            goals_against_pm = goals_against / n
            goal_diff_pm = goals_for_pm - goals_against_pm

            # Percentages
            clean_sheets = sum(1 for d in team_data if d["goals_against"] == 0)
            failed_to_score = sum(1 for d in team_data if d["goals_for"] == 0)
            btts = sum(1 for d in team_data if d["goals_for"] > 0 and d["goals_against"] > 0)

            over_1_5 = sum(1 for d in team_data if d["goals_for"] + d["goals_against"] > 1.5)
            over_2_5 = sum(1 for d in team_data if d["goals_for"] + d["goals_against"] > 2.5)
            over_3_5 = sum(1 for d in team_data if d["goals_for"] + d["goals_against"] > 3.5)

            # Cards and corners from stats
            corners_for = 0
            corners_against = 0
            corners_count = 0
            yellow_total = 0
            yellow_count = 0
            red_total = 0
            red_count = 0

            # By-time goals (P1)
            goals_0_15 = 0
            goals_76_90p = 0
            conceded_0_15 = 0
            conceded_76_90p = 0
            total_goals_scored = goals_for
            total_goals_conceded = goals_against

            for d in team_data:
                m = d["match"]
                is_home = d["is_home"]
                stats = m.stats or {}
                events = m.events if isinstance(m.events, list) else []

                # Guard: some matches store stats as raw API-Football array
                # instead of normalised {home: {}, away: {}} dict
                if not isinstance(stats, dict):
                    stats = {}

                # Get team stats
                team_stats_data = stats.get("home" if is_home else "away", {})
                opp_stats_data = stats.get("away" if is_home else "home", {})

                # Corners
                team_corners = team_stats_data.get("corner_kicks") or team_stats_data.get("corners")
                opp_corners = opp_stats_data.get("corner_kicks") or opp_stats_data.get("corners")
                if team_corners is not None:
                    corners_for += team_corners
                    corners_count += 1
                if opp_corners is not None:
                    corners_against += opp_corners

                # Cards
                team_yellow = team_stats_data.get("yellow_cards")
                team_red = team_stats_data.get("red_cards")
                if team_yellow is not None:
                    yellow_total += team_yellow
                    yellow_count += 1
                if team_red is not None:
                    red_total += team_red

                # By-time goals from events
                team_external_id = m.home_team_id if is_home else m.away_team_id
                for event in events:
                    if event.get("type") != "Goal":
                        continue
                    minute = event.get("minute") or 0
                    event_team_id = event.get("team_id")

                    # Check if this team scored
                    is_team_goal = False
                    if event_team_id:
                        # Compare with match team_id (could be external or internal)
                        # We need to be flexible here
                        is_team_goal = event.get("team_name") and (
                            # Match by checking if home/away position matches
                            (is_home and event_team_id == m.home_team_id) or
                            (not is_home and event_team_id == m.away_team_id)
                        )

                    if is_team_goal:
                        if minute <= 15:
                            goals_0_15 += 1
                        elif minute >= 76:
                            goals_76_90p += 1
                    else:
                        # Opponent scored (this team conceded)
                        if minute <= 15:
                            conceded_0_15 += 1
                        elif minute >= 76:
                            conceded_76_90p += 1

            # Store for ranking
            team_stats.append({
                "team_id": team_id,
                "n": n,
                "goals_for_pm": goals_for_pm,
                "goals_against_pm": goals_against_pm,
                "goal_diff_pm": goal_diff_pm,
                "corners_for_pm": corners_for / corners_count if corners_count > 0 else None,
                "cards_pm": (yellow_total + 2 * red_total) / yellow_count if yellow_count > 0 else None,
                "clean_sheet_pct": clean_sheets / n * 100,
                "failed_to_score_pct": failed_to_score / n * 100,
                "btts_yes_pct": btts / n * 100,
                "over_1_5_pct": over_1_5 / n * 100,
                "over_2_5_pct": over_2_5 / n * 100,
                "over_3_5_pct": over_3_5 / n * 100,
                "corners_against_pm": corners_against / corners_count if corners_count > 0 else None,
                "yellow_cards_pm": yellow_total / yellow_count if yellow_count > 0 else None,
                "red_cards_pm": red_total / yellow_count if yellow_count > 0 else None,
                "goals_0_15_pct": goals_0_15 / total_goals_scored * 100 if total_goals_scored > 0 else None,
                "goals_76_90p_pct": goals_76_90p / total_goals_scored * 100 if total_goals_scored > 0 else None,
                "conceded_0_15_pct": conceded_0_15 / total_goals_conceded * 100 if total_goals_conceded > 0 else None,
                "conceded_76_90p_pct": conceded_76_90p / total_goals_conceded * 100 if total_goals_conceded > 0 else None,
            })

        # Calculate ranks (only for teams with min sample)
        eligible_teams = [t for t in team_stats if t["n"] >= MIN_SAMPLE_MATCHES]
        total_teams = len(eligible_teams)

        if eligible_teams:
            # Sort for each rank metric
            by_attack = sorted(eligible_teams, key=lambda x: x["goals_for_pm"], reverse=True)
            by_defense = sorted(eligible_teams, key=lambda x: x["goals_against_pm"], reverse=True)
            by_goal_diff = sorted(eligible_teams, key=lambda x: x["goal_diff_pm"], reverse=True)
            by_corners = sorted([t for t in eligible_teams if t["corners_for_pm"] is not None],
                               key=lambda x: x["corners_for_pm"], reverse=True)
            by_cards = sorted([t for t in eligible_teams if t["cards_pm"] is not None],
                             key=lambda x: x["cards_pm"], reverse=True)

            # Assign ranks
            for i, t in enumerate(by_attack):
                t["rank_attack"] = i + 1
            for i, t in enumerate(by_defense):
                t["rank_defense"] = i + 1
            for i, t in enumerate(by_goal_diff):
                t["rank_goal_diff"] = i + 1
            for i, t in enumerate(by_corners):
                t["rank_corners"] = i + 1
            for i, t in enumerate(by_cards):
                t["rank_cards"] = i + 1

        # Create profile objects
        for ts in team_stats:
            n = ts["n"]
            min_sample_ok = n >= MIN_SAMPLE_MATCHES
            rank_confidence = "high" if n >= HIGH_CONFIDENCE_MATCHES else "low"

            profile = LeagueTeamProfile(
                league_id=league_id,
                season=season,
                team_id=ts["team_id"],
                as_of_date=as_of_datetime,
                matches_played=n,
                min_sample_ok=min_sample_ok,
                rank_confidence=rank_confidence,
                goals_for_per_match=round(ts["goals_for_pm"], 2),
                goals_against_per_match=round(ts["goals_against_pm"], 2),
                goal_difference_per_match=round(ts["goal_diff_pm"], 2),
                clean_sheet_pct=round(ts["clean_sheet_pct"], 1),
                failed_to_score_pct=round(ts["failed_to_score_pct"], 1),
                btts_yes_pct=round(ts["btts_yes_pct"], 1),
                over_1_5_pct=round(ts["over_1_5_pct"], 1),
                over_2_5_pct=round(ts["over_2_5_pct"], 1),
                over_3_5_pct=round(ts["over_3_5_pct"], 1),
                corners_for_per_match=round(ts["corners_for_pm"], 1) if ts["corners_for_pm"] else None,
                corners_against_per_match=round(ts["corners_against_pm"], 1) if ts["corners_against_pm"] else None,
                yellow_cards_per_match=round(ts["yellow_cards_pm"], 1) if ts["yellow_cards_pm"] else None,
                red_cards_per_match=round(ts["red_cards_pm"], 2) if ts["red_cards_pm"] else None,
                rank_best_attack=ts.get("rank_attack") if min_sample_ok else None,
                rank_worst_defense=ts.get("rank_defense") if min_sample_ok else None,
                rank_goal_difference=ts.get("rank_goal_diff") if min_sample_ok else None,
                rank_most_corners=ts.get("rank_corners") if min_sample_ok else None,
                rank_most_cards=ts.get("rank_cards") if min_sample_ok else None,
                total_teams_in_league=total_teams if min_sample_ok else None,
                goals_scored_0_15_pct=round(ts["goals_0_15_pct"], 1) if ts["goals_0_15_pct"] is not None else None,
                goals_scored_76_90p_pct=round(ts["goals_76_90p_pct"], 1) if ts["goals_76_90p_pct"] is not None else None,
                goals_conceded_0_15_pct=round(ts["conceded_0_15_pct"], 1) if ts["conceded_0_15_pct"] is not None else None,
                goals_conceded_76_90p_pct=round(ts["conceded_76_90p_pct"], 1) if ts["conceded_76_90p_pct"] is not None else None,
                last_computed_at=datetime.now(timezone.utc),
            )

            # Upsert
            existing = await self.session.execute(
                select(LeagueTeamProfile)
                .where(
                    and_(
                        LeagueTeamProfile.league_id == league_id,
                        LeagueTeamProfile.season == season,
                        LeagueTeamProfile.team_id == ts["team_id"],
                        func.date(LeagueTeamProfile.as_of_date) == as_of_date,
                    )
                )
            )
            existing_profile = existing.scalar_one_or_none()

            if existing_profile:
                for key, value in profile.__dict__.items():
                    if not key.startswith("_") and key != "id":
                        setattr(existing_profile, key, value)
                profiles.append(existing_profile)
            else:
                self.session.add(profile)
                profiles.append(profile)

        await self.session.flush()

        logger.info(
            f"[AGGREGATES] Computed {len(profiles)} team profiles for league {league_id} season {season}"
        )

        return profiles

    async def get_league_context(
        self,
        league_id: int,
        season: int,
    ) -> Optional[dict]:
        """
        Get league context for derived_facts.

        Returns a dict suitable for inclusion in narrative payload.
        """
        result = await self.session.execute(
            select(LeagueSeasonBaseline)
            .where(
                and_(
                    LeagueSeasonBaseline.league_id == league_id,
                    LeagueSeasonBaseline.season == season,
                )
            )
            .order_by(LeagueSeasonBaseline.as_of_date.desc())
            .limit(1)
        )
        baseline = result.scalar_one_or_none()

        if not baseline:
            return None

        return {
            "sample_n_matches": baseline.sample_n_matches,
            "as_of_date": baseline.as_of_date.isoformat() if baseline.as_of_date else None,
            "goals_avg_per_match": baseline.goals_avg_per_match,
            "over_1_5_pct": baseline.over_1_5_pct,
            "over_2_5_pct": baseline.over_2_5_pct,
            "over_3_5_pct": baseline.over_3_5_pct,
            "btts_yes_pct": baseline.btts_yes_pct,
            "clean_sheet_pct_home": baseline.clean_sheet_pct_home,
            "clean_sheet_pct_away": baseline.clean_sheet_pct_away,
            "corners_avg_per_match": baseline.corners_avg_per_match,
            "yellow_cards_avg_per_match": baseline.yellow_cards_avg_per_match,
        }

    async def get_team_context(
        self,
        league_id: int,
        season: int,
        team_id: int,
    ) -> Optional[dict]:
        """
        Get team context for derived_facts.

        Returns a dict suitable for inclusion in narrative payload.
        """
        result = await self.session.execute(
            select(LeagueTeamProfile)
            .where(
                and_(
                    LeagueTeamProfile.league_id == league_id,
                    LeagueTeamProfile.season == season,
                    LeagueTeamProfile.team_id == team_id,
                )
            )
            .order_by(LeagueTeamProfile.as_of_date.desc())
            .limit(1)
        )
        profile = result.scalar_one_or_none()

        if not profile:
            return None

        return {
            "matches_played": profile.matches_played,
            "min_sample_ok": profile.min_sample_ok,
            "rank_confidence": profile.rank_confidence,
            "goals_for_per_match": profile.goals_for_per_match,
            "goals_against_per_match": profile.goals_against_per_match,
            "clean_sheet_pct": profile.clean_sheet_pct,
            "btts_yes_pct": profile.btts_yes_pct,
            "over_2_5_pct": profile.over_2_5_pct,
            "rank_best_attack": profile.rank_best_attack,
            "rank_worst_defense": profile.rank_worst_defense,
            "total_teams_in_league": profile.total_teams_in_league,
            "corners_for_per_match": profile.corners_for_per_match,
            "yellow_cards_per_match": profile.yellow_cards_per_match,
            "goals_scored_0_15_pct": profile.goals_scored_0_15_pct,
            "goals_scored_76_90p_pct": profile.goals_scored_76_90p_pct,
        }


# Convenience functions for use outside service class
async def compute_league_baseline(
    session: AsyncSession,
    league_id: int,
    season: int,
    as_of_date: Optional[date] = None,
) -> Optional[LeagueSeasonBaseline]:
    """Compute league baseline (convenience wrapper)."""
    service = AggregatesService(session)
    return await service.compute_league_baseline(league_id, season, as_of_date)


async def compute_team_profiles(
    session: AsyncSession,
    league_id: int,
    season: int,
    as_of_date: Optional[date] = None,
) -> list[LeagueTeamProfile]:
    """Compute team profiles (convenience wrapper)."""
    service = AggregatesService(session)
    return await service.compute_team_profiles(league_id, season, as_of_date)


async def get_league_context(
    session: AsyncSession,
    league_id: int,
    season: int,
) -> Optional[dict]:
    """Get league context (convenience wrapper)."""
    service = AggregatesService(session)
    return await service.get_league_context(league_id, season)


async def get_team_context(
    session: AsyncSession,
    league_id: int,
    season: int,
    team_id: int,
) -> Optional[dict]:
    """Get team context (convenience wrapper)."""
    service = AggregatesService(session)
    return await service.get_team_context(league_id, season, team_id)
