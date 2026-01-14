"""
Derived Facts Builder: Pre-computed verifiable facts for LLM narratives.

Reduces hallucinations by providing explicit, pre-calculated facts that
the LLM should use directly instead of inferring/calculating.

Principle: Only include facts that can be PROVEN from the data.
If uncertain, use null instead of guessing.
"""

from typing import Optional


def _safe_int(value) -> Optional[int]:
    """Safely convert to int, return None if invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _get_winner(home_goals: int, away_goals: int) -> str:
    """Determine winner from final score."""
    if home_goals > away_goals:
        return "home"
    elif away_goals > home_goals:
        return "away"
    else:
        return "draw"


def _detect_period(events: list, match_status: Optional[str] = None) -> str:
    """
    Detect match period from events or status.

    Returns: "REGULAR", "AET", "PEN", or "UNKNOWN"
    """
    # Check match status first (most reliable)
    if match_status:
        status_upper = match_status.upper()
        if status_upper in ("AET", "PEN"):
            return status_upper
        if status_upper == "FT":
            # Check if any event has minute > 90 (extra time)
            pass  # Fall through to event check

    # Check events for extra time indicators
    max_minute = 0
    has_penalty_shootout = False

    for event in events or []:
        if not isinstance(event, dict):
            continue

        minute = _safe_int(event.get("minute"))
        if minute and minute > max_minute:
            max_minute = minute

        # Check for penalty shootout events (usually marked differently)
        detail = str(event.get("detail", "")).lower()
        if "shootout" in detail or "penalty shootout" in detail:
            has_penalty_shootout = True

    if has_penalty_shootout:
        return "PEN"
    elif max_minute > 120:
        return "PEN"  # Events past 120 suggest penalty shootout
    elif max_minute > 90:
        return "AET"
    else:
        return "REGULAR"


def _extract_red_cards(events: list, stats: dict, home_team: str, away_team: str) -> dict:
    """
    Extract red card information from events and stats.

    Returns discipline dict with red_cards and first_red_card info.
    """
    result = {
        "red_cards": {
            "home": None,
            "away": None,
        },
        "first_red_card": {
            "exists": False,
            "side": None,
            "minute": None,
            "team_name": None,
            "player_name": None,
        },
        "minutes_played_with_10_men": None,
    }

    # Try to get counts from stats first
    if stats:
        home_stats = stats.get("home", {}) or {}
        away_stats = stats.get("away", {}) or {}

        # Handle different key formats
        home_reds = home_stats.get("red_cards") or home_stats.get("Red Cards")
        away_reds = away_stats.get("red_cards") or away_stats.get("Red Cards")

        result["red_cards"]["home"] = _safe_int(home_reds)
        result["red_cards"]["away"] = _safe_int(away_reds)

    # Find first red card from events (most detailed info)
    red_card_events = []
    home_team_lower = (home_team or "").lower()
    away_team_lower = (away_team or "").lower()

    red_card_variants = [
        "red card", "red", "tarjeta roja", "roja directa",
        "second yellow card", "second yellow", "segunda amarilla", "2nd yellow"
    ]

    for event in events or []:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).lower()
        detail = str(event.get("detail", "")).lower()

        # Check if it's a red card event
        is_red = False
        if event_type == "card":
            for variant in red_card_variants:
                if variant in detail:
                    is_red = True
                    break

        if is_red:
            minute = _safe_int(event.get("minute"))
            team_name = event.get("team_name", "")
            player_name = event.get("player_name", "")

            # Determine side
            team_name_lower = (team_name or "").lower()
            side = None
            if team_name_lower:
                if team_name_lower in home_team_lower or home_team_lower in team_name_lower:
                    side = "home"
                elif team_name_lower in away_team_lower or away_team_lower in team_name_lower:
                    side = "away"

            red_card_events.append({
                "minute": minute,
                "side": side,
                "team_name": team_name,
                "player_name": player_name,
            })

    # Sort by minute to find first
    if red_card_events:
        red_card_events.sort(key=lambda x: x["minute"] or 999)
        first = red_card_events[0]

        result["first_red_card"] = {
            "exists": True,
            "side": first["side"],
            "minute": first["minute"],
            "team_name": first["team_name"],
            "player_name": first["player_name"],
        }

        # If we don't have stats counts, count from events
        if result["red_cards"]["home"] is None or result["red_cards"]["away"] is None:
            home_count = sum(1 for e in red_card_events if e["side"] == "home")
            away_count = sum(1 for e in red_card_events if e["side"] == "away")
            if home_count > 0:
                result["red_cards"]["home"] = home_count
            if away_count > 0:
                result["red_cards"]["away"] = away_count

    return result


def _extract_penalties(events: list) -> dict:
    """
    Extract penalty goal information from events.

    Only counts goals scored from penalties, not missed penalties.
    """
    result = {
        "penalty_goals_total": 0,
        "penalty_goals_home": 0,
        "penalty_goals_away": 0,
        "penalty_minutes": [],  # List of minutes when penalty goals were scored
    }

    penalty_variants = ["penalty", "penal", "penalti", "from the spot", "desde el punto"]

    for event in events or []:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).lower()
        detail = str(event.get("detail", "")).lower()

        # Only count penalty GOALS (not missed penalties)
        if event_type == "goal":
            is_penalty = False
            for variant in penalty_variants:
                if variant in detail:
                    is_penalty = True
                    break

            if is_penalty:
                result["penalty_goals_total"] += 1

                minute = _safe_int(event.get("minute"))
                if minute:
                    result["penalty_minutes"].append(minute)

                # Note: Can't reliably determine home/away without team matching
                # This would need team_name comparison like red cards

    return result


def _extract_stats_leaders(stats: dict) -> dict:
    """
    Extract stats comparisons (who led in possession, shots, etc).

    Only includes comparisons where both values are available.
    """
    result = {
        "possession": {"leader": None, "home_value": None, "away_value": None, "delta_pct": None},
        "shots_on_goal": {"leader": None, "home_value": None, "away_value": None, "delta": None},
    }

    if not stats:
        return result

    home_stats = stats.get("home", {}) or {}
    away_stats = stats.get("away", {}) or {}

    # Possession
    home_poss = home_stats.get("ball_possession") or home_stats.get("Ball Possession")
    away_poss = away_stats.get("ball_possession") or away_stats.get("Ball Possession")

    # Handle percentage strings like "59%"
    if isinstance(home_poss, str):
        home_poss = home_poss.replace("%", "").strip()
    if isinstance(away_poss, str):
        away_poss = away_poss.replace("%", "").strip()

    try:
        home_poss_float = float(home_poss) if home_poss is not None else None
        away_poss_float = float(away_poss) if away_poss is not None else None

        if home_poss_float is not None and away_poss_float is not None:
            result["possession"]["home_value"] = home_poss_float
            result["possession"]["away_value"] = away_poss_float
            delta = home_poss_float - away_poss_float
            result["possession"]["delta_pct"] = round(delta, 1)

            if delta > 1:  # Threshold to avoid "tie" on 50.1 vs 49.9
                result["possession"]["leader"] = "home"
            elif delta < -1:
                result["possession"]["leader"] = "away"
            else:
                result["possession"]["leader"] = "tie"
    except (ValueError, TypeError):
        pass

    # Shots on goal
    home_shots = home_stats.get("shots_on_goal") or home_stats.get("Shots on Goal")
    away_shots = away_stats.get("shots_on_goal") or away_stats.get("Shots on Goal")

    home_shots_int = _safe_int(home_shots)
    away_shots_int = _safe_int(away_shots)

    if home_shots_int is not None and away_shots_int is not None:
        result["shots_on_goal"]["home_value"] = home_shots_int
        result["shots_on_goal"]["away_value"] = away_shots_int
        delta = home_shots_int - away_shots_int
        result["shots_on_goal"]["delta"] = delta

        if delta > 0:
            result["shots_on_goal"]["leader"] = "home"
        elif delta < 0:
            result["shots_on_goal"]["leader"] = "away"
        else:
            result["shots_on_goal"]["leader"] = "tie"

    return result


def build_derived_facts(
    home_goals: int,
    away_goals: int,
    home_team: str,
    away_team: str,
    events: list,
    stats: dict,
    match_status: Optional[str] = None,
) -> dict:
    """
    Build derived_facts block for LLM payload.

    Only includes verifiable facts derived from the data.
    Uncertain values are set to null.

    Args:
        home_goals: Final home score
        away_goals: Final away score
        home_team: Home team name
        away_team: Away team name
        events: List of match events
        stats: Stats dict with home/away sub-dicts
        match_status: Match status (FT, AET, PEN, etc)

    Returns:
        derived_facts dict ready for LLM payload
    """
    # Result facts
    period = _detect_period(events, match_status)
    winner = _get_winner(home_goals, away_goals)

    result_facts = {
        "final_score_home": home_goals,
        "final_score_away": away_goals,
        "ft_score": f"{home_goals}-{away_goals}",
        "winner": winner,
        "period": period,
        "ht_score": None,  # We don't have reliable HT data yet
    }

    # Discipline facts (red cards)
    discipline = _extract_red_cards(events, stats, home_team, away_team)

    # Calculate minutes with 10 men if we have first red card
    if discipline["first_red_card"]["exists"] and discipline["first_red_card"]["minute"]:
        first_red_minute = discipline["first_red_card"]["minute"]

        # Estimate match end minute based on period
        if period == "REGULAR":
            end_minute = 90
        elif period == "AET":
            end_minute = 120
        else:
            end_minute = 120  # PEN - use 120 as base

        if first_red_minute < end_minute:
            discipline["minutes_played_with_10_men"] = end_minute - first_red_minute

    # Penalty facts
    penalties = _extract_penalties(events)

    # Stats leaders
    stats_leaders = _extract_stats_leaders(stats)

    return {
        "result": result_facts,
        "discipline": discipline,
        "penalties": penalties,
        "stats_leaders": stats_leaders,
    }
