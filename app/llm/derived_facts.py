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


def _build_timeline(events: list, home_team: str, away_team: str) -> dict:
    """
    Build score timeline from goal events.

    Returns timeline dict with score progression and key metrics.
    """
    result = {
        "score_timeline": [],
        "lead_changes_count": 0,
        "equalizers_count": 0,
        "decisive_goal_minute": None,
        "late_events": {
            "goals_76_90p": 0,
            "reds_76_90p": 0,
            "pens_76_90p": 0,
        },
    }

    home_team_lower = (home_team or "").lower()
    away_team_lower = (away_team or "").lower()

    # Extract goal events
    goal_events = []
    for event in events or []:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).lower()
        if event_type != "goal":
            continue

        minute = _safe_int(event.get("minute"))
        if minute is None:
            continue

        team_name = event.get("team_name", "")
        team_name_lower = (team_name or "").lower()

        # Determine side
        side = None
        if team_name_lower:
            if team_name_lower in home_team_lower or home_team_lower in team_name_lower:
                side = "home"
            elif team_name_lower in away_team_lower or away_team_lower in team_name_lower:
                side = "away"

        if side:
            detail = str(event.get("detail", "")).lower()
            is_penalty = any(p in detail for p in ["penalty", "penal", "penalti"])

            goal_events.append({
                "minute": minute,
                "side": side,
                "is_penalty": is_penalty,
            })

    # Sort by minute
    goal_events.sort(key=lambda x: x["minute"])

    # Build timeline
    home_score = 0
    away_score = 0
    previous_leader = None

    for goal in goal_events:
        if goal["side"] == "home":
            home_score += 1
        else:
            away_score += 1

        result["score_timeline"].append({
            "minute": goal["minute"],
            "side": goal["side"],
            "new_score_home": home_score,
            "new_score_away": away_score,
            "is_penalty": goal["is_penalty"],
        })

        # Determine current leader
        if home_score > away_score:
            current_leader = "home"
        elif away_score > home_score:
            current_leader = "away"
        else:
            current_leader = "tie"

        # Check for lead change or equalizer
        if current_leader == "tie" and previous_leader in ("home", "away"):
            result["equalizers_count"] += 1
        elif current_leader in ("home", "away") and previous_leader is not None:
            if previous_leader == "tie":
                pass  # Breaking tie, not a lead change
            elif previous_leader != current_leader:
                result["lead_changes_count"] += 1

        # Track decisive goal (last goal that changes winner)
        if current_leader in ("home", "away"):
            result["decisive_goal_minute"] = goal["minute"]

        previous_leader = current_leader

        # Late events (76-90+)
        if goal["minute"] >= 76:
            result["late_events"]["goals_76_90p"] += 1
            if goal["is_penalty"]:
                result["late_events"]["pens_76_90p"] += 1

    # Count late red cards
    for event in events or []:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).lower()
        detail = str(event.get("detail", "")).lower()

        if event_type == "card" and ("red" in detail or "roja" in detail):
            minute = _safe_int(event.get("minute"))
            if minute and minute >= 76:
                result["late_events"]["reds_76_90p"] += 1

    return result


def _extract_yellow_cards(events: list, stats: dict, home_team: str, away_team: str) -> dict:
    """
    Extract yellow card information from events and stats.

    Returns yellow card counts and first card info.
    """
    result = {
        "yellow_cards": {
            "home": None,
            "away": None,
        },
        "first_card": {
            "exists": False,
            "minute": None,
            "side": None,
            "card_type": None,
            "player_name": None,
        },
        "early_card_flag_15": False,  # Card in first 15 minutes
        "late_card_flag_80p": False,  # Card after minute 80
    }

    # Try to get counts from stats first
    if stats:
        home_stats = stats.get("home", {}) or {}
        away_stats = stats.get("away", {}) or {}

        home_yellows = home_stats.get("yellow_cards") or home_stats.get("Yellow Cards")
        away_yellows = away_stats.get("yellow_cards") or away_stats.get("Yellow Cards")

        result["yellow_cards"]["home"] = _safe_int(home_yellows)
        result["yellow_cards"]["away"] = _safe_int(away_yellows)

    home_team_lower = (home_team or "").lower()
    away_team_lower = (away_team or "").lower()

    # Find all card events (yellow and red)
    card_events = []
    for event in events or []:
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type", "")).lower()
        if event_type != "card":
            continue

        minute = _safe_int(event.get("minute"))
        if minute is None:
            continue

        detail = str(event.get("detail", "")).lower()
        team_name = event.get("team_name", "")
        player_name = event.get("player_name", "")

        # Determine card type
        if "red" in detail or "roja" in detail:
            card_type = "red"
        elif "yellow" in detail or "amarilla" in detail:
            card_type = "yellow"
        else:
            continue

        # Determine side
        team_name_lower = (team_name or "").lower()
        side = None
        if team_name_lower:
            if team_name_lower in home_team_lower or home_team_lower in team_name_lower:
                side = "home"
            elif team_name_lower in away_team_lower or away_team_lower in team_name_lower:
                side = "away"

        card_events.append({
            "minute": minute,
            "side": side,
            "card_type": card_type,
            "player_name": player_name,
        })

    # Sort by minute to find first
    if card_events:
        card_events.sort(key=lambda x: x["minute"])
        first = card_events[0]

        result["first_card"] = {
            "exists": True,
            "minute": first["minute"],
            "side": first["side"],
            "card_type": first["card_type"],
            "player_name": first["player_name"],
        }

        # Check flags
        if first["minute"] <= 15:
            result["early_card_flag_15"] = True

        # Check for late card
        for card in card_events:
            if card["minute"] >= 80:
                result["late_card_flag_80p"] = True
                break

    return result


def _extract_efficiency(stats: dict, home_goals: int, away_goals: int) -> dict:
    """
    Calculate shooting efficiency ratios.

    Returns efficiency metrics (null if insufficient data).
    """
    result = {
        "shots_to_goal_ratio": {"home": None, "away": None},
        "sog_to_goal_ratio": {"home": None, "away": None},
    }

    if not stats:
        return result

    home_stats = stats.get("home", {}) or {}
    away_stats = stats.get("away", {}) or {}

    # Total shots to goals ratio (goals / shots, higher = more efficient)
    home_shots = _safe_int(home_stats.get("total_shots") or home_stats.get("Total Shots"))
    away_shots = _safe_int(away_stats.get("total_shots") or away_stats.get("Total Shots"))

    if home_shots and home_shots > 0:
        result["shots_to_goal_ratio"]["home"] = round(home_goals / home_shots, 3) if home_goals else 0.0
    if away_shots and away_shots > 0:
        result["shots_to_goal_ratio"]["away"] = round(away_goals / away_shots, 3) if away_goals else 0.0

    # Shots on goal to goals ratio
    home_sog = _safe_int(home_stats.get("shots_on_goal") or home_stats.get("Shots on Goal"))
    away_sog = _safe_int(away_stats.get("shots_on_goal") or away_stats.get("Shots on Goal"))

    if home_sog and home_sog > 0:
        result["sog_to_goal_ratio"]["home"] = round(home_goals / home_sog, 3) if home_goals else 0.0
    if away_sog and away_sog > 0:
        result["sog_to_goal_ratio"]["away"] = round(away_goals / away_sog, 3) if away_goals else 0.0

    return result


def _build_market_context(
    market_odds: dict,
    model_probs: dict,
) -> dict:
    """
    Build market context from odds and model probabilities.

    Args:
        market_odds: {home, draw, away} decimal odds
        model_probs: {home, draw, away} model probabilities

    Returns market context dict.
    """
    result = {
        "market_favorite_side": None,
        "market_implied_probs_normalized": {"home": None, "draw": None, "away": None},
        "prediction_vs_market_gap": {"home": None, "draw": None, "away": None},
    }

    if not market_odds:
        return result

    home_odds = market_odds.get("home")
    draw_odds = market_odds.get("draw")
    away_odds = market_odds.get("away")

    if not all([home_odds, draw_odds, away_odds]):
        return result

    try:
        # Calculate implied probabilities
        home_implied = 1 / float(home_odds)
        draw_implied = 1 / float(draw_odds)
        away_implied = 1 / float(away_odds)

        # Normalize (remove overround)
        total = home_implied + draw_implied + away_implied
        home_norm = round(home_implied / total, 4)
        draw_norm = round(draw_implied / total, 4)
        away_norm = round(away_implied / total, 4)

        result["market_implied_probs_normalized"] = {
            "home": home_norm,
            "draw": draw_norm,
            "away": away_norm,
        }

        # Determine market favorite (lowest odds = highest implied prob)
        if home_odds < draw_odds and home_odds < away_odds:
            result["market_favorite_side"] = "home"
        elif away_odds < draw_odds and away_odds < home_odds:
            result["market_favorite_side"] = "away"
        elif draw_odds < home_odds and draw_odds < away_odds:
            result["market_favorite_side"] = "draw"

        # Calculate gap vs model
        if model_probs:
            model_home = model_probs.get("home")
            model_draw = model_probs.get("draw")
            model_away = model_probs.get("away")

            if model_home is not None:
                result["prediction_vs_market_gap"]["home"] = round(model_home - home_norm, 4)
            if model_draw is not None:
                result["prediction_vs_market_gap"]["draw"] = round(model_draw - draw_norm, 4)
            if model_away is not None:
                result["prediction_vs_market_gap"]["away"] = round(model_away - away_norm, 4)

    except (ValueError, TypeError, ZeroDivisionError):
        pass

    return result


def _build_betting_context(
    prediction_correct: bool,
    value_bet: dict,
    actual_result: str,
) -> dict:
    """
    Build betting context with conflict detection.

    Args:
        prediction_correct: Whether main prediction was correct
        value_bet: Value bet dict (outcome, is_value_bet, etc) or None
        actual_result: Actual match result (home/draw/away)

    Returns betting context dict.
    """
    result = {
        "prediction_correct": prediction_correct,
        "value_bet_present": False,
        "value_bet_outcome": None,
        "value_bet_result": None,
        "conflict_flag": None,
    }

    if not value_bet or not value_bet.get("is_value_bet"):
        return result

    result["value_bet_present"] = True
    result["value_bet_outcome"] = value_bet.get("outcome")

    # Determine if value bet won
    vb_outcome = value_bet.get("outcome")
    if vb_outcome and actual_result:
        if vb_outcome.lower() == actual_result.lower():
            result["value_bet_result"] = "WON"
        else:
            result["value_bet_result"] = "LOST"

    # Detect conflicts
    if prediction_correct and result["value_bet_result"] == "LOST":
        result["conflict_flag"] = "pred_ok_value_lost"
    elif not prediction_correct and result["value_bet_result"] == "WON":
        result["conflict_flag"] = "pred_fail_value_won"
    elif prediction_correct and result["value_bet_result"] == "WON":
        result["conflict_flag"] = "aligned"

    return result


def _build_relative_context(
    league_context: dict,
    home_team_context: dict,
    away_team_context: dict,
) -> dict:
    """
    Build relative context comparing teams to league baselines.

    Only includes comparisons where data is available.
    Uses factual language (above/below average, top X).
    """
    result = {
        "home_vs_league": {},
        "away_vs_league": {},
        "matchup_highlights": [],
    }

    if not league_context:
        return result

    league_over_2_5 = league_context.get("over_2_5_pct")
    league_goals_avg = league_context.get("goals_avg_per_match")

    # Home team vs league
    if home_team_context:
        home_over_2_5 = home_team_context.get("over_2_5_pct")
        if home_over_2_5 is not None and league_over_2_5 is not None:
            delta = home_over_2_5 - league_over_2_5
            result["home_vs_league"]["over_2_5_delta"] = round(delta, 1)
            result["home_vs_league"]["over_2_5_above_avg"] = delta > 0

        # Attack rank context
        rank_attack = home_team_context.get("rank_best_attack")
        total_teams = home_team_context.get("total_teams_in_league")
        if rank_attack and total_teams:
            result["home_vs_league"]["attack_rank"] = rank_attack
            result["home_vs_league"]["attack_top_third"] = rank_attack <= (total_teams / 3)

    # Away team vs league
    if away_team_context:
        away_over_2_5 = away_team_context.get("over_2_5_pct")
        if away_over_2_5 is not None and league_over_2_5 is not None:
            delta = away_over_2_5 - league_over_2_5
            result["away_vs_league"]["over_2_5_delta"] = round(delta, 1)
            result["away_vs_league"]["over_2_5_above_avg"] = delta > 0

        # Attack rank context
        rank_attack = away_team_context.get("rank_best_attack")
        total_teams = away_team_context.get("total_teams_in_league")
        if rank_attack and total_teams:
            result["away_vs_league"]["attack_rank"] = rank_attack
            result["away_vs_league"]["attack_top_third"] = rank_attack <= (total_teams / 3)

    # Matchup highlights (factual observations)
    if home_team_context and away_team_context:
        # Both teams high-scoring
        home_goals_pm = home_team_context.get("goals_for_per_match", 0)
        away_goals_pm = away_team_context.get("goals_for_per_match", 0)
        if league_goals_avg and home_goals_pm > league_goals_avg and away_goals_pm > league_goals_avg:
            result["matchup_highlights"].append("both_teams_above_avg_scoring")

        # Both teams concede a lot
        home_concede = home_team_context.get("goals_against_per_match", 0)
        away_concede = away_team_context.get("goals_against_per_match", 0)
        half_avg = (league_goals_avg / 2) if league_goals_avg else 1.0
        if home_concede > half_avg and away_concede > half_avg:
            result["matchup_highlights"].append("both_teams_defensive_issues")

        # High over 2.5 matchup
        home_over_2_5 = home_team_context.get("over_2_5_pct", 0)
        away_over_2_5 = away_team_context.get("over_2_5_pct", 0)
        if home_over_2_5 > 60 and away_over_2_5 > 60:
            result["matchup_highlights"].append("high_over_2_5_matchup")

    return result


def _build_data_completeness(
    stats: dict,
    events: list,
    market_odds: dict,
) -> dict:
    """
    Assess data completeness for LLM self-limiting.

    Returns completeness flags and list of missing fields.
    """
    result = {
        "stats_present": False,
        "events_present": False,
        "odds_present": False,
        "standings_present": False,  # Placeholder for future
        "baselines_present": False,  # Placeholder for future
        "missing_fields": [],
    }

    # Check stats
    if stats:
        home_stats = stats.get("home", {}) or {}
        away_stats = stats.get("away", {}) or {}

        # Consider stats present if we have at least possession or shots
        has_possession = (
            home_stats.get("ball_possession") is not None or
            home_stats.get("Ball Possession") is not None
        )
        has_shots = (
            home_stats.get("shots_on_goal") is not None or
            home_stats.get("Shots on Goal") is not None
        )

        result["stats_present"] = has_possession or has_shots

        # Track specific missing fields
        key_fields = [
            ("ball_possession", "Ball Possession"),
            ("shots_on_goal", "Shots on Goal"),
            ("total_shots", "Total Shots"),
            ("corner_kicks", "Corner Kicks"),
        ]

        for field, alt_field in key_fields:
            home_val = home_stats.get(field) or home_stats.get(alt_field)
            away_val = away_stats.get(field) or away_stats.get(alt_field)
            if home_val is None and away_val is None:
                result["missing_fields"].append(field)

    # Check events
    result["events_present"] = bool(events and len(events) > 0)

    # Check odds
    if market_odds:
        result["odds_present"] = all([
            market_odds.get("home"),
            market_odds.get("draw"),
            market_odds.get("away"),
        ])

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
    market_odds: Optional[dict] = None,
    model_probs: Optional[dict] = None,
    value_bet: Optional[dict] = None,
    prediction_correct: Optional[bool] = None,
    league_context: Optional[dict] = None,
    home_team_context: Optional[dict] = None,
    away_team_context: Optional[dict] = None,
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
        market_odds: {home, draw, away} decimal odds (optional)
        model_probs: {home, draw, away} model probabilities (optional)
        value_bet: Value bet dict from prediction (optional)
        prediction_correct: Whether main prediction was correct (optional)

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

    # Yellow cards and first card info
    yellow_card_info = _extract_yellow_cards(events, stats, home_team, away_team)
    discipline["yellow_cards"] = yellow_card_info["yellow_cards"]
    discipline["first_card"] = yellow_card_info["first_card"]
    discipline["early_card_flag_15"] = yellow_card_info["early_card_flag_15"]
    discipline["late_card_flag_80p"] = yellow_card_info["late_card_flag_80p"]

    # Penalty facts
    penalties = _extract_penalties(events)

    # Stats leaders
    stats_leaders = _extract_stats_leaders(stats)

    # Timeline (P2: score progression, lead changes, late events)
    timeline = _build_timeline(events, home_team, away_team)

    # Efficiency (P2: shooting efficiency ratios)
    efficiency = _extract_efficiency(stats, home_goals, away_goals)

    # Market context (P2: implied probs, favorite, gap)
    market_context = _build_market_context(market_odds or {}, model_probs or {})

    # Betting context (P2: value bet result, conflict detection)
    betting_context = _build_betting_context(
        prediction_correct=prediction_correct or False,
        value_bet=value_bet or {},
        actual_result=winner,
    )

    # Data completeness (P2: for LLM self-limiting)
    data_completeness = _build_data_completeness(stats, events, market_odds or {})

    # Update data completeness with context availability
    if league_context:
        data_completeness["baselines_present"] = True
    if home_team_context and away_team_context:
        data_completeness["team_profiles_present"] = True

    # Build context_usable flag (True if both teams have min_sample_ok)
    context_usable = False
    if home_team_context and away_team_context:
        home_ok = home_team_context.get("min_sample_ok", False)
        away_ok = away_team_context.get("min_sample_ok", False)
        context_usable = home_ok and away_ok

    # Build relative context (only if usable)
    relative_context = None
    if context_usable and league_context:
        relative_context = _build_relative_context(
            league_context,
            home_team_context,
            away_team_context,
        )

    return {
        "result": result_facts,
        "discipline": discipline,
        "penalties": penalties,
        "stats_leaders": stats_leaders,
        "timeline": timeline,
        "efficiency": efficiency,
        "market_context": market_context,
        "betting_context": betting_context,
        "data_completeness": data_completeness,
        # New context blocks
        "league_context": league_context,
        "team_context": {
            "home": home_team_context,
            "away": away_team_context,
        } if home_team_context or away_team_context else None,
        "context_usable": context_usable,
        "relative_context": relative_context,
    }
