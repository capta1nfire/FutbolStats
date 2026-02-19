"""Shared feature enrichment functions for Auto-Lab and Feature Lab.

Pure functions that compute derived features from a base DataFrame
produced by FeatureEngineer.build_training_dataset(). Each function
takes a DataFrame and returns it with new columns added. All are
PIT-safe (use pre-match data only, update after each row).

Extracted from scripts/feature_lab.py to avoid duplication.
"""

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (match feature_lab.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

ELO_INITIAL = 1500
ELO_K = 32
ELO_HOME_ADV = 100


# ═══════════════════════════════════════════════════════════════════════════
# ELO (standard + goal-weighted)
# ═══════════════════════════════════════════════════════════════════════════

def compute_elo_goals(df):
    """Compute Elo ratings based on actual goals.

    PIT-safe: each match uses Elo BEFORE the match.
    Produces: elo_home, elo_away, elo_diff
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}

    elo_home_col = []
    elo_away_col = []

    for _, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]

        r_h = ratings.get(h_id, ELO_INITIAL)
        r_a = ratings.get(a_id, ELO_INITIAL)

        elo_home_col.append(r_h)
        elo_away_col.append(r_a)

        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + ELO_HOME_ADV)) / 400.0))
        exp_a = 1.0 - exp_h

        hg, ag = row["home_goals"], row["away_goals"]
        if hg > ag:
            s_h, s_a = 1.0, 0.0
        elif hg == ag:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        ratings[h_id] = r_h + ELO_K * (s_h - exp_h)
        ratings[a_id] = r_a + ELO_K * (s_a - exp_a)

    df["elo_home"] = elo_home_col
    df["elo_away"] = elo_away_col
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    return df


def compute_elo_goal_weighted(df):
    """Elo where K scales by goal margin.

    Produces: elo_gw_home, elo_gw_away, elo_gw_diff
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}
    cols_h, cols_a = [], []

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        r_h = ratings.get(h_id, ELO_INITIAL)
        r_a = ratings.get(a_id, ELO_INITIAL)
        cols_h.append(r_h)
        cols_a.append(r_a)

        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + ELO_HOME_ADV)) / 400.0))
        exp_a = 1.0 - exp_h

        hg, ag = row["home_goals"], row["away_goals"]
        gdiff = abs(hg - ag)
        k_eff = ELO_K * np.log1p(gdiff)

        if hg > ag:
            s_h, s_a = 1.0, 0.0
        elif hg == ag:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        ratings[h_id] = r_h + k_eff * (s_h - exp_h)
        ratings[a_id] = r_a + k_eff * (s_a - exp_a)

    df["elo_gw_home"] = cols_h
    df["elo_gw_away"] = cols_a
    df["elo_gw_diff"] = df["elo_gw_home"] - df["elo_gw_away"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FORM
# ═══════════════════════════════════════════════════════════════════════════

def compute_form_features(df):
    """Compute form-based features per team.

    Produces: home_win_rate5, away_win_rate5, form_diff (+ extras)
    """
    df = df.sort_values("date").reset_index(drop=True)
    team_history = {}

    form_cols = {
        "home_win_rate5": [], "away_win_rate5": [],
        "home_draw_rate5": [], "away_draw_rate5": [],
    }

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        for team_id, prefix, gs, gc in [
            (h_id, "home", hg, ag),
            (a_id, "away", ag, hg),
        ]:
            hist = team_history.get(team_id, [])
            last5 = hist[-5:] if hist else []

            if len(last5) >= 2:
                wins = sum(1 for h in last5 if h["result"] == "W")
                draws = sum(1 for h in last5 if h["result"] == "D")
                n = len(last5)
                form_cols[f"{prefix}_win_rate5"].append(wins / n)
                form_cols[f"{prefix}_draw_rate5"].append(draws / n)
            else:
                form_cols[f"{prefix}_win_rate5"].append(0.33)
                form_cols[f"{prefix}_draw_rate5"].append(0.33)

            res = "W" if gs > gc else ("D" if gs == gc else "L")
            team_history.setdefault(team_id, []).append({
                "result": res, "gs": gs, "gc": gc
            })

    for col_name, values in form_cols.items():
        df[col_name] = values

    df["form_diff"] = df["home_win_rate5"] - df["away_win_rate5"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# OPPONENT-ADJUSTED RATINGS
# ═══════════════════════════════════════════════════════════════════════════

def compute_opponent_adjusted_ratings(df):
    """Opponent-adjusted attack/defense ratings.

    Produces: opp_att_home, opp_def_home, opp_att_away, opp_def_away, opp_rating_diff
    """
    df = df.sort_values("date").reset_index(drop=True)

    att_ratings = {}
    def_ratings = {}
    ALPHA = 0.15
    INIT_ATT = 1.0
    INIT_DEF = 1.0

    cols = {k: [] for k in ["opp_att_home", "opp_def_home",
                             "opp_att_away", "opp_def_away"]}

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        att_h = att_ratings.get(h_id, INIT_ATT)
        def_h = def_ratings.get(h_id, INIT_DEF)
        att_a = att_ratings.get(a_id, INIT_ATT)
        def_a = def_ratings.get(a_id, INIT_DEF)

        cols["opp_att_home"].append(att_h)
        cols["opp_def_home"].append(def_h)
        cols["opp_att_away"].append(att_a)
        cols["opp_def_away"].append(def_a)

        opp_def_a = max(def_a, 0.3)
        opp_att_a = max(att_a, 0.3)
        opp_def_h = max(def_h, 0.3)
        opp_att_h = max(att_h, 0.3)

        adj_scored_h = hg / opp_def_a
        adj_conceded_h = ag / opp_att_a
        adj_scored_a = ag / opp_def_h
        adj_conceded_a = hg / opp_att_h

        att_ratings[h_id] = att_h * (1 - ALPHA) + adj_scored_h * ALPHA
        def_ratings[h_id] = def_h * (1 - ALPHA) + adj_conceded_h * ALPHA
        att_ratings[a_id] = att_a * (1 - ALPHA) + adj_scored_a * ALPHA
        def_ratings[a_id] = def_a * (1 - ALPHA) + adj_conceded_a * ALPHA

    for col_name, values in cols.items():
        df[col_name] = values

    df["opp_rating_diff"] = (
        (df["opp_att_home"] - df["opp_def_home"]) -
        (df["opp_att_away"] - df["opp_def_away"])
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# OVERPERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════

def compute_overperformance(df):
    """Overperformance vs Elo expectations.

    Requires: elo_home, elo_away (call compute_elo_goals first).
    Produces: overperf_home, overperf_away, overperf_diff
    """
    df = df.sort_values("date").reset_index(drop=True)

    team_overperf = {}
    cols_h, cols_a = [], []

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        for team_id, col_list in [(h_id, cols_h), (a_id, cols_a)]:
            hist = team_overperf.get(team_id, [])
            last5 = hist[-5:]
            col_list.append(float(np.mean(last5)) if last5 else 0.0)

        elo_h = row.get("elo_home", ELO_INITIAL)
        elo_a = row.get("elo_away", ELO_INITIAL)
        exp_h = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + ELO_HOME_ADV)) / 400.0))

        if hg > ag:
            pts_h, pts_a = 1.0, 0.0
        elif hg == ag:
            pts_h, pts_a = 0.33, 0.33
        else:
            pts_h, pts_a = 0.0, 1.0

        team_overperf.setdefault(h_id, []).append(pts_h - exp_h)
        team_overperf.setdefault(a_id, []).append(pts_a - (1.0 - exp_h))

    df["overperf_home"] = cols_h
    df["overperf_away"] = cols_a
    df["overperf_diff"] = df["overperf_home"] - df["overperf_away"]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# ALL-IN-ONE ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════

def enrich_for_lab(df):
    """Apply all enrichment functions in correct order.

    Call this on the DataFrame from FeatureEngineer.build_training_dataset()
    before running FAST_TESTS evaluation.

    Order matters: Elo first (overperformance depends on it).
    """
    df = compute_elo_goals(df)
    df = compute_elo_goal_weighted(df)
    df = compute_form_features(df)
    df = compute_opponent_adjusted_ratings(df)
    df = compute_overperformance(df)
    return df
