#!/usr/bin/env python3
from __future__ import annotations
"""
Feature Lab — Experimental Feature Testing
============================================
Local-only script for testing different feature combinations.
Compares performance across leagues (e.g. Argentina vs Primeira Liga).

PIT (Point-in-Time) Safety:
  - Elo ratings: sequential update, each match uses pre-match Elo
  - Rolling features: computed from history[-ROLLING_WINDOW:] before current match
  - Odds: frozen at match time (closing preferred, opening fallback, never mixed
    within the same match; odds_snapshot column tracks which type each match uses)
  - xG: rolling averages from Understat/FotMob, lagged (pre-match only)
  - Temporal split: strict chronological (sorted by date + match_id tiebreaker),
    no future leakage
  - Universe system: pre-filtered DataFrames ensure identical N, split_idx, and
    split_date for all tests within the same data availability universe

Usage:
  source .env
  python scripts/feature_lab.py --extract                # Extract fresh from DB
  python scripts/feature_lab.py                          # Use cached data
  python scripts/feature_lab.py --league 128             # Single league
  python scripts/feature_lab.py --league 128 --league 94 # Multiple leagues
  python scripts/feature_lab.py --lockbox                # 70/15/15 one-shot eval
  python scripts/feature_lab.py --shap --league 128      # SHAP explainability
  python scripts/feature_lab.py --optuna --league 128    # Optuna hypertuning
"""

import json
import sys
import argparse
import warnings
from datetime import datetime
from math import exp
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Constants ────────────────────────────────────────────────

ROLLING_WINDOW = 10
TIME_DECAY_LAMBDA = 0.01
DRAW_WEIGHT = 1.0
N_SEEDS = 5
N_BOOTSTRAP = 1000
TEST_FRACTION = 0.2

# Elo constants
ELO_INITIAL = 1500
ELO_K = 32
ELO_HOME_ADV = 100  # Home team gets +100 Elo for expected score calc

# Production XGBoost hyperparams (from engine.py)
PROD_HYPERPARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.0283,
    "n_estimators": 114,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "reg_alpha": 2.8e-05,
    "reg_lambda": 0.000904,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

# ─── Residual model hyperparams (low capacity, strong regularization) ────
RESIDUAL_HYPERPARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 2,
    "learning_rate": 0.01,
    "n_estimators": 200,
    "min_child_weight": 15,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

# ─── Two-Stage hyperparams (from engine.py TwoStageEngine) ───
TWO_STAGE_PARAMS_S1 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "verbosity": 0,
}
TWO_STAGE_PARAMS_S2 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0,
}
TWO_STAGE_DRAW_WEIGHT = 1.0

# ─── Diagnostic flags (set from CLI in main()) ──────────────
_DECOMPOSE = False
_DEVIG_SENSITIVITY = False

# Feature sets
BASELINE_FEATURES = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

NO_REST_FEATURES = [f for f in BASELINE_FEATURES
                    if f not in ("home_rest_days", "away_rest_days", "rest_diff")]

ELO_FEATURES = ["elo_home", "elo_away", "elo_diff"]
ODDS_FEATURES = ["odds_home", "odds_draw", "odds_away"]
IMPLIED_DRAW_FEATURES = ["implied_draw"]  # Derived from odds, used by Two-Stage Stage 1

# xG features (rolling averages from Understat/FotMob)
XG_CORE = ["home_xg_for_avg", "away_xg_for_avg", "xg_diff"]
XG_DEFENSE = ["home_xg_against_avg", "away_xg_against_avg", "xg_defense_diff"]
XG_OVERPERF = ["home_xg_overperf", "away_xg_overperf", "xg_overperf_diff"]
XG_ALL = XG_CORE + XG_DEFENSE + XG_OVERPERF

# ─── Surgical feature groups ─────────────────────────────────

# Argentina SIGNAL features (from feature_diagnostic)
ARG_SIGNAL = ["home_matches_played", "home_goals_conceded_avg"]

# Defense-only: what each team concedes
DEFENSE_PAIR = ["home_goals_conceded_avg", "away_goals_conceded_avg"]

# Attack-only: what each team scores
ATTACK_PAIR = ["home_goals_scored_avg", "away_goals_scored_avg"]

# Strength summary: one number that captures attack-defense balance
STRENGTH_MINIMAL = ["goal_diff_avg"]

# Competitiveness group (v1.0.1 additions)
COMPETITIVENESS = ["abs_attack_diff", "abs_defense_diff", "abs_strength_gap"]

# Baseline without competitiveness (original 14 features, matches prod model v1.0.0)
BASELINE_14_FEATURES = [f for f in BASELINE_FEATURES if f not in COMPETITIVENESS]

# Core stats (goals only, no shots/corners/rest)
GOALS_CORE = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "goal_diff_avg",
]

# Clean baseline: remove all NOISE features identified globally
CLEAN_FEATURES = [f for f in BASELINE_FEATURES
                  if f not in ("home_rest_days", "away_rest_days", "rest_diff",
                               "home_shots_avg", "away_shots_avg")]

ELO_GW_FEATURES = ["elo_gw_home", "elo_gw_away", "elo_gw_diff"]
ELO_K10 = [f"elo_k10_{s}" for s in ("home", "away", "diff")]
ELO_K20 = [f"elo_k20_{s}" for s in ("home", "away", "diff")]
ELO_K50 = [f"elo_k50_{s}" for s in ("home", "away", "diff")]
ELO_K64 = [f"elo_k64_{s}" for s in ("home", "away", "diff")]
ELO_SPLIT = ["elo_honly_home", "elo_aonly_away", "elo_split_diff"]
ELO_MOMENTUM = ["elo_momentum_home", "elo_momentum_away", "elo_momentum_diff"]
ELO_PROBS = ["elo_prob_home", "elo_prob_away", "elo_draw_proxy"]

FORM_CORE = ["home_win_rate5", "away_win_rate5", "form_diff"]
FORM_DRAW = ["home_draw_rate5", "away_draw_rate5", "draw_propensity"]
FORM_FULL = FORM_CORE + FORM_DRAW + [
    "home_unbeaten", "away_unbeaten", "unbeaten_diff",
    "home_volatility", "away_volatility",
    "home_clean_sheet5", "away_clean_sheet5",
    "home_scoring_streak", "away_scoring_streak",
]

MATCHUP_FEATURES = [
    "matchup_h_attack_v_a_defense", "matchup_a_attack_v_h_defense",
    "expected_openness", "defensive_solidity",
]

H2H_FEATURES = ["h2h_home_winrate", "h2h_n_meetings"]
SURPRISE_FEATURES = ["surprise_home", "surprise_away", "surprise_sum"]
CALENDAR_FEATURES = ["match_month", "season_phase"]

# ─── Wave 7: Opponent-Adjusted Ratings (ABE #1) ─────────────
OPP_ADJ_FEATURES = ["opp_att_home", "opp_def_home", "opp_att_away", "opp_def_away",
                     "opp_rating_diff"]

# ─── Wave 8: ABE Features ───────────────────────────────────
OVERPERF_FEATURES = ["overperf_home", "overperf_away", "overperf_diff"]
DRAW_AWARE_FEATURES = ["draw_tendency_home", "draw_tendency_away",
                       "draw_elo_interaction", "league_draw_rate"]
HOME_BIAS_FEATURES = ["home_bias_home", "home_bias_away", "home_bias_diff"]

# ─── Wave 9: Kimi Features ──────────────────────────────────
INTERACTION_FEATURES = ["elo_x_rest", "elo_x_season", "elo_x_defense",
                        "form_x_defense"]
EFFICIENCY_FEATURES = ["finish_eff_home", "finish_eff_away",
                       "def_eff_home", "def_eff_away",
                       "efficiency_diff"]

# ─── Wave 10: XI Continuity (ATI) ─────────────────────────────
XI_CONTINUITY_FEATURES = ["xi_continuity_home", "xi_continuity_away", "xi_continuity_diff"]

# ─── Wave 12: MTV (Match Talent Variance) ────────────────────
MTV_FEATURES = [
    "home_talent_delta", "away_talent_delta",
    "talent_delta_diff", "shock_magnitude",
]

# ─── Wave 11: Geographic Features (FS-09) ────────────────────
GEO_FEATURES = [
    "altitude_home_m",
    "altitude_diff_m",
    "altitude_high",         # binary: home altitude > 2000m
    "travel_distance_km",
    "travel_distance_log",   # log(1 + distance_km)
]

# ─── Wave 12: Standings Urgency Features (FS-08) ─────────────
# Derived from accumulated results (PIT-safe: only uses matches before current date)
STANDINGS_FEATURES = [
    "home_position",           # Position in table (1-N)
    "away_position",
    "position_diff",           # home_pos - away_pos (negative = home higher)
    "home_points_per_game",    # PPG in current season
    "away_points_per_game",
    "season_progress",         # matches_played / expected_total (0.0-1.0)
]

# ─── Wave 13: Precipitation Features (FS-10) ─────────────────
# Real precipitation from IDEAM stations (Colombia only, datos.gov.co)
# Source: match_weather_hist.weather_data->'ideam'
PRECIP_FEATURES = [
    "precip_total_mm",         # Total mm across 3h match window (h0+h1+h2)
    "precip_max_hour_mm",      # Max mm in any single hour (intensity proxy)
    "precip_is_rainy",         # Binary: total > 1.0mm
]

TESTS = {
    # ═══════════════════════════════════════════════════════
    # SECTION A: ANCHORS (reference points)
    # ═══════════════════════════════════════════════════════
    "A0_baseline_17":       BASELINE_FEATURES,
    "A1_only_elo_k32":      ELO_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION B: SINGLE FEATURES (what's the minimum?)
    # ═══════════════════════════════════════════════════════
    "B0_1f_goaldiff":       STRENGTH_MINIMAL,
    "B1_1f_h_defense":      ["home_goals_conceded_avg"],
    "B2_1f_h_matchplay":    ["home_matches_played"],
    "B3_1f_elo_diff":       ["elo_diff"],
    "B4_1f_elo_prob_h":     ["elo_prob_home"],
    "B5_1f_form_diff":      ["form_diff"],

    # ═══════════════════════════════════════════════════════
    # SECTION C: PAIRS (which 2 features capture the most?)
    # ═══════════════════════════════════════════════════════
    "C0_defense_pair":      DEFENSE_PAIR,
    "C1_attack_pair":       ATTACK_PAIR,
    "C2_arg_signal":        ARG_SIGNAL,
    "C3_elo_diff_form":     ["elo_diff", "form_diff"],
    "C4_elo_diff_defense":  ["elo_diff", "home_goals_conceded_avg"],
    "C5_elo_prob_draw":     ["elo_prob_home", "draw_propensity"],
    "C6_matchup_core":      ["matchup_h_attack_v_a_defense", "matchup_a_attack_v_h_defense"],

    # ═══════════════════════════════════════════════════════
    # SECTION D: ELO VARIANTS (which Elo is best?)
    # ═══════════════════════════════════════════════════════
    "D0_elo_gw":            ELO_GW_FEATURES,
    "D1_elo_k10":           ELO_K10,
    "D2_elo_k20":           ELO_K20,
    "D3_elo_k50":           ELO_K50,
    "D4_elo_k64":           ELO_K64,
    "D5_elo_split":         ELO_SPLIT,
    "D6_elo_momentum":      ELO_MOMENTUM,
    "D7_elo_probs":         ELO_PROBS,
    "D8_elo_all":           ELO_FEATURES + ELO_GW_FEATURES + ELO_SPLIT + ELO_MOMENTUM,

    # ═══════════════════════════════════════════════════════
    # SECTION E: FORM & STREAKS (does momentum matter?)
    # ═══════════════════════════════════════════════════════
    "E0_form_core":         FORM_CORE,
    "E1_form_draw":         FORM_DRAW,
    "E2_form_full":         FORM_FULL,
    "E3_form_elo":          FORM_CORE + ELO_FEATURES,
    "E4_form_full_elo":     FORM_FULL + ELO_FEATURES,
    "E5_streaks_elo":       ["home_unbeaten", "away_unbeaten",
                             "home_scoring_streak", "away_scoring_streak"] + ELO_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION F: MATCHUPS (the actual confrontation)
    # ═══════════════════════════════════════════════════════
    "F0_matchup_only":      MATCHUP_FEATURES,
    "F1_matchup_elo":       MATCHUP_FEATURES + ELO_FEATURES,
    "F2_matchup_form_elo":  MATCHUP_FEATURES + FORM_CORE + ELO_FEATURES,
    "F3_h2h":               H2H_FEATURES,
    "F4_h2h_elo":           H2H_FEATURES + ELO_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION G: SURPRISE & META
    # ═══════════════════════════════════════════════════════
    "G0_surprise":          SURPRISE_FEATURES,
    "G1_surprise_elo":      SURPRISE_FEATURES + ELO_FEATURES,
    "G2_calendar":          CALENDAR_FEATURES,
    "G3_calendar_elo":      CALENDAR_FEATURES + ELO_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION H: BEST-OF COMBOS (mixing winners)
    # ═══════════════════════════════════════════════════════
    "H0_arg_signal_elo":    ARG_SIGNAL + ELO_FEATURES,
    "H1_defense_elo":       DEFENSE_PAIR + ELO_FEATURES,
    "H2_defense_form_elo":  DEFENSE_PAIR + FORM_CORE + ELO_FEATURES,
    "H3_defense_matchup_elo": DEFENSE_PAIR + MATCHUP_FEATURES + ELO_FEATURES,
    "H4_kitchen_sink":      (DEFENSE_PAIR + MATCHUP_FEATURES + FORM_CORE +
                             ELO_FEATURES + ELO_PROBS + H2H_FEATURES),
    "H5_elo_gw_defense":    DEFENSE_PAIR + ELO_GW_FEATURES,
    "H6_elo_gw_form":       FORM_CORE + ELO_GW_FEATURES,
    "H7_elo_split_defense": DEFENSE_PAIR + ELO_SPLIT,
    "H8_surprise_form_elo": SURPRISE_FEATURES + FORM_CORE + ELO_FEATURES,
    "H9_minimal_power":     ["elo_diff", "home_goals_conceded_avg", "form_diff",
                             "matchup_h_attack_v_a_defense"],

    # ═══════════════════════════════════════════════════════
    # SECTION I: CLEAN BASELINES (noise removal)
    # ═══════════════════════════════════════════════════════
    "I0_clean_no_noise":    CLEAN_FEATURES,
    "I1_clean_elo":         CLEAN_FEATURES + ELO_FEATURES,
    "I2_goals_core":        GOALS_CORE,
    "I3_goals_core_elo":    GOALS_CORE + ELO_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION J: ODDS (if available)
    # ═══════════════════════════════════════════════════════
    "J0_only_odds":         ODDS_FEATURES,
    "J1_elo_odds":          ELO_FEATURES + ODDS_FEATURES,
    "J2_full_odds":         BASELINE_FEATURES + ODDS_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION K: ABE FEATURES (opponent-adjusted, overperf, draw-aware, home bias)
    # ═══════════════════════════════════════════════════════
    "K0_opp_adj_only":      OPP_ADJ_FEATURES,
    "K1_opp_adj_elo":       OPP_ADJ_FEATURES + ELO_FEATURES,
    "K2_overperf_only":     OVERPERF_FEATURES,
    "K3_overperf_elo":      OVERPERF_FEATURES + ELO_FEATURES,
    "K4_draw_aware":        DRAW_AWARE_FEATURES,
    "K5_draw_aware_elo":    DRAW_AWARE_FEATURES + ELO_FEATURES,
    "K6_home_bias":         HOME_BIAS_FEATURES,
    "K7_home_bias_elo":     HOME_BIAS_FEATURES + ELO_FEATURES,
    "K8_all_abe":           OPP_ADJ_FEATURES + OVERPERF_FEATURES + DRAW_AWARE_FEATURES + HOME_BIAS_FEATURES,
    "K9_all_abe_elo":       OPP_ADJ_FEATURES + OVERPERF_FEATURES + DRAW_AWARE_FEATURES + HOME_BIAS_FEATURES + ELO_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION L: KIMI FEATURES (interactions, efficiency)
    # ═══════════════════════════════════════════════════════
    "L0_interactions":      INTERACTION_FEATURES,
    "L1_interactions_elo":  INTERACTION_FEATURES + ELO_FEATURES,
    "L2_efficiency":        EFFICIENCY_FEATURES,
    "L3_efficiency_elo":    EFFICIENCY_FEATURES + ELO_FEATURES,
    "L4_kimi_all_elo":      INTERACTION_FEATURES + EFFICIENCY_FEATURES + ELO_FEATURES,

    # ═══════════════════════════════════════════════════════
    # SECTION M: GRAND COMBOS (ABE + Kimi + Lab winners)
    # ═══════════════════════════════════════════════════════
    "M0_h0_opp_adj":        ARG_SIGNAL + ELO_FEATURES + OPP_ADJ_FEATURES,
    "M1_h0_overperf":       ARG_SIGNAL + ELO_FEATURES + OVERPERF_FEATURES,
    "M2_h0_interactions":   ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES,
    "M3_h0_draw_aware":     ARG_SIGNAL + ELO_FEATURES + DRAW_AWARE_FEATURES,
    "M4_smart_minimal":     DEFENSE_PAIR + ELO_FEATURES + OPP_ADJ_FEATURES + OVERPERF_FEATURES,
    "M5_defense_elo_abe":   DEFENSE_PAIR + ELO_FEATURES + OPP_ADJ_FEATURES + OVERPERF_FEATURES + HOME_BIAS_FEATURES,
    "M6_defense_elo_kimi":  DEFENSE_PAIR + ELO_FEATURES + INTERACTION_FEATURES + EFFICIENCY_FEATURES,
    "M7_ultimate":          (DEFENSE_PAIR + ELO_FEATURES + OPP_ADJ_FEATURES +
                             OVERPERF_FEATURES + DRAW_AWARE_FEATURES +
                             INTERACTION_FEATURES + EFFICIENCY_FEATURES),
    "M8_power_5":           ["elo_diff", "opp_rating_diff", "overperf_diff",
                             "home_goals_conceded_avg", "draw_elo_interaction"],
    "M9_power_7":           ["elo_diff", "opp_rating_diff", "overperf_diff",
                             "home_goals_conceded_avg", "draw_elo_interaction",
                             "finish_eff_home", "elo_x_defense"],

    # ═══════════════════════════════════════════════════════════
    # SECTION N: ODDS COMBOS (winners + market signal)
    # ═══════════════════════════════════════════════════════════
    "N0_odds_elo":          ELO_FEATURES + ODDS_FEATURES,
    "N1_odds_defense_elo":  DEFENSE_PAIR + ELO_FEATURES + ODDS_FEATURES,
    "N2_odds_m2_combo":     ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES + ODDS_FEATURES,
    "N3_odds_efficiency":   EFFICIENCY_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "N4_odds_abe_best":     HOME_BIAS_FEATURES + OPP_ADJ_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "N5_odds_kimi_all":     INTERACTION_FEATURES + EFFICIENCY_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "N6_odds_clean":        CLEAN_FEATURES + ODDS_FEATURES,
    "N7_odds_power7":       ["elo_diff", "opp_rating_diff", "overperf_diff",
                             "home_goals_conceded_avg", "draw_elo_interaction",
                             "finish_eff_home", "elo_x_defense"] + ODDS_FEATURES,
    "N8_odds_minimal":      ["elo_diff", "home_goals_conceded_avg"] + ODDS_FEATURES,
    "N9_odds_ultimate":     (DEFENSE_PAIR + ELO_FEATURES + OPP_ADJ_FEATURES +
                             INTERACTION_FEATURES + EFFICIENCY_FEATURES + ODDS_FEATURES),

    # ═══════════════════════════════════════════════════════════
    # SECTION P: xG FEATURES (Understat EUR / FotMob LATAM)
    # ═══════════════════════════════════════════════════════════
    "P0_xg_core":           XG_CORE,
    "P1_xg_all":            XG_ALL,
    "P2_xg_elo":            XG_CORE + ELO_FEATURES,
    "P3_xg_defense_elo":    XG_CORE + XG_DEFENSE + ELO_FEATURES,
    "P4_xg_overperf_elo":   XG_ALL + ELO_FEATURES,
    "P5_xg_odds":           XG_CORE + ODDS_FEATURES,
    "P6_xg_elo_odds":       XG_CORE + ELO_FEATURES + ODDS_FEATURES,
    "P7_xg_all_elo_odds":   XG_ALL + ELO_FEATURES + ODDS_FEATURES,
    "P8_xg_defense_odds":   DEFENSE_PAIR + XG_CORE + XG_DEFENSE + ODDS_FEATURES,
    "P9_xg_ultimate":       (XG_ALL + ELO_FEATURES + DEFENSE_PAIR +
                             ODDS_FEATURES + OPP_ADJ_FEATURES),

    # ═══════════════════════════════════════════════════════════
    # SECTION Q: XI CONTINUITY (ATI — lineup stability signal)
    # ═══════════════════════════════════════════════════════════
    "Q0_xi_only":           XI_CONTINUITY_FEATURES,
    "Q1_xi_elo":            XI_CONTINUITY_FEATURES + ELO_FEATURES,
    "Q2_xi_elo_form":       XI_CONTINUITY_FEATURES + ELO_FEATURES + FORM_CORE,
    "Q3_xi_defense_elo":    XI_CONTINUITY_FEATURES + DEFENSE_PAIR + ELO_FEATURES,
    "Q4_xi_odds":           XI_CONTINUITY_FEATURES + ODDS_FEATURES,
    "Q5_xi_elo_odds":       XI_CONTINUITY_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "Q6_xi_full":           XI_CONTINUITY_FEATURES + DEFENSE_PAIR + ELO_FEATURES + FORM_CORE + ODDS_FEATURES,
    "Q7_xi_xg_elo_odds":    XI_CONTINUITY_FEATURES + XG_CORE + ELO_FEATURES + ODDS_FEATURES,
    # Q8: incremental test — does xi improve M2 (best base test)?
    "Q8_m2_plus_xi":        ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES + XI_CONTINUITY_FEATURES,

    # ═══════════════════════════════════════════════════════════
    # SECTION U: GEOGRAPHIC FEATURES (FS-09 — travel distance, altitude)
    # XGBoost handles NaN natively for teams without geo data
    # ═══════════════════════════════════════════════════════════
    "U0_geo_only":          GEO_FEATURES,
    "U1_geo_elo":           GEO_FEATURES + ELO_FEATURES,
    "U2_geo_defense_elo":   GEO_FEATURES + DEFENSE_PAIR + ELO_FEATURES,
    "U3_geo_odds":          GEO_FEATURES + ODDS_FEATURES,
    "U4_geo_elo_odds":      GEO_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "U5_geo_full":          GEO_FEATURES + DEFENSE_PAIR + ELO_FEATURES + ODDS_FEATURES,

    # ═══════════════════════════════════════════════════════════
    # SECTION T: STANDINGS URGENCY (FS-08 — derived from results)
    # PIT-safe: only uses matches played before each match date
    # ═══════════════════════════════════════════════════════════
    "T0_standings_only":    STANDINGS_FEATURES,
    "T1_standings_elo":     STANDINGS_FEATURES + ELO_FEATURES,
    "T2_standings_odds":    STANDINGS_FEATURES + ODDS_FEATURES,
    "T3_standings_elo_odds": STANDINGS_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "T4_standings_full":    STANDINGS_FEATURES + DEFENSE_PAIR + ELO_FEATURES + ODDS_FEATURES,

    # ═══════════════════════════════════════════════════════════
    # SECTION V: COMBINED (geo + standings + core)
    # ═══════════════════════════════════════════════════════════
    "V0_geo_standings":     GEO_FEATURES + STANDINGS_FEATURES,
    "V1_geo_standings_elo": GEO_FEATURES + STANDINGS_FEATURES + ELO_FEATURES,
    "V2_geo_standings_full": GEO_FEATURES + STANDINGS_FEATURES + DEFENSE_PAIR + ELO_FEATURES + ODDS_FEATURES,

    # ═══════════════════════════════════════════════════════════
    # SECTION X: PRECIPITATION (FS-10 — IDEAM real station data)
    # Colombia-only: uses match_weather_hist.weather_data->'ideam'
    # Tests whether match-day precipitation adds predictive signal
    # ═══════════════════════════════════════════════════════════
    "X0_precip_only":       PRECIP_FEATURES,
    "X1_precip_elo":        PRECIP_FEATURES + ELO_FEATURES,
    "X2_precip_odds":       PRECIP_FEATURES + ODDS_FEATURES,
    "X3_precip_elo_odds":   PRECIP_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "X4_precip_full":       PRECIP_FEATURES + DEFENSE_PAIR + ELO_FEATURES + ODDS_FEATURES,

    # ═══════════════════════════════════════════════════════════
    # SECTION S: MTV (Match Talent Variance — historical talent_delta)
    # talent_delta = PTS(XI_real) - PTS(XI_expected); shock_magnitude = max(|h|,|a|)
    # Requires pre-computed parquet: data/historical_mtv_features.parquet
    # ═══════════════════════════════════════════════════════════
    "S0_mtv_only":        MTV_FEATURES,
    "S1_mtv_elo":         MTV_FEATURES + ELO_FEATURES,
    "S2_mtv_elo_form":    MTV_FEATURES + ELO_FEATURES + FORM_CORE,
    "S3_mtv_defense_elo": MTV_FEATURES + DEFENSE_PAIR + ELO_FEATURES,
    "S4_mtv_odds":        MTV_FEATURES + ODDS_FEATURES,
    "S5_mtv_elo_odds":    MTV_FEATURES + ELO_FEATURES + ODDS_FEATURES,
    "S6_mtv_full":        MTV_FEATURES + DEFENSE_PAIR + ELO_FEATURES + FORM_CORE + ODDS_FEATURES,
    "S7_mtv_xi_elo":      MTV_FEATURES + XI_CONTINUITY_FEATURES + ELO_FEATURES,
    "S8_mtv_xg_elo_odds": MTV_FEATURES + XG_CORE + ELO_FEATURES + ODDS_FEATURES,
    "S9_m2_plus_mtv":     ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES + MTV_FEATURES,
}

# ═══════════════════════════════════════════════════════════════
# SECTION W: TWO-STAGE ARCHITECTURE (Shadow B architecture)
# Same feature sets as key One-Stage tests, but trained with
# Two-Stage decomposition: Stage 1 (draw vs non-draw) + Stage 2
# (home vs away). Tests with implied_draw add market draw signal
# to Stage 1 only. Evaluated by evaluate_two_stage().
# ═══════════════════════════════════════════════════════════════
TWO_STAGE_TESTS = {
    # W0-W2: Architecture ablation (same features as One-Stage anchors)
    "W0_ts_baseline":       BASELINE_FEATURES,
    "W1_ts_elo":            ELO_FEATURES,
    "W2_ts_baseline_elo":   BASELINE_FEATURES + ELO_FEATURES,

    # W3-W5: With odds (raw odds as features, no implied_draw)
    "W3_ts_odds":           ODDS_FEATURES,
    "W4_ts_elo_odds":       ELO_FEATURES + ODDS_FEATURES,
    "W5_ts_full_odds":      BASELINE_FEATURES + ODDS_FEATURES,

    # W6-W9: With implied_draw (the Shadow Stage 1 secret weapon)
    "W6_ts_implied_only":   BASELINE_FEATURES + IMPLIED_DRAW_FEATURES,
    "W7_ts_implied_elo":    BASELINE_FEATURES + ELO_FEATURES + IMPLIED_DRAW_FEATURES,
    "W8_ts_implied_odds":   BASELINE_FEATURES + ODDS_FEATURES + IMPLIED_DRAW_FEATURES,
    "W9_ts_kitchen_sink":   (BASELINE_FEATURES + ELO_FEATURES + ODDS_FEATURES +
                             IMPLIED_DRAW_FEATURES + FORM_CORE),

    # W10-W12: Best One-Stage combos in Two-Stage (head-to-head comparison)
    "W10_ts_defense_elo":    DEFENSE_PAIR + ELO_FEATURES,
    "W11_ts_m2_combo":       ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES,
    "W12_ts_m2_odds":        ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES + ODDS_FEATURES,
}

# ═══════════════════════════════════════════════════════════════
# SECTION R: MARKET RESIDUAL (correction over market prior)
# Same features as best tests, but model starts from market odds
# and learns only a small residual correction g(x).
# All R tests require odds → minimum universe = "odds"
# NOTE: odds are NOT included as features — they enter via base_margin
# ═══════════════════════════════════════════════════════════════
RESIDUAL_TESTS = {
    "R0_residual_baseline":     NO_REST_FEATURES + ELO_FEATURES,
    "R1_residual_form":         NO_REST_FEATURES + ELO_FEATURES + FORM_CORE,
    "R2_residual_interactions": ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES,
    "R3_residual_defense":      NO_REST_FEATURES + ELO_FEATURES + DEFENSE_PAIR,
    "R4_residual_full":         NO_REST_FEATURES + ELO_FEATURES + DEFENSE_PAIR + FORM_CORE + OPP_ADJ_FEATURES,
    "R5_residual_xg":           NO_REST_FEATURES + ELO_FEATURES + XG_CORE,
    "R6_residual_kitchen_sink": NO_REST_FEATURES + ELO_FEATURES + DEFENSE_PAIR + FORM_CORE + OPP_ADJ_FEATURES + XG_CORE + XG_DEFENSE,
    "R7_residual_mtv":          NO_REST_FEATURES + ELO_FEATURES + MTV_FEATURES,
    "R8_residual_mtv_full":     NO_REST_FEATURES + ELO_FEATURES + DEFENSE_PAIR + FORM_CORE + MTV_FEATURES,
}

# ─── Section O: Optuna candidates (top performers to re-tune) ────
# These are the champions/top-5 from lab runs across all leagues.
# Run with --optuna flag to tune each one with Optuna per-league.
OPTUNA_CANDIDATES = {
    # Universal top performers (appeared in top-5 across multiple leagues)
    "O0_elo_gw_defense":    DEFENSE_PAIR + ELO_GW_FEATURES,       # top in ITA, GER, ESP
    "O1_elo_gw_form":       FORM_CORE + ELO_GW_FEATURES,          # #1 La Liga
    "O2_defense_form_elo":  DEFENSE_PAIR + FORM_CORE + ELO_FEATURES,  # #1 Premier League
    "O3_elo_k20":           ELO_K20,                               # top in ITA, ESP
    "O4_defense_elo_kimi":  DEFENSE_PAIR + ELO_FEATURES + INTERACTION_FEATURES + EFFICIENCY_FEATURES,  # top ENG
    # LATAM champions
    "O5_m2_interactions":   ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES,  # #1 Argentina
    "O6_efficiency_elo":    EFFICIENCY_FEATURES + ELO_FEATURES,    # #1 Liga MX
    # ABE combos
    "O7_all_abe_elo":       OPP_ADJ_FEATURES + OVERPERF_FEATURES + DRAW_AWARE_FEATURES + HOME_BIAS_FEATURES + ELO_FEATURES,
    "O8_smart_minimal":     DEFENSE_PAIR + ELO_FEATURES + OPP_ADJ_FEATURES + OVERPERF_FEATURES,
    # Anchors (to measure if Optuna helps baseline/elo)
    "O9_baseline_17":       BASELINE_FEATURES,
    "OA_only_elo":          ELO_FEATURES,
    # xG + odds combos (promising in ITA, FRA)
    "OB_xg_odds":           XG_CORE + ODDS_FEATURES,
    "OC_xg_all_elo_odds":   XG_ALL + ELO_FEATURES + ODDS_FEATURES,
    "OD_xg_overperf_elo":   XG_ALL + ELO_FEATURES,
    "OE_xg_defense_odds":   DEFENSE_PAIR + XG_CORE + XG_DEFENSE + ODDS_FEATURES,
    "OF_abe_elo_odds":      OPP_ADJ_FEATURES + OVERPERF_FEATURES + DRAW_AWARE_FEATURES + HOME_BIAS_FEATURES + ELO_FEATURES + ODDS_FEATURES,
}

LEAGUE_NAMES = {
    # LATAM
    128: "Argentina",
    239: "Colombia",
    242: "Ecuador",
    281: "Peru",
    299: "Venezuela",
    344: "Bolivia",
    265: "Chile",
    250: "Paraguay",
    268: "Uruguay",
    71:  "Brasil",
    262: "Liga MX",
    253: "MLS",
    # EUR Top 5
    140: "La Liga",
    39:  "Premier League",
    135: "Serie A",
    78:  "Bundesliga",
    61:  "Ligue 1",
    # EUR Expansion
    94:  "Primeira Liga",
    88:  "Eredivisie",
    144: "Belgian Pro",
    203: "Süper Lig",
    40:  "Championship",
    307: "Saudi Pro",
    # Special
    0:   "CROSS-LEAGUE",
}

# Split-season leagues: primary league_id → all league_ids to extract together
SPLIT_LEAGUE_IDS: dict[int, list[int]] = {
    250: [250, 252],  # Paraguay Apertura + Clausura
    268: [268, 270],  # Uruguay Apertura + Clausura
}


# ─── Data Extraction ─────────────────────────────────────────

def extract_league_data(league_id: int, output_dir: str = "scripts/output/lab") -> pd.DataFrame:
    """Extract PIT-safe training data for a single league.

    Replicates feature_diagnostic._extract_via_sql() pattern
    but scoped to one league for faster iteration.
    For split-season leagues (Paraguay, Uruguay), extracts all sub-league_ids together.
    """
    import psycopg2
    from app.config import get_settings
    settings = get_settings()

    db_url = settings.DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    conn = psycopg2.connect(db_url)
    league_name = LEAGUE_NAMES.get(league_id, f"league_{league_id}")

    # Split-season: extract all sub-leagues together
    extract_ids = SPLIT_LEAGUE_IDS.get(league_id, [league_id])
    print(f"\n  Extracting {league_name} (ids={extract_ids})...")

    query = """
        SELECT m.id AS match_id, m.date, m.league_id,
               m.home_team_id, m.away_team_id,
               m.home_goals, m.away_goals,
               m.stats, m.match_weight,
               m.odds_home AS odds_home_close,
               m.odds_draw AS odds_draw_close,
               m.odds_away AS odds_away_close,
               m.opening_odds_home AS odds_home_open,
               m.opening_odds_draw AS odds_draw_open,
               m.opening_odds_away AS odds_away_open,
               m.opening_odds_kind,
               COALESCE(u.xg_home, f.xg_home, m.xg_home) AS xg_home_raw,
               COALESCE(u.xg_away, f.xg_away, m.xg_away) AS xg_away_raw,
               wh.lat AS home_lat_raw, wh.lon AS home_lon_raw,
               wh.stadium_altitude_m AS home_altitude_raw,
               wa.lat AS away_lat_raw, wa.lon AS away_lon_raw,
               wa.stadium_altitude_m AS away_altitude_raw
        FROM matches m
        LEFT JOIN match_understat_team u ON m.id = u.match_id
        LEFT JOIN match_fotmob_stats f ON m.id = f.match_id
        LEFT JOIN team_wikidata_enrichment wh ON m.home_team_id = wh.team_id
        LEFT JOIN team_wikidata_enrichment wa ON m.away_team_id = wa.team_id
        WHERE m.status = 'FT'
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND m.tainted = false
          AND m.league_id = ANY(%s)
        ORDER BY m.date, m.id
    """
    matches = pd.read_sql(query, conn, params=(extract_ids,))

    # ── Fix 0: odds_snapshot — consistent triplet, no per-column mixing ──
    has_close = (matches["odds_home_close"].notna() &
                 matches["odds_draw_close"].notna() &
                 matches["odds_away_close"].notna())
    has_open = (matches["odds_home_open"].notna() &
                matches["odds_draw_open"].notna() &
                matches["odds_away_open"].notna())

    matches["odds_snapshot"] = "missing"
    matches.loc[has_open & ~has_close, "odds_snapshot"] = "opening"
    matches.loc[has_close, "odds_snapshot"] = "closing"

    # Use closing triplet if available, else opening triplet, else NaN
    matches["odds_home"] = matches["odds_home_close"].where(has_close,
                           matches["odds_home_open"].where(has_open))
    matches["odds_draw"] = matches["odds_draw_close"].where(has_close,
                           matches["odds_draw_open"].where(has_open))
    matches["odds_away"] = matches["odds_away_close"].where(has_close,
                           matches["odds_away_open"].where(has_open))

    snap_counts = matches["odds_snapshot"].value_counts()
    print(f"  Odds snapshot: {dict(snap_counts)} "
          f"({snap_counts.get('closing', 0) + snap_counts.get('opening', 0)}/{len(matches)} "
          f"= {(snap_counts.get('closing', 0) + snap_counts.get('opening', 0)) / len(matches) * 100:.1f}% coverage)")

    # ── XI Continuity: load match_lineups ────────────────────
    lineup_query = """
        SELECT ml.match_id, ml.team_id, ml.starting_xi_ids
        FROM match_lineups ml
        JOIN matches m ON m.id = ml.match_id
        WHERE m.league_id = ANY(%s)
          AND m.status = 'FT'
          AND ml.starting_xi_ids IS NOT NULL
          AND array_length(ml.starting_xi_ids, 1) >= 7
        ORDER BY m.date, ml.match_id
    """
    lineups_df = pd.read_sql(lineup_query, conn, params=(extract_ids,))
    print(f"  Lineups loaded: {len(lineups_df)} rows")

    # ── Precipitation: load IDEAM data from match_weather_hist ──
    precip_query = """
        SELECT match_id, weather_data->'ideam' AS ideam_json
        FROM match_weather_hist
        WHERE match_id = ANY(%s)
          AND jsonb_exists(weather_data, 'ideam')
    """
    _precip_df = pd.read_sql(precip_query, conn, params=(matches["match_id"].tolist(),))
    print(f"  Precipitation (IDEAM) loaded: {len(_precip_df)} rows")

    conn.close()
    print(f"  Raw matches: {len(matches)}")

    if matches.empty:
        print(f"  [ERROR] No matches for league {league_id}")
        return pd.DataFrame()

    # Flatten stats JSON
    def extract_side_stats(stats, side):
        if not stats or not isinstance(stats, dict):
            return 0, 0
        s = stats.get(side, {})
        shots = s.get("total_shots", s.get("shots_on_goal", 0)) or 0
        corners = s.get("corner_kicks", 0) or 0
        return int(shots), int(corners)

    matches["home_shots"] = matches["stats"].apply(lambda s: extract_side_stats(s, "home")[0])
    matches["home_corners"] = matches["stats"].apply(lambda s: extract_side_stats(s, "home")[1])
    matches["away_shots"] = matches["stats"].apply(lambda s: extract_side_stats(s, "away")[0])
    matches["away_corners"] = matches["stats"].apply(lambda s: extract_side_stats(s, "away")[1])
    matches["match_weight"] = matches["match_weight"].fillna(1.0)

    # Rolling features per team
    print("  Computing rolling features...")
    home_rows = matches[["match_id", "date", "home_team_id", "home_goals", "away_goals",
                          "home_shots", "home_corners", "match_weight",
                          "xg_home_raw", "xg_away_raw"]].copy()
    home_rows.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                          "shots", "corners", "match_weight", "xg_for", "xg_against"]

    away_rows = matches[["match_id", "date", "away_team_id", "away_goals", "home_goals",
                          "away_shots", "away_corners", "match_weight",
                          "xg_away_raw", "xg_home_raw"]].copy()
    away_rows.columns = ["match_id", "date", "team_id", "goals_scored", "goals_conceded",
                          "shots", "corners", "match_weight", "xg_for", "xg_against"]

    team_matches = pd.concat([home_rows, away_rows]).sort_values(["team_id", "date"])

    def compute_team_rolling(group, tid):
        group = group.sort_values("date")
        results = []
        history = []

        for _, row in group.iterrows():
            if len(history) > 0:
                window = history[-ROLLING_WINDOW:]
                ref_date = row["date"]
                total_w = 0
                sum_gs, sum_gc, sum_sh, sum_co = 0.0, 0.0, 0.0, 0.0
                # xG rolling (only count matches that have xG)
                xg_total_w = 0
                sum_xg_for, sum_xg_against = 0.0, 0.0

                for h in window:
                    days = (ref_date - h["date"]).days
                    decay = exp(-TIME_DECAY_LAMBDA * days)
                    w = h["match_weight"] * decay
                    total_w += w
                    sum_gs += h["goals_scored"] * w
                    sum_gc += h["goals_conceded"] * w
                    sum_sh += h["shots"] * w
                    sum_co += h["corners"] * w
                    if h["xg_for"] is not None:
                        xg_total_w += w
                        sum_xg_for += h["xg_for"] * w
                        sum_xg_against += h["xg_against"] * w

                if total_w > 0:
                    goals_scored_avg = sum_gs / total_w
                    goals_conceded_avg = sum_gc / total_w
                    shots_avg = sum_sh / total_w
                    corners_avg = sum_co / total_w
                else:
                    goals_scored_avg, goals_conceded_avg = 1.0, 1.0
                    shots_avg, corners_avg = 10.0, 4.0

                if xg_total_w > 0:
                    xg_for_avg = sum_xg_for / xg_total_w
                    xg_against_avg = sum_xg_against / xg_total_w
                else:
                    xg_for_avg, xg_against_avg = None, None

                rest_days = (ref_date - history[-1]["date"]).days
                matches_played = len(history)
            else:
                goals_scored_avg, goals_conceded_avg = 1.0, 1.0
                shots_avg, corners_avg = 10.0, 4.0
                xg_for_avg, xg_against_avg = None, None
                rest_days, matches_played = 30, 0

            results.append({
                "match_id": row["match_id"],
                "team_id": tid,
                "goals_scored_avg": round(goals_scored_avg, 3),
                "goals_conceded_avg": round(goals_conceded_avg, 3),
                "shots_avg": round(shots_avg, 3),
                "corners_avg": round(corners_avg, 3),
                "rest_days": rest_days,
                "matches_played": matches_played,
                "xg_for_avg": round(xg_for_avg, 3) if xg_for_avg is not None else None,
                "xg_against_avg": round(xg_against_avg, 3) if xg_against_avg is not None else None,
            })

            xg_f = row["xg_for"] if pd.notna(row["xg_for"]) else None
            xg_a = row["xg_against"] if pd.notna(row["xg_against"]) else None
            history.append({
                "date": row["date"],
                "goals_scored": row["goals_scored"],
                "goals_conceded": row["goals_conceded"],
                "shots": row["shots"],
                "corners": row["corners"],
                "match_weight": row["match_weight"],
                "xg_for": xg_f,
                "xg_against": xg_a,
            })

        return pd.DataFrame(results)

    # pandas 3.0: groupby excludes key column from groups, iterate manually
    team_feature_parts = []
    for tid, group in team_matches.groupby("team_id"):
        team_feature_parts.append(compute_team_rolling(group, tid))
    team_features = pd.concat(team_feature_parts).reset_index(drop=True)

    # Merge home/away features back
    home_feats = team_features.merge(
        matches[["match_id", "home_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "home_team_id"],
    ).drop(columns=["team_id", "home_team_id"])
    home_feats = home_feats.rename(columns={
        c: f"home_{c}" for c in ["goals_scored_avg", "goals_conceded_avg",
                                   "shots_avg", "corners_avg", "rest_days", "matches_played",
                                   "xg_for_avg", "xg_against_avg"]
    })

    away_feats = team_features.merge(
        matches[["match_id", "away_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "away_team_id"],
    ).drop(columns=["team_id", "away_team_id"])
    away_feats = away_feats.rename(columns={
        c: f"away_{c}" for c in ["goals_scored_avg", "goals_conceded_avg",
                                   "shots_avg", "corners_avg", "rest_days", "matches_played",
                                   "xg_for_avg", "xg_against_avg"]
    })

    # Build final dataset
    _build_cols = ["match_id", "date", "league_id", "home_team_id", "away_team_id",
                   "home_goals", "away_goals", "odds_home", "odds_draw", "odds_away",
                   "odds_home_open", "odds_draw_open", "odds_away_open",
                   "odds_home_close", "odds_draw_close", "odds_away_close",
                   "opening_odds_kind", "odds_snapshot"]
    # Geo raw columns (Wave 11)
    for gc in ["home_lat_raw", "home_lon_raw", "home_altitude_raw",
               "away_lat_raw", "away_lon_raw", "away_altitude_raw"]:
        if gc in matches.columns:
            _build_cols.append(gc)
    df = matches[_build_cols].copy()
    df = df.merge(home_feats, on="match_id", how="left")
    df = df.merge(away_feats, on="match_id", how="left")

    # Derived features
    df["goal_diff_avg"] = df["home_goals_scored_avg"] - df["away_goals_scored_avg"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["abs_attack_diff"] = (df["home_goals_scored_avg"] - df["away_goals_scored_avg"]).abs()
    df["abs_defense_diff"] = (df["home_goals_conceded_avg"] - df["away_goals_conceded_avg"]).abs()
    home_net = df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]
    away_net = df["away_goals_scored_avg"] - df["away_goals_conceded_avg"]
    df["abs_strength_gap"] = (home_net - away_net).abs()

    # xG derived features
    df["xg_diff"] = df["home_xg_for_avg"] - df["away_xg_for_avg"]
    df["xg_defense_diff"] = df["home_xg_against_avg"] - df["away_xg_against_avg"]
    df["home_xg_overperf"] = df["home_goals_scored_avg"] - df["home_xg_for_avg"]
    df["away_xg_overperf"] = df["away_goals_scored_avg"] - df["away_xg_for_avg"]
    df["xg_overperf_diff"] = df["home_xg_overperf"] - df["away_xg_overperf"]

    # Result label (0=H, 1=D, 2=A)
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], 0,
        np.where(df["home_goals"] == df["away_goals"], 1, 2)
    )

    # Compute ALL features (Elo + variants + form + matchup + surprise + calendar)
    print("  Computing Elo-goals ratings...")
    df = compute_elo_goals(df)
    df = compute_all_experimental_features(df)

    # ── XI Continuity (Section Q) ────────────────────────────
    XI_WINDOW = 15
    print("  Computing XI continuity...")
    if not lineups_df.empty:
        # Build lookup: match_id -> {team_id -> starting_xi_ids}
        lineup_lookup: dict[int, dict[int, list[int]]] = {}
        for _, lrow in lineups_df.iterrows():
            mid = lrow["match_id"]
            tid = lrow["team_id"]
            xi = lrow["starting_xi_ids"]
            if isinstance(xi, list) and len(xi) >= 7:
                lineup_lookup.setdefault(mid, {})[tid] = xi

        # Chronological pass (PIT-safe: update history AFTER computing)
        df = df.sort_values("date").reset_index(drop=True)
        xi_home_col: list = []
        xi_away_col: list = []
        team_xi_history: dict = {}

        for _, row in df.iterrows():
            mid = row["match_id"]
            h_id = row["home_team_id"]
            a_id = row["away_team_id"]
            match_xis = lineup_lookup.get(mid, {})

            h_xi = match_xis.get(h_id)
            a_xi = match_xis.get(a_id)

            # Home xi_continuity
            h_hist = team_xi_history.get(h_id, [])
            if h_xi and len(h_hist) >= 3:
                window = h_hist[-XI_WINDOW:]
                n = len(window)
                pcounts: dict[int, int] = {}
                for past_xi in window:
                    for pid in past_xi:
                        pcounts[pid] = pcounts.get(pid, 0) + 1
                rates = [pcounts.get(pid, 0) / n for pid in h_xi]
                xi_home_col.append(round(sum(rates) / len(rates), 4))
            else:
                xi_home_col.append(None)

            # Away xi_continuity
            a_hist = team_xi_history.get(a_id, [])
            if a_xi and len(a_hist) >= 3:
                window = a_hist[-XI_WINDOW:]
                n = len(window)
                pcounts2: dict[int, int] = {}
                for past_xi in window:
                    for pid in past_xi:
                        pcounts2[pid] = pcounts2.get(pid, 0) + 1
                rates = [pcounts2.get(pid, 0) / n for pid in a_xi]
                xi_away_col.append(round(sum(rates) / len(rates), 4))
            else:
                xi_away_col.append(None)

            # Update histories AFTER computing (PIT-safe)
            if h_xi:
                team_xi_history.setdefault(h_id, []).append(h_xi)
            if a_xi:
                team_xi_history.setdefault(a_id, []).append(a_xi)

        df["xi_continuity_home"] = xi_home_col
        df["xi_continuity_away"] = xi_away_col
        df["xi_continuity_diff"] = df["xi_continuity_home"] - df["xi_continuity_away"]
    else:
        df["xi_continuity_home"] = None
        df["xi_continuity_away"] = None
        df["xi_continuity_diff"] = None

    # ── Geographic Features (Wave 11 — FS-09) ──────────────────
    print("  Computing geographic features...")
    _R = 6371.0  # Earth radius in km
    h_lat = np.radians(df["home_lat_raw"].astype(float))
    h_lon = np.radians(df["home_lon_raw"].astype(float))
    a_lat = np.radians(df["away_lat_raw"].astype(float))
    a_lon = np.radians(df["away_lon_raw"].astype(float))
    dlat = a_lat - h_lat
    dlon = a_lon - h_lon
    a_hav = np.sin(dlat / 2) ** 2 + np.cos(h_lat) * np.cos(a_lat) * np.sin(dlon / 2) ** 2
    dist_km = _R * 2 * np.arcsin(np.sqrt(a_hav))

    df["altitude_home_m"] = df["home_altitude_raw"].astype(float)
    df["altitude_diff_m"] = df["home_altitude_raw"].astype(float) - df["away_altitude_raw"].astype(float)
    df["altitude_high"] = (df["home_altitude_raw"].astype(float) > 2000).astype(float)
    df.loc[df["home_altitude_raw"].isna(), "altitude_high"] = np.nan
    df["travel_distance_km"] = dist_km
    df["travel_distance_log"] = np.log1p(dist_km)

    n_geo = df["travel_distance_km"].notna().sum()
    n_alt = df["altitude_home_m"].notna().sum()
    print(f"  Geo: distance {n_geo}/{len(df)} ({100*n_geo/len(df):.0f}%), "
          f"altitude {n_alt}/{len(df)} ({100*n_alt/len(df):.0f}%)")

    # ── Standings Features (Wave 12 — FS-08) ─────────────────
    # Derive standings from accumulated results (PIT-safe)
    print("  Computing standings features...")
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Season key: cross-year leagues (Aug-May) use year of August start
    df["_season_key"] = df["date"].apply(
        lambda d: d.year if d.month >= 7 else d.year - 1)

    # Use index-keyed dict to avoid groupby ordering issues
    st_vals = {}

    for (lid, skey), grp in df.groupby(["league_id", "_season_key"]):
        grp = grp.sort_values(["date", "match_id"])
        team_pts = {}
        team_gd = {}
        team_mp = {}
        all_teams = set()

        for idx in grp.index:
            row = df.loc[idx]
            h_id = row["home_team_id"]
            a_id = row["away_team_id"]
            all_teams.add(h_id)
            all_teams.add(a_id)

            if team_mp:
                table = sorted(all_teams,
                               key=lambda t: (team_pts.get(t, 0), team_gd.get(t, 0)),
                               reverse=True)
                pos_map = {t: i + 1 for i, t in enumerate(table)}
                n_teams = len(table)
                h_pos = pos_map.get(h_id, n_teams)
                a_pos = pos_map.get(a_id, n_teams)
                h_mp_val = team_mp.get(h_id, 0)
                a_mp_val = team_mp.get(a_id, 0)
                avg_mp = sum(team_mp.values()) / max(len(team_mp), 1)
                expected = max((n_teams - 1) * 2, 1)

                st_vals[idx] = {
                    "home_position": h_pos,
                    "away_position": a_pos,
                    "position_diff": h_pos - a_pos,
                    "home_points_per_game": round(team_pts.get(h_id, 0) / max(h_mp_val, 1), 3),
                    "away_points_per_game": round(team_pts.get(a_id, 0) / max(a_mp_val, 1), 3),
                    "season_progress": round(min(avg_mp / expected, 1.0), 3),
                }
            else:
                st_vals[idx] = {
                    "home_position": None, "away_position": None,
                    "position_diff": None, "home_points_per_game": None,
                    "away_points_per_game": None, "season_progress": 0.0,
                }

            # Update AFTER use (PIT-safe)
            hg, ag = row["home_goals"], row["away_goals"]
            if hg > ag:
                team_pts[h_id] = team_pts.get(h_id, 0) + 3
            elif hg == ag:
                team_pts[h_id] = team_pts.get(h_id, 0) + 1
                team_pts[a_id] = team_pts.get(a_id, 0) + 1
            else:
                team_pts[a_id] = team_pts.get(a_id, 0) + 3
            team_gd[h_id] = team_gd.get(h_id, 0) + (hg - ag)
            team_gd[a_id] = team_gd.get(a_id, 0) + (ag - hg)
            team_mp[h_id] = team_mp.get(h_id, 0) + 1
            team_mp[a_id] = team_mp.get(a_id, 0) + 1

    for col in STANDINGS_FEATURES:
        df[col] = df.index.map(lambda i, c=col: st_vals.get(i, {}).get(c))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert to numeric
    for col in STANDINGS_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_standings = df["home_position"].notna().sum()
    print(f"  Standings: {n_standings}/{len(df)} ({100*n_standings/len(df):.0f}%)")

    # ── Precipitation Features (Wave 13 — FS-10) ─────────────────
    # Source: match_weather_hist IDEAM data (Colombia only, datos.gov.co)
    # _precip_df was loaded before conn.close() in the extraction phase
    print("  Computing precipitation features...")
    import json as _json

    def _parse_precip(raw):
        """Extract total_mm and max_hour_mm from IDEAM JSONB."""
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None, None
        try:
            if isinstance(raw, str):
                d = _json.loads(raw)
            elif isinstance(raw, dict):
                d = raw
            else:
                return None, None
            hours = []
            for key in ("h0", "h1", "h2"):
                h = d.get(key)
                if h and "rain_mm" in h:
                    hours.append(float(h["rain_mm"]))
                else:
                    hours.append(0.0)
            total = sum(hours)
            max_h = max(hours)
            return total, max_h
        except Exception:
            return None, None

    # Parse IDEAM data
    precip_vals = {}
    for _, row in _precip_df.iterrows():
        t, m = _parse_precip(row["ideam_json"])
        if t is not None:
            precip_vals[row["match_id"]] = (t, m)

    df["precip_total_mm"] = df["match_id"].map(
        lambda mid: precip_vals.get(mid, (None, None))[0])
    df["precip_max_hour_mm"] = df["match_id"].map(
        lambda mid: precip_vals.get(mid, (None, None))[1])
    df["precip_total_mm"] = pd.to_numeric(df["precip_total_mm"], errors="coerce")
    df["precip_max_hour_mm"] = pd.to_numeric(df["precip_max_hour_mm"], errors="coerce")
    df["precip_is_rainy"] = (df["precip_total_mm"] > 1.0).astype(float)
    df.loc[df["precip_total_mm"].isna(), "precip_is_rainy"] = np.nan

    n_precip = df["precip_total_mm"].notna().sum()
    n_rainy = (df["precip_is_rainy"] == 1.0).sum() if n_precip > 0 else 0
    print(f"  Precip: {n_precip}/{len(df)} ({100*n_precip/len(df):.0f}%) "
          f"| Rainy matches: {n_rainy}/{n_precip if n_precip > 0 else 1} "
          f"({100*n_rainy/max(n_precip,1):.0f}%)")

    # Coverage report
    n_total = len(df)
    n_odds = df[ODDS_FEATURES].notna().all(axis=1).sum()
    n_elo = df["elo_home"].notna().sum()
    n_xg = df["home_xg_for_avg"].notna().sum()
    n_xi = df["xi_continuity_home"].notna().sum()
    print(f"  Final: {n_total} matches | Odds: {n_odds}/{n_total} ({100*n_odds/n_total:.0f}%) "
          f"| Elo: {n_elo}/{n_total} ({100*n_elo/n_total:.0f}%) "
          f"| xG: {n_xg}/{n_total} ({100*n_xg/n_total:.0f}%) "
          f"| XI: {n_xi}/{n_total} ({100*n_xi/n_total:.0f}%)")

    # Save
    out_path = Path(output_dir) / f"lab_data_{league_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    return df


# ─── Elo-Goals Computation ───────────────────────────────────

def compute_elo_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings based on actual goals (not xG).

    PIT-safe: each match uses Elo BEFORE the match.
    Sequential update in chronological order.
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}  # team_id -> current Elo

    elo_home_col = []
    elo_away_col = []

    for _, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]

        # Pre-match Elo (PIT-safe: use BEFORE update)
        r_h = ratings.get(h_id, ELO_INITIAL)
        r_a = ratings.get(a_id, ELO_INITIAL)

        elo_home_col.append(r_h)
        elo_away_col.append(r_a)

        # Expected scores (with home advantage)
        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + ELO_HOME_ADV)) / 400.0))
        exp_a = 1.0 - exp_h

        # Actual scores
        hg, ag = row["home_goals"], row["away_goals"]
        if hg > ag:
            s_h, s_a = 1.0, 0.0
        elif hg == ag:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        # Update ratings
        ratings[h_id] = r_h + ELO_K * (s_h - exp_h)
        ratings[a_id] = r_a + ELO_K * (s_a - exp_a)

    df["elo_home"] = elo_home_col
    df["elo_away"] = elo_away_col
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    return df


# ─── WAVE 2: Elo Variants ────────────────────────────────────

def compute_elo_goal_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """Elo where K scales by goal margin.
    Dominant wins (3-0) update more than scrappy wins (1-0).
    Formula: K_eff = K * ln(1 + goal_diff)
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
        k_eff = ELO_K * np.log1p(gdiff)  # ln(1 + |goal_diff|)

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


def compute_elo_k_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo for multiple K values to find optimal sensitivity."""
    for k_val in [10, 20, 50, 64]:
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
            hg, ag = row["home_goals"], row["away_goals"]
            s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

            ratings[h_id] = r_h + k_val * (s_h - exp_h)
            ratings[a_id] = r_a + k_val * ((1 - s_h) - (1 - exp_h))

        df[f"elo_k{k_val}_home"] = cols_h
        df[f"elo_k{k_val}_away"] = cols_a
        df[f"elo_k{k_val}_diff"] = df[f"elo_k{k_val}_home"] - df[f"elo_k{k_val}_away"]
    return df


def compute_elo_home_away_split(df: pd.DataFrame) -> pd.DataFrame:
    """Separate Elo ratings for home and away performance.
    Some teams are lions at home but kittens away.
    """
    df = df.sort_values("date").reset_index(drop=True)
    home_ratings = {}  # Elo only from home games
    away_ratings = {}  # Elo only from away games
    cols = {"elo_honly_home": [], "elo_aonly_away": []}

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        r_h = home_ratings.get(h_id, ELO_INITIAL)
        r_a = away_ratings.get(a_id, ELO_INITIAL)
        cols["elo_honly_home"].append(r_h)
        cols["elo_aonly_away"].append(r_a)

        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / 400.0))
        hg, ag = row["home_goals"], row["away_goals"]
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        home_ratings[h_id] = r_h + ELO_K * (s_h - exp_h)
        away_ratings[a_id] = r_a + ELO_K * ((1 - s_h) - (1 - exp_h))

    df["elo_honly_home"] = cols["elo_honly_home"]
    df["elo_aonly_away"] = cols["elo_aonly_away"]
    df["elo_split_diff"] = df["elo_honly_home"] - df["elo_aonly_away"]
    return df


def compute_elo_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Elo momentum: rate of change over last 5 matches.
    Positive = team is on the rise. Negative = declining.
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}       # current rating
    elo_history = {}   # team_id -> list of last N ratings

    cols_mom_h, cols_mom_a = [], []

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        r_h = ratings.get(h_id, ELO_INITIAL)
        r_a = ratings.get(a_id, ELO_INITIAL)

        # Momentum = current - avg_of_last_5 (before this match)
        hist_h = elo_history.get(h_id, [ELO_INITIAL])
        hist_a = elo_history.get(a_id, [ELO_INITIAL])
        mom_h = r_h - np.mean(hist_h[-5:])
        mom_a = r_a - np.mean(hist_a[-5:])
        cols_mom_h.append(mom_h)
        cols_mom_a.append(mom_a)

        # Update Elo
        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + ELO_HOME_ADV)) / 400.0))
        hg, ag = row["home_goals"], row["away_goals"]
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        new_r_h = r_h + ELO_K * (s_h - exp_h)
        new_r_a = r_a + ELO_K * ((1 - s_h) - (1 - exp_h))
        ratings[h_id] = new_r_h
        ratings[a_id] = new_r_a

        # Track history
        elo_history.setdefault(h_id, []).append(new_r_h)
        elo_history.setdefault(a_id, []).append(new_r_a)

    df["elo_momentum_home"] = cols_mom_h
    df["elo_momentum_away"] = cols_mom_a
    df["elo_momentum_diff"] = df["elo_momentum_home"] - df["elo_momentum_away"]
    return df


# ─── WAVE 3: Form & Streak Features ──────────────────────────

def compute_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute form-based features per team.
    - win_rate_last5: % of wins in last 5 matches
    - draw_rate_last5: % of draws (draw-prone teams)
    - unbeaten_streak: consecutive matches without losing
    - volatility: std of goals scored in last 5
    - clean_sheet_rate: % of last 5 with 0 goals conceded
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Build team-centric match history
    team_history = {}  # team_id -> list of {result, goals_scored, goals_conceded}

    form_cols = {
        "home_win_rate5": [], "away_win_rate5": [],
        "home_draw_rate5": [], "away_draw_rate5": [],
        "home_unbeaten": [], "away_unbeaten": [],
        "home_volatility": [], "away_volatility": [],
        "home_clean_sheet5": [], "away_clean_sheet5": [],
        "home_scoring_streak": [], "away_scoring_streak": [],
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

                # Volatility: std of goals scored
                goals = [h["gs"] for h in last5]
                form_cols[f"{prefix}_volatility"].append(float(np.std(goals)))

                # Clean sheet rate
                cs = sum(1 for h in last5 if h["gc"] == 0)
                form_cols[f"{prefix}_clean_sheet5"].append(cs / n)

                # Unbeaten streak (count backwards from most recent)
                streak = 0
                for h in reversed(hist):
                    if h["result"] != "L":
                        streak += 1
                    else:
                        break
                form_cols[f"{prefix}_unbeaten"].append(min(streak, 20))

                # Scoring streak (consecutive scoring)
                s_streak = 0
                for h in reversed(hist):
                    if h["gs"] > 0:
                        s_streak += 1
                    else:
                        break
                form_cols[f"{prefix}_scoring_streak"].append(min(s_streak, 20))

            else:
                # Defaults
                form_cols[f"{prefix}_win_rate5"].append(0.33)
                form_cols[f"{prefix}_draw_rate5"].append(0.33)
                form_cols[f"{prefix}_volatility"].append(1.0)
                form_cols[f"{prefix}_clean_sheet5"].append(0.2)
                form_cols[f"{prefix}_unbeaten"].append(0)
                form_cols[f"{prefix}_scoring_streak"].append(0)

            # Record this match
            res = "W" if gs > gc else ("D" if gs == gc else "L")
            team_history.setdefault(team_id, []).append({
                "result": res, "gs": gs, "gc": gc
            })

    for col_name, values in form_cols.items():
        df[col_name] = values

    # Derived form features
    df["form_diff"] = df["home_win_rate5"] - df["away_win_rate5"]
    df["draw_propensity"] = df["home_draw_rate5"] + df["away_draw_rate5"]
    df["unbeaten_diff"] = df["home_unbeaten"] - df["away_unbeaten"]

    return df


# ─── WAVE 4: Matchup Features ────────────────────────────────

def compute_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-features that capture the actual confrontation.
    Attack vs Defense matchups: the real battle.
    """
    # Confrontation: home attack vs away defense, and vice versa
    df["matchup_h_attack_v_a_defense"] = (
        df["home_goals_scored_avg"] - df["away_goals_conceded_avg"]
    )
    df["matchup_a_attack_v_h_defense"] = (
        df["away_goals_scored_avg"] - df["home_goals_conceded_avg"]
    )
    # Positive = attacker dominates defender

    # Total expected goals (proxy for how open the match will be)
    df["expected_openness"] = (
        df["home_goals_scored_avg"] + df["away_goals_scored_avg"]
    )

    # Defensive solidity: both teams concede little = tight match
    df["defensive_solidity"] = -(
        df["home_goals_conceded_avg"] + df["away_goals_conceded_avg"]
    )

    return df


def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head record between the two teams.
    Some matchups are historically lopsided regardless of form.
    """
    df = df.sort_values("date").reset_index(drop=True)
    h2h_history = {}  # (team_a, team_b) -> list of results from team_a perspective

    cols_h2h_home_wr = []
    cols_h2h_n = []

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        # H2H key: always (min, max) for consistency
        key = (min(h_id, a_id), max(h_id, a_id))
        hist = h2h_history.get(key, [])

        # Home team's win rate in H2H
        if hist:
            h_wins = sum(1 for h in hist if
                         (h["winner"] == h_id))
            cols_h2h_home_wr.append(h_wins / len(hist))
            cols_h2h_n.append(min(len(hist), 20))
        else:
            cols_h2h_home_wr.append(0.5)
            cols_h2h_n.append(0)

        # Record
        winner = h_id if hg > ag else (a_id if ag > hg else None)
        h2h_history.setdefault(key, []).append({"winner": winner})

    df["h2h_home_winrate"] = cols_h2h_home_wr
    df["h2h_n_meetings"] = cols_h2h_n
    return df


# ─── WAVE 5: Surprise & Meta Features ────────────────────────

def compute_surprise_features(df: pd.DataFrame) -> pd.DataFrame:
    """How 'surprising' are this team's recent results?
    Teams that frequently upset expectations may continue to do so.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Need elo_home/elo_away already computed
    team_surprises = {}  # team_id -> list of surprise values

    cols_surprise_h, cols_surprise_a = [], []

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        # Surprise for each team
        for team_id, prefix, cols in [
            (h_id, "home", cols_surprise_h),
            (a_id, "away", cols_surprise_a),
        ]:
            hist = team_surprises.get(team_id, [])
            last5 = hist[-5:]
            if last5:
                cols.append(float(np.mean(last5)))
            else:
                cols.append(0.0)

        # Compute surprise for this match
        elo_h = row.get("elo_home", ELO_INITIAL)
        elo_a = row.get("elo_away", ELO_INITIAL)
        exp_h = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + ELO_HOME_ADV)) / 400.0))

        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        surprise = abs(actual_h - exp_h)  # 0 = expected, ~1 = huge upset

        team_surprises.setdefault(h_id, []).append(surprise)
        team_surprises.setdefault(a_id, []).append(surprise)

    df["surprise_home"] = cols_surprise_h
    df["surprise_away"] = cols_surprise_a
    df["surprise_sum"] = df["surprise_home"] + df["surprise_away"]
    return df


def compute_elo_expected_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Elo to expected win probability.
    This is what bookmakers essentially do — useful as direct feature.
    """
    elo_h = df["elo_home"].values
    elo_a = df["elo_away"].values
    df["elo_prob_home"] = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + ELO_HOME_ADV)) / 400.0))
    df["elo_prob_away"] = 1.0 - df["elo_prob_home"]
    # Draw probability proxy: closer the probs, higher draw chance
    df["elo_draw_proxy"] = 1.0 - abs(df["elo_prob_home"] - df["elo_prob_away"])
    return df


# ─── WAVE 6: Seasonal & Calendar Features ────────────────────

def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based features. Season phase matters."""
    df["match_month"] = pd.to_datetime(df["date"]).dt.month
    # Season phase: early (1-3), mid (4-6), late (7-9), final (10-12)
    df["season_phase"] = pd.to_datetime(df["date"]).dt.month % 12 // 3
    return df


# ─── WAVE 7: Opponent-Adjusted Ratings (ABE #1) ─────────────

def compute_opponent_adjusted_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Opponent-adjusted attack/defense ratings.

    ABE's #1 pick: instead of raw goal averages, rate teams based on
    WHO they played. Scoring 2 vs River Plate is worth more than 2 vs
    a bottom team. Uses iterative EMA approach.

    For each team:
      att_power = EMA of (goals_scored / opponent_def_rating)
      def_power = EMA of (goals_conceded / opponent_att_rating)

    Higher att_power = stronger attack. Lower def_power = stronger defense.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Running ratings per team (attack, defense)
    att_ratings = {}  # team_id -> float (higher = better attack)
    def_ratings = {}  # team_id -> float (lower = better defense)
    ALPHA = 0.15  # EMA smoothing factor (balance reactivity vs stability)
    INIT_ATT = 1.0
    INIT_DEF = 1.0

    cols = {k: [] for k in ["opp_att_home", "opp_def_home",
                             "opp_att_away", "opp_def_away"]}

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        # Pre-match ratings (PIT-safe)
        att_h = att_ratings.get(h_id, INIT_ATT)
        def_h = def_ratings.get(h_id, INIT_DEF)
        att_a = att_ratings.get(a_id, INIT_ATT)
        def_a = def_ratings.get(a_id, INIT_DEF)

        cols["opp_att_home"].append(att_h)
        cols["opp_def_home"].append(def_h)
        cols["opp_att_away"].append(att_a)
        cols["opp_def_away"].append(def_a)

        # Post-match update: normalize by opponent quality
        # Home scored hg against away defense def_a
        opp_def_a = max(def_a, 0.3)  # Floor to avoid division explosion
        opp_att_a = max(att_a, 0.3)
        opp_def_h = max(def_h, 0.3)
        opp_att_h = max(att_h, 0.3)

        # "I scored 3 goals vs a defense rated 0.5 → adj_scored = 3/0.5 = 6.0 (impressive)"
        # "I scored 3 goals vs a defense rated 2.0 → adj_scored = 3/2.0 = 1.5 (expected)"
        adj_scored_h = hg / opp_def_a
        adj_conceded_h = ag / opp_att_a
        adj_scored_a = ag / opp_def_h
        adj_conceded_a = hg / opp_att_h

        # EMA update
        att_ratings[h_id] = att_h * (1 - ALPHA) + adj_scored_h * ALPHA
        def_ratings[h_id] = def_h * (1 - ALPHA) + adj_conceded_h * ALPHA
        att_ratings[a_id] = att_a * (1 - ALPHA) + adj_scored_a * ALPHA
        def_ratings[a_id] = def_a * (1 - ALPHA) + adj_conceded_a * ALPHA

    for col_name, values in cols.items():
        df[col_name] = values

    # Derived: overall rating diff (net quality difference)
    df["opp_rating_diff"] = (
        (df["opp_att_home"] - df["opp_def_home"]) -
        (df["opp_att_away"] - df["opp_def_away"])
    )
    return df


# ─── WAVE 8: ABE Features (overperf, draw-aware, home bias) ─

def compute_overperformance(df: pd.DataFrame) -> pd.DataFrame:
    """Overperformance vs Elo expectations (ABE #3).

    Better than raw win_rate because it discounts opponent strength.
    overperf > 0 = team is winning more than Elo expects (hot streak vs strong opponents)
    overperf < 0 = team is underperforming (losing to weaker teams)
    """
    df = df.sort_values("date").reset_index(drop=True)

    team_overperf = {}  # team_id -> list of (actual_pts - expected_pts) per match
    cols_h, cols_a = [], []

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        # Pre-match: average overperf over last 5
        for team_id, col_list in [(h_id, cols_h), (a_id, cols_a)]:
            hist = team_overperf.get(team_id, [])
            last5 = hist[-5:]
            col_list.append(float(np.mean(last5)) if last5 else 0.0)

        # Post-match: compute overperf for this match
        elo_h = row.get("elo_home", ELO_INITIAL)
        elo_a = row.get("elo_away", ELO_INITIAL)
        exp_h = 1.0 / (1.0 + 10.0 ** ((elo_a - (elo_h + ELO_HOME_ADV)) / 400.0))
        exp_a = 1.0 - exp_h

        # Points: W=3, D=1, L=0 → normalized to [0,1] scale
        if hg > ag:
            pts_h, pts_a = 1.0, 0.0
        elif hg == ag:
            pts_h, pts_a = 0.33, 0.33
        else:
            pts_h, pts_a = 0.0, 1.0

        # Overperf = actual - expected
        team_overperf.setdefault(h_id, []).append(pts_h - exp_h)
        team_overperf.setdefault(a_id, []).append(pts_a - exp_a)

    df["overperf_home"] = cols_h
    df["overperf_away"] = cols_a
    df["overperf_diff"] = df["overperf_home"] - df["overperf_away"]
    return df


def compute_draw_aware_features(df: pd.DataFrame) -> pd.DataFrame:
    """Draw-aware features (ABE #4).

    Argentina is draw-heavy. Standard Elo treats draws as 0.5 for both
    teams, but some teams and matchup types draw disproportionately.

    Features:
    - draw_tendency: how often this team draws relative to Elo-expected
    - draw_elo_interaction: when Elo is close, draws are more likely
    - league_draw_rate: overall draw rate in the league (context)
    """
    df = df.sort_values("date").reset_index(drop=True)

    # League-wide draw rate (rolling, using all matches so far)
    total_matches = 0
    total_draws = 0

    team_draws = {}  # team_id -> list of (is_draw,)
    cols_tend_h, cols_tend_a = [], []
    cols_league_dr = []

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]
        is_draw = 1.0 if hg == ag else 0.0

        # League draw rate up to now
        cols_league_dr.append(total_draws / max(total_matches, 1))

        # Team draw tendency (last 10 matches)
        for team_id, col_list in [(h_id, cols_tend_h), (a_id, cols_tend_a)]:
            hist = team_draws.get(team_id, [])
            last10 = hist[-10:]
            col_list.append(float(np.mean(last10)) if last10 else 0.27)  # default ~league avg

        # Update
        total_matches += 1
        total_draws += is_draw
        team_draws.setdefault(h_id, []).append(is_draw)
        team_draws.setdefault(a_id, []).append(is_draw)

    df["draw_tendency_home"] = cols_tend_h
    df["draw_tendency_away"] = cols_tend_a
    df["league_draw_rate"] = cols_league_dr

    # Interaction: when Elo is close AND both teams draw a lot, draw is more likely
    elo_closeness = 1.0 - abs(df["elo_prob_home"] - df["elo_prob_away"])
    df["draw_elo_interaction"] = (
        elo_closeness * (df["draw_tendency_home"] + df["draw_tendency_away"]) / 2
    )
    return df


def compute_home_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team home advantage bias (ABE #5).

    Some teams are disproportionately strong at home (atmosphere, altitude,
    travel distance for visitors). This measures deviation from league average
    with shrinkage towards 0 (regularized).
    """
    df = df.sort_values("date").reset_index(drop=True)

    team_home_record = {}  # team_id -> {"home_wins": int, "home_matches": int,
                           #              "away_wins": int, "away_matches": int}
    cols_bias_h, cols_bias_a = [], []

    # League average home win rate (rolling)
    total_home_wins = 0
    total_matches = 0

    SHRINKAGE_N = 10  # Regularization: need N home matches before full weight

    for _, row in df.iterrows():
        h_id, a_id = row["home_team_id"], row["away_team_id"]
        hg, ag = row["home_goals"], row["away_goals"]

        league_home_wr = total_home_wins / max(total_matches, 1) if total_matches > 0 else 0.45

        # Home team's home bias
        rec_h = team_home_record.get(h_id, {"hw": 0, "hm": 0, "aw": 0, "am": 0})
        if rec_h["hm"] >= 3:
            raw_home_wr = rec_h["hw"] / rec_h["hm"]
            raw_away_wr = rec_h["aw"] / max(rec_h["am"], 1)
            raw_bias = raw_home_wr - raw_away_wr
            # Shrinkage: weight towards 0 based on sample size
            shrink_w = min(rec_h["hm"] / SHRINKAGE_N, 1.0)
            cols_bias_h.append(raw_bias * shrink_w)
        else:
            cols_bias_h.append(0.0)

        # Away team's home bias (how strong they are at home → less impact when away)
        rec_a = team_home_record.get(a_id, {"hw": 0, "hm": 0, "aw": 0, "am": 0})
        if rec_a["hm"] >= 3:
            raw_home_wr = rec_a["hw"] / rec_a["hm"]
            raw_away_wr = rec_a["aw"] / max(rec_a["am"], 1)
            raw_bias = raw_home_wr - raw_away_wr
            shrink_w = min(rec_a["hm"] / SHRINKAGE_N, 1.0)
            cols_bias_a.append(raw_bias * shrink_w)
        else:
            cols_bias_a.append(0.0)

        # Update records
        is_h_win = 1 if hg > ag else 0
        is_a_win = 1 if ag > hg else 0
        total_home_wins += is_h_win
        total_matches += 1

        rec_h = team_home_record.setdefault(h_id, {"hw": 0, "hm": 0, "aw": 0, "am": 0})
        rec_h["hw"] += is_h_win
        rec_h["hm"] += 1

        rec_a = team_home_record.setdefault(a_id, {"hw": 0, "hm": 0, "aw": 0, "am": 0})
        rec_a["aw"] += is_a_win
        rec_a["am"] += 1

    df["home_bias_home"] = cols_bias_h
    df["home_bias_away"] = cols_bias_a
    df["home_bias_diff"] = df["home_bias_home"] - df["home_bias_away"]
    return df


# ─── WAVE 9: Kimi Features (interactions, efficiency) ───────

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative interaction features (Kimi #1).

    Football effects are multiplicative, not additive.
    The favorite (high Elo) suffers more from fatigue than the underdog.
    Strong defense matters more when facing a strong attack.
    """
    # elo_diff × rest_diff: fatigue affects favorites more
    df["elo_x_rest"] = df["elo_diff"] * df["rest_diff"]

    # elo_diff × season_phase: Elo matters more early season? Or late?
    df["elo_x_season"] = df["elo_diff"] * df["season_phase"]

    # elo_diff × defensive quality: Elo advantage amplified by good defense
    df["elo_x_defense"] = df["elo_diff"] * (
        df["away_goals_conceded_avg"] - df["home_goals_conceded_avg"]
    )

    # form × defense: hot team + solid defense = danger
    df["form_x_defense"] = df["form_diff"] * (
        df["home_goals_conceded_avg"] - df["away_goals_conceded_avg"]
    )

    return df


def compute_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tactical efficiency ratios (Kimi #4).

    Goals/shots = finishing efficiency. A team with 1.5 goals from 10 shots
    is more efficient than one with 1.5 goals from 15 shots.
    This captures quality of chances without needing xG.
    """
    # Finishing efficiency: goals per shot (avoid div by zero)
    h_shots = df["home_shots_avg"].clip(lower=1.0)
    a_shots = df["away_shots_avg"].clip(lower=1.0)

    df["finish_eff_home"] = df["home_goals_scored_avg"] / h_shots
    df["finish_eff_away"] = df["away_goals_scored_avg"] / a_shots

    # Defensive efficiency: goals conceded per opponent shot
    # Lower = better defense (concede fewer goals per shot faced)
    df["def_eff_home"] = df["home_goals_conceded_avg"] / a_shots
    df["def_eff_away"] = df["away_goals_conceded_avg"] / h_shots

    # Net efficiency difference
    df["efficiency_diff"] = (
        (df["finish_eff_home"] - df["def_eff_home"]) -
        (df["finish_eff_away"] - df["def_eff_away"])
    )
    return df


# ─── Master Feature Computation ──────────────────────────────

def compute_all_experimental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature computation waves."""
    print("  [Wave 2] Elo variants (goal-weighted, K-sweep, split, momentum)...")
    df = compute_elo_goal_weighted(df)
    df = compute_elo_k_sweep(df)
    df = compute_elo_home_away_split(df)
    df = compute_elo_momentum(df)

    print("  [Wave 3] Form features (streaks, volatility, clean sheets)...")
    df = compute_form_features(df)

    print("  [Wave 4] Matchup features (confrontation, H2H)...")
    df = compute_matchup_features(df)
    df = compute_h2h_features(df)

    print("  [Wave 5] Surprise & meta features...")
    df = compute_surprise_features(df)
    df = compute_elo_expected_probs(df)

    print("  [Wave 6] Calendar features...")
    df = compute_calendar_features(df)

    print("  [Wave 7] Opponent-adjusted ratings (ABE)...")
    df = compute_opponent_adjusted_ratings(df)

    print("  [Wave 8] Overperf + draw-aware + home bias (ABE)...")
    df = compute_overperformance(df)
    df = compute_draw_aware_features(df)
    df = compute_home_bias(df)

    print("  [Wave 9] Interactions + efficiency (Kimi)...")
    df = compute_interaction_features(df)
    df = compute_efficiency_features(df)

    return df


# ─── Universe System (Fix 1 + Fix 3) ─────────────────────────

# All optional feature groups that define universe boundaries
_ODDS_SET = set(ODDS_FEATURES) | set(IMPLIED_DRAW_FEATURES)  # implied_draw requires odds
_XG_SET = set(XG_ALL)  # XG_CORE + XG_DEFENSE + XG_OVERPERF
_XI_SET = set(XI_CONTINUITY_FEATURES)
_MTV_SET = set(MTV_FEATURES)
_GEO_SET = set(GEO_FEATURES)
_STANDINGS_SET = set(STANDINGS_FEATURES)
_PRECIP_SET = set(PRECIP_FEATURES)


def classify_test_universe(feature_names: list) -> str:
    """Determine which universe a test belongs to based on its features."""
    feats = set(feature_names)
    needs_odds = bool(feats & _ODDS_SET)
    needs_xg = bool(feats & _XG_SET)
    needs_xi = bool(feats & _XI_SET)
    needs_mtv = bool(feats & _MTV_SET)
    needs_geo = bool(feats & _GEO_SET)
    needs_standings = bool(feats & _STANDINGS_SET)
    needs_precip = bool(feats & _PRECIP_SET)
    parts = []
    if needs_mtv:
        parts.append("mtv")
    if needs_xi:
        parts.append("xi")
    if needs_geo:
        parts.append("geo")
    if needs_standings:
        parts.append("st")
    if needs_precip:
        parts.append("precip")
    if needs_odds:
        parts.append("odds")
    if needs_xg:
        parts.append("xg")
    return "_".join(parts) if parts else "base"


def compute_universes(df: pd.DataFrame, tests_dict: dict) -> dict:
    """Pre-compute fixed universes for fair comparison across tests.

    Returns dict mapping universe_id -> DataFrame, all sorted by ["date","match_id"]
    with NaN-free base features. Tests within the same universe share identical
    N, split_idx, and split_date.

    Universes (2^4 possible, only those needed are populated):
      - base:            all non-optional features present
      - odds:            base ∩ valid odds triplet (odds_home > 1.0)
      - xg:              base ∩ xG core present
      - odds_xg:         base ∩ odds ∩ xG
      - xi:              base ∩ xi_continuity present
      - xi_odds:         base ∩ xi ∩ odds
      - xi_xg:           base ∩ xi ∩ xG
      - xi_odds_xg:      base ∩ xi ∩ odds ∩ xG
      - mtv:             base ∩ talent_delta present
      - mtv_odds:        base ∩ mtv ∩ odds
      - mtv_xg:          base ∩ mtv ∩ xG
      - mtv_odds_xg:     base ∩ mtv ∩ odds ∩ xG
      - mtv_xi:          base ∩ mtv ∩ xi
      - mtv_xi_odds:     base ∩ mtv ∩ xi ∩ odds
      - mtv_xi_odds_xg:  base ∩ mtv ∩ xi ∩ odds ∩ xG
    """
    _ALL_OPTIONAL = (_ODDS_SET | _XG_SET | _XI_SET | _MTV_SET |
                     _GEO_SET | _STANDINGS_SET | _PRECIP_SET)

    # Collect all "base" features (non-optional) across all tests
    all_base_feats = set()
    for feats in tests_dict.values():
        non_optional = [f for f in feats if f not in _ALL_OPTIONAL]
        all_base_feats.update(non_optional)

    # Only keep features that actually exist in the dataframe
    available_base = sorted(f for f in all_base_feats if f in df.columns)
    missing_base = all_base_feats - set(df.columns)
    if missing_base:
        print(f"  [UNIVERSE] Warning: {len(missing_base)} base features not in data: "
              f"{sorted(missing_base)[:5]}...")

    # Universe: base (dropna on all base features)
    df_base = df.dropna(subset=available_base).copy()
    df_base = df_base.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Determine which universes are actually needed
    needed = set()
    for feats in tests_dict.values():
        needed.add(classify_test_universe(feats))

    # Build filter masks on df_base
    odds_mask = (df_base["odds_home"].notna() &
                 df_base["odds_draw"].notna() &
                 df_base["odds_away"].notna() &
                 (df_base["odds_home"] > 1.0))

    xg_cols = [c for c in XG_CORE if c in df_base.columns]
    xg_mask = df_base[xg_cols].notna().all(axis=1) if xg_cols else pd.Series(False, index=df_base.index)

    xi_cols = [c for c in XI_CONTINUITY_FEATURES if c in df_base.columns]
    xi_mask = df_base[xi_cols].notna().all(axis=1) if xi_cols else pd.Series(False, index=df_base.index)

    mtv_cols = [c for c in MTV_FEATURES if c in df_base.columns]
    mtv_mask = df_base[mtv_cols].notna().all(axis=1) if mtv_cols else pd.Series(False, index=df_base.index)

    geo_cols = [c for c in GEO_FEATURES if c in df_base.columns]
    geo_mask = df_base[geo_cols].notna().all(axis=1) if geo_cols else pd.Series(False, index=df_base.index)

    st_cols = [c for c in STANDINGS_FEATURES if c in df_base.columns]
    st_mask = df_base[st_cols].notna().all(axis=1) if st_cols else pd.Series(False, index=df_base.index)

    precip_cols = [c for c in PRECIP_FEATURES if c in df_base.columns]
    precip_mask = df_base[precip_cols].notna().all(axis=1) if precip_cols else pd.Series(False, index=df_base.index)

    # Build universes
    universes: dict[str, pd.DataFrame] = {"base": df_base}

    def _add(uid: str, mask: pd.Series) -> None:
        if uid in needed:
            universes[uid] = df_base[mask].copy().reset_index(drop=True)

    _add("odds", odds_mask)
    _add("xg", xg_mask)
    _add("odds_xg", odds_mask & xg_mask)
    _add("xi", xi_mask)
    _add("xi_odds", xi_mask & odds_mask)
    _add("xi_xg", xi_mask & xg_mask)
    _add("xi_odds_xg", xi_mask & odds_mask & xg_mask)
    _add("mtv", mtv_mask)
    _add("mtv_odds", mtv_mask & odds_mask)
    _add("mtv_xg", mtv_mask & xg_mask)
    _add("mtv_odds_xg", mtv_mask & odds_mask & xg_mask)
    _add("mtv_xi", mtv_mask & xi_mask)
    _add("mtv_xi_odds", mtv_mask & xi_mask & odds_mask)
    _add("mtv_xi_odds_xg", mtv_mask & xi_mask & odds_mask & xg_mask)
    # Geo universes
    _add("geo", geo_mask)
    _add("geo_odds", geo_mask & odds_mask)
    _add("geo_st", geo_mask & st_mask)
    _add("geo_st_odds", geo_mask & st_mask & odds_mask)
    # Standings universes
    _add("st", st_mask)
    _add("st_odds", st_mask & odds_mask)
    _add("st_odds_xg", st_mask & odds_mask & xg_mask)
    # Precipitation universes
    _add("precip", precip_mask)
    _add("precip_odds", precip_mask & odds_mask)

    # Report
    for uid, udf in universes.items():
        if udf.empty:
            print(f"  [UNIVERSE] {uid}: EMPTY (0 rows)")
        else:
            split_idx = int(len(udf) * (1 - TEST_FRACTION))
            split_date = udf.iloc[split_idx]["date"]
            print(f"  [UNIVERSE] {uid}: N={len(udf)} "
                  f"(train={split_idx}, test={len(udf)-split_idx}, "
                  f"split={split_date})")

    return universes


# ─── Model Training & Evaluation ─────────────────────────────

def multiclass_brier(y_true, y_prob):
    """Multi-class Brier score (lower = better)."""
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def brier_decomposition(y_true, y_prob, n_bins=10):
    """Murphy (1973) multiclass Brier decomposition.

    Brier = Reliability - Resolution + Uncertainty
    - Reliability (REL): penalizes miscalibration (lower = better)
    - Resolution (RES): rewards discrimination (higher = better)
    - Uncertainty (UNC): irreducible, depends on class distribution

    Returns dict with reliability, resolution, uncertainty,
    brier_reconstructed (REL - RES + UNC, approx equals Brier).
    Note: binned approximation — reconstruction error grows with few bins
    or skewed distributions. Tolerance < 0.01 is expected.
    """
    n = len(y_true)
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]

    # Uncertainty (base rate)
    base_rates = y_onehot.mean(axis=0)
    unc = float(np.sum(base_rates * (1 - base_rates)))

    # Bin by predicted probability per class
    rel_total, res_total = 0.0, 0.0
    empty_bins = 0
    total_bins = n_classes * n_bins
    for k in range(n_classes):
        probs_k = y_prob[:, k]
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs_k, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        for b in range(n_bins):
            mask = bin_indices == b
            n_b = mask.sum()
            if n_b == 0:
                empty_bins += 1
                continue
            avg_pred = float(probs_k[mask].mean())
            avg_obs = float(y_onehot[mask, k].mean())
            rel_total += n_b * (avg_pred - avg_obs) ** 2
            res_total += n_b * (avg_obs - base_rates[k]) ** 2

    rel_total /= n
    res_total /= n
    reconstructed = rel_total - res_total + unc

    return {
        "reliability": round(rel_total, 6),
        "resolution": round(res_total, 6),
        "uncertainty": round(unc, 6),
        "brier_reconstructed": round(reconstructed, 6),
        "n_bins": n_bins,
        "empty_bins": empty_bins,
        "total_bins": total_bins,
    }


def multiclass_ece(y_true, y_prob, n_bins=15):
    """Per-class Expected Calibration Error (ECE).

    Returns dict: {ece_home, ece_draw, ece_away, ece_avg, curve_data}
    """
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    class_names = ["home", "draw", "away"]
    result = {}
    curve_data = []

    for k, name in enumerate(class_names):
        probs_k = y_prob[:, k]
        true_k = y_onehot[:, k]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs_k >= bins[i]) & (probs_k < bins[i + 1])
            n_b = mask.sum()
            if n_b == 0:
                continue
            avg_pred = float(probs_k[mask].mean())
            avg_obs = float(true_k[mask].mean())
            ece += (n_b / len(probs_k)) * abs(avg_pred - avg_obs)
            curve_data.append({
                "class": name,
                "bin_mid": round((bins[i] + bins[i + 1]) / 2, 4),
                "avg_predicted": round(avg_pred, 5),
                "avg_observed": round(avg_obs, 5),
                "n": int(n_b),
            })
        result["ece_%s" % name] = round(ece, 5)

    result["ece_avg"] = round(
        np.mean([result["ece_%s" % n] for n in class_names]), 5)
    result["curve_data"] = curve_data
    return result


def _diagnose_cal_vs_res(decomp, ece_result):
    """Classify model weakness as CALIBRATION_ISSUE or RESOLUTION_ISSUE."""
    rel = decomp["reliability"]
    res = decomp["resolution"]
    ece_avg = ece_result["ece_avg"]
    if ece_avg > 0.05 and rel > res:
        return "CALIBRATION_ISSUE"
    return "RESOLUTION_ISSUE"


def _prepare_dataset(df_universe: pd.DataFrame, feature_names: list,
                     test_name: str, min_total: int = 100,
                     min_test: int = 50,
                     lockbox_mode: bool = False) -> Optional[dict]:
    """Temporal split on pre-filtered universe. Shared by evaluate/optuna/shap.

    The universe DataFrame is already filtered by compute_universes() — no dropna
    here. This ensures all tests within the same universe share identical N,
    split_idx, and split_date for fair comparison.

    Returns dict with keys: df_train, df_test, X_tr, y_tr, X_te, y_te, df_sorted.
    In lockbox mode also: df_val, X_val, y_val, df_lockbox, X_lock, y_lock.
    Returns {"error": ...} dict on failure.
    """
    missing = [f for f in feature_names if f not in df_universe.columns]
    if missing:
        return {"error": f"missing_features: {missing}"}

    # Defensive: verify no NaN in required features (universe should be clean)
    nan_rows = df_universe[feature_names].isna().any(axis=1).sum()
    if nan_rows > 0:
        return {"error": f"universe_nan_leak: {nan_rows} rows with NaN in {test_name}"}

    if len(df_universe) < min_total:
        return {"error": f"insufficient_data: {len(df_universe)}"}

    # Already sorted by compute_universes() — ["date", "match_id"]
    df_sorted = df_universe

    if lockbox_mode:
        # 70/15/15 split
        n = len(df_sorted)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        df_train = df_sorted.iloc[:train_end]
        df_val = df_sorted.iloc[train_end:val_end]
        df_lockbox = df_sorted.iloc[val_end:]

        if len(df_val) < min_test:
            return {"error": f"insufficient_val: {len(df_val)}"}
        if len(df_lockbox) < min_test:
            return {"error": f"insufficient_lockbox: {len(df_lockbox)}"}

        return {
            "df_train": df_train, "df_test": df_val, "df_sorted": df_sorted,
            "X_tr": df_train[feature_names].values.astype(np.float32),
            "y_tr": df_train["result"].values.astype(int),
            "X_te": df_val[feature_names].values.astype(np.float32),
            "y_te": df_val["result"].values.astype(int),
            "df_val": df_val,
            "df_lockbox": df_lockbox,
            "X_lock": df_lockbox[feature_names].values.astype(np.float32),
            "y_lock": df_lockbox["result"].values.astype(int),
        }

    # Standard 80/20 split
    split_idx = int(len(df_sorted) * (1 - TEST_FRACTION))
    df_train = df_sorted.iloc[:split_idx]
    df_test = df_sorted.iloc[split_idx:]

    if len(df_test) < min_test:
        return {"error": f"insufficient_test: {len(df_test)}"}

    X_tr = df_train[feature_names].values.astype(np.float32)
    y_tr = df_train["result"].values.astype(int)
    X_te = df_test[feature_names].values.astype(np.float32)
    y_te = df_test["result"].values.astype(int)

    return {
        "df_train": df_train, "df_test": df_test, "df_sorted": df_sorted,
        "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te,
    }


def train_xgb(X_train, y_train, seed=42):
    """Train XGBoost with production hyperparams."""
    params = {**PROD_HYPERPARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    sample_weight[y_train == 1] = DRAW_WEIGHT
    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    return model


# ─── Two-Stage Training (Shadow architecture) ────────────────

def train_two_stage(X_train, y_train, feature_names, seed=42):
    """Train Two-Stage model: Stage 1 (draw vs non-draw) + Stage 2 (home vs away).

    Stage 1: Binary classifier on all features (including implied_draw if present).
    Stage 2: Binary classifier on non-draw samples only (excludes implied_draw).
    Composition: p_draw = S1, p_home = (1-p_draw)*S2_home, p_away = (1-p_draw)*S2_away.
    """
    # Stage 1: draw (1) vs non-draw (0)
    y_s1 = (y_train == 1).astype(int)
    params_s1 = {**TWO_STAGE_PARAMS_S1, "random_state": seed}
    model_s1 = xgb.XGBClassifier(**params_s1)
    sw_s1 = np.ones(len(y_s1), dtype=np.float32)
    sw_s1[y_s1 == 1] = TWO_STAGE_DRAW_WEIGHT
    model_s1.fit(X_train, y_s1, sample_weight=sw_s1, verbose=False)

    # Stage 2: fav (1) vs underdog (0), non-draw only
    non_draw = y_train != 1

    # Check if odds are available in features to determine favorite
    if "odds_home" in feature_names and "odds_away" in feature_names:
        h_idx = feature_names.index("odds_home")
        a_idx = feature_names.index("odds_away")
        odds_h = X_train[non_draw][:, h_idx]
        odds_a = X_train[non_draw][:, a_idx]
        # NaN-safe vectorization
        valid_odds = ~np.isnan(odds_h) & ~np.isnan(odds_a) & (odds_h > 0) & (odds_a > 0)
        is_home_fav = np.where(valid_odds, odds_h <= odds_a, True)
    else:
        # Fallback si el test de la Sección W no incluye odds
        is_home_fav = np.ones(non_draw.sum(), dtype=bool)
    home_won = (y_train[non_draw] == 0)
    y_s2 = np.where(is_home_fav, home_won, ~home_won).astype(int)

    # Remove implied_draw from Stage 2 (it's a draw-specific signal)
    if "implied_draw" in feature_names:
        id_idx = feature_names.index("implied_draw")
        s2_cols = [i for i in range(X_train.shape[1]) if i != id_idx]
    else:
        s2_cols = list(range(X_train.shape[1]))

    X_s2 = X_train[non_draw][:, s2_cols]
    params_s2 = {**TWO_STAGE_PARAMS_S2, "random_state": seed}
    model_s2 = xgb.XGBClassifier(**params_s2)
    model_s2.fit(X_s2, y_s2, verbose=False)

    return model_s1, model_s2, s2_cols


def predict_two_stage(model_s1, model_s2, X_test, s2_cols, feature_names):
    """Predict probabilities using Two-Stage composition (fav/underdog semantic).

    Returns (N, 3) array: [p_home, p_draw, p_away] summing to 1.0.
    """
    p_draw = model_s1.predict_proba(X_test)[:, 1]
    X_s2 = X_test[:, s2_cols]

    p_s2_raw = model_s2.predict_proba(X_s2)[:, 1]

    if "odds_home" in feature_names and "odds_away" in feature_names:
        h_idx = feature_names.index("odds_home")
        a_idx = feature_names.index("odds_away")
        odds_h = X_test[:, h_idx]
        odds_a = X_test[:, a_idx]

        valid_odds = ~np.isnan(odds_h) & ~np.isnan(odds_a) & (odds_h > 0) & (odds_a > 0)
        is_home_fav = np.where(valid_odds, odds_h <= odds_a, True)
    else:
        is_home_fav = np.ones(len(X_test), dtype=bool)

    # Swap-back: if home is fav, p_s2_raw is P(home | non-draw). If away is fav, it's P(away | non-draw)
    p_home_given_nd = np.where(is_home_fav, p_s2_raw, 1 - p_s2_raw)

    p_home = (1 - p_draw) * p_home_given_nd
    p_away = (1 - p_draw) * (1 - p_home_given_nd)

    return np.column_stack([p_home, p_draw, p_away])


def compute_implied_draw(df):
    """Compute implied_draw from odds (pre-kickoff market draw probability)."""
    has_odds = (df["odds_home"].notna() & df["odds_draw"].notna() &
                df["odds_away"].notna() & (df["odds_draw"] > 0))
    inv_h = 1.0 / df["odds_home"].where(df["odds_home"] > 0)
    inv_d = 1.0 / df["odds_draw"].where(df["odds_draw"] > 0)
    inv_a = 1.0 / df["odds_away"].where(df["odds_away"] > 0)
    total = inv_h + inv_d + inv_a
    df["implied_draw"] = (inv_d / total).where(has_odds, np.nan)
    return df


# ─── Market Residual helpers ─────────────────────────────────

def compute_market_logits(df, odds_cols=("odds_home", "odds_draw", "odds_away")):
    """Convert de-vigged odds to log-probabilities for base_margin."""
    inv = np.column_stack([1.0 / df[c].astype(np.float64).values for c in odds_cols])
    probs = inv / inv.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(probs)


def train_xgb_residual(X_train, y_train, market_logits_train, seed=42):
    """Train XGBoost with market priors via base_margin.

    The model starts from market probabilities and learns only
    a small correction g(x). With strong regularization, g(x) → 0
    if no systematic bias exists.
    """
    params = {**RESIDUAL_HYPERPARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    sample_weight[y_train == 1] = DRAW_WEIGHT
    model.fit(X_train, y_train,
              sample_weight=sample_weight,
              base_margin=market_logits_train.flatten(),
              verbose=False)
    return model


def predict_with_base_margin(model, X_test, market_logits_test):
    """Predict probabilities with base_margin via DMatrix.

    Using DMatrix directly is the most reliable path across xgboost versions.
    predict_proba's inplace_predict expects (N,3) but fit() expects (N*3,),
    so we bypass predict_proba entirely.
    """
    dtest = xgb.DMatrix(X_test, base_margin=market_logits_test.flatten())
    raw = model.get_booster().predict(dtest)
    return raw.reshape(-1, 3)


def bootstrap_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap 95% CI for Brier score."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    briers = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b = multiclass_brier(y_true[idx], y_prob[idx])
        briers.append(b)
    return float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))


def bootstrap_paired_delta(y_true, y_prob_model, y_prob_market,
                           n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap 95% CI for Δ(model_brier - market_brier) using paired samples.

    Uses per-match Brier contributions so both model and market are evaluated
    on exactly the same bootstrap samples → captures correlation.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    n_classes = y_prob_model.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]

    # Per-match Brier contributions
    model_per_match = np.sum((y_prob_model - y_onehot) ** 2, axis=1)
    market_per_match = np.sum((y_prob_market - y_onehot) ** 2, axis=1)
    delta_per_match = model_per_match - market_per_match

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        deltas.append(float(np.mean(delta_per_match[idx])))

    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


OPTUNA_N_TRIALS = 50
OPTUNA_CV_FOLDS = 3


def optuna_tune(X_train, y_train, n_trials=OPTUNA_N_TRIALS, n_folds=OPTUNA_CV_FOLDS,
                seed=42) -> tuple[dict, float]:
    """Find optimal XGBoost hyperparams via Optuna with temporal CV on train set.

    Returns (best_params_dict, best_cv_brier).
    Uses 3-fold forward-chaining (temporal) to avoid future leakage.
    """
    n = len(X_train)
    fold_size = n // (n_folds + 1)
    # Forward-chaining folds: train on [0:i], validate on [i:i+fold_size]
    folds = []
    for i in range(1, n_folds + 1):
        tr_end = fold_size * i
        va_end = min(tr_end + fold_size, n)
        if va_end > tr_end:
            folds.append((list(range(0, tr_end)), list(range(tr_end, va_end))))

    sample_weight_full = np.ones(len(y_train), dtype=np.float32)
    sample_weight_full[y_train == 1] = DRAW_WEIGHT

    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "random_state": seed,
        }

        briers = []
        for tr_idx, va_idx in folds:
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]
            sw_tr = sample_weight_full[tr_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
            y_prob = model.predict_proba(X_va)
            briers.append(multiclass_brier(y_va, y_prob))

        return float(np.mean(briers))

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    })
    return best, study.best_value


def evaluate_feature_set_optuna(df_universe: pd.DataFrame, feature_names: list,
                                test_name: str) -> Optional[dict]:
    """Like evaluate_feature_set but with Optuna-tuned hyperparams per test."""
    prep = _prepare_dataset(df_universe, feature_names, test_name)
    if "error" in prep:
        return {"test": test_name, **prep}

    df_train, df_test, df_sorted = prep["df_train"], prep["df_test"], prep["df_sorted"]
    X_tr, y_tr, X_te, y_te = prep["X_tr"], prep["y_tr"], prep["X_te"], prep["y_te"]

    # Optuna tuning on train set
    print(f"    Optuna tuning ({OPTUNA_N_TRIALS} trials, {OPTUNA_CV_FOLDS}-fold temporal CV)...")
    best_params, cv_brier = optuna_tune(X_tr, y_tr)
    print(f"    Best CV Brier: {cv_brier:.5f} | depth={best_params['max_depth']} "
          f"lr={best_params['learning_rate']:.4f} n_est={best_params['n_estimators']} "
          f"mcw={best_params['min_child_weight']}")

    # Evaluate with tuned params (multi-seed) — collect for ensemble
    all_briers, all_logloss, all_accuracy = [], [], []
    all_y_probs = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 42 + 7
        params = {**best_params, "random_state": seed}
        model = xgb.XGBClassifier(**params)
        sw = np.ones(len(y_tr), dtype=np.float32)
        sw[y_tr == 1] = DRAW_WEIGHT
        model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        y_prob = model.predict_proba(X_te)

        all_briers.append(multiclass_brier(y_te, y_prob))
        all_logloss.append(log_loss(y_te, y_prob, labels=[0, 1, 2]))
        all_accuracy.append(float(np.mean(model.predict(X_te) == y_te)))
        all_y_probs.append(y_prob)

    # Fix 2: CI from ensemble
    ensemble_prob = np.mean(all_y_probs, axis=0)
    brier_ensemble = multiclass_brier(y_te, ensemble_prob)
    ci_lo, ci_hi = bootstrap_ci(y_te, ensemble_prob)

    # Diagnostic decomposition (FS-01 / FS-02)
    diag = {}
    if _DECOMPOSE:
        decomp_10 = brier_decomposition(y_te, ensemble_prob, n_bins=10)
        decomp_20 = brier_decomposition(y_te, ensemble_prob, n_bins=20)
        ece = multiclass_ece(y_te, ensemble_prob, n_bins=15)
        diag = {
            "brier_decomposition_10": decomp_10,
            "brier_decomposition_20": decomp_20,
            "ece": {k: v for k, v in ece.items() if k != "curve_data"},
            "diagnosis": _diagnose_cal_vs_res(decomp_10, ece),
        }

    result = {
        "test": test_name,
        "universe": classify_test_universe(feature_names),
        "n_features": len(feature_names),
        "features": feature_names,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "date_range": [str(df_sorted["date"].min()), str(df_sorted["date"].max())],
        "split_date": str(df_test["date"].min()),
        "brier_ensemble": round(brier_ensemble, 5),
        "brier_ci95": [round(ci_lo, 5), round(ci_hi, 5)],
        "brier_seed_mean": round(float(np.mean(all_briers)), 5),
        "brier_seed_std": round(float(np.std(all_briers)), 5),
        "brier_seed_range": [round(min(all_briers), 5), round(max(all_briers), 5)],
        "logloss_mean": round(float(np.mean(all_logloss)), 5),
        "accuracy_mean": round(float(np.mean(all_accuracy)), 4),
        "optuna_params": {k: v for k, v in best_params.items()
                          if k not in ("objective", "num_class",
                                       "eval_metric", "verbosity")},
        "optuna_cv_brier": round(cv_brier, 5),
        "optuna_n_trials": OPTUNA_N_TRIALS,
    }
    result.update(diag)
    return result


def evaluate_feature_set(df_universe: pd.DataFrame, feature_names: list,
                         test_name: str,
                         lockbox_mode: bool = False) -> Optional[dict]:
    """Train & evaluate a feature set. Returns metrics dict or None if insufficient data."""
    prep = _prepare_dataset(df_universe, feature_names, test_name,
                            lockbox_mode=lockbox_mode)
    if "error" in prep:
        return {"test": test_name, **prep}

    df_train, df_test, df_sorted = prep["df_train"], prep["df_test"], prep["df_sorted"]
    X_tr, y_tr, X_te, y_te = prep["X_tr"], prep["y_tr"], prep["X_te"], prep["y_te"]

    # Multi-seed evaluation — collect all predictions for ensemble
    all_briers, all_logloss, all_accuracy = [], [], []
    all_y_probs = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 42 + 7
        model = train_xgb(X_tr, y_tr, seed=seed)
        y_prob = model.predict_proba(X_te)

        all_briers.append(multiclass_brier(y_te, y_prob))
        all_logloss.append(log_loss(y_te, y_prob, labels=[0, 1, 2]))
        all_accuracy.append(float(np.mean(model.predict(X_te) == y_te)))
        all_y_probs.append(y_prob)

    # Fix 2: CI from ensemble (average of all seed predictions)
    ensemble_prob = np.mean(all_y_probs, axis=0)
    brier_ensemble = multiclass_brier(y_te, ensemble_prob)
    ci_lo, ci_hi = bootstrap_ci(y_te, ensemble_prob)

    # Diagnostic decomposition (FS-01 / FS-02)
    diag = {}
    if _DECOMPOSE:
        decomp_10 = brier_decomposition(y_te, ensemble_prob, n_bins=10)
        decomp_20 = brier_decomposition(y_te, ensemble_prob, n_bins=20)
        ece = multiclass_ece(y_te, ensemble_prob, n_bins=15)
        diag = {
            "brier_decomposition_10": decomp_10,
            "brier_decomposition_20": decomp_20,
            "ece": {k: v for k, v in ece.items() if k != "curve_data"},
            "diagnosis": _diagnose_cal_vs_res(decomp_10, ece),
        }

    # Label distribution in test
    test_dist = df_test["result"].value_counts().sort_index()
    test_pct = (test_dist / len(df_test) * 100).round(1).to_dict()

    result = {
        "test": test_name,
        "universe": classify_test_universe(feature_names),
        "n_features": len(feature_names),
        "features": feature_names,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "date_range": [str(df_sorted["date"].min()), str(df_sorted["date"].max())],
        "split_date": str(df_test["date"].min()),
        "brier_ensemble": round(brier_ensemble, 5),
        "brier_ci95": [round(ci_lo, 5), round(ci_hi, 5)],
        "brier_seed_mean": round(float(np.mean(all_briers)), 5),
        "brier_seed_std": round(float(np.std(all_briers)), 5),
        "brier_seed_range": [round(min(all_briers), 5), round(max(all_briers), 5)],
        "logloss_mean": round(float(np.mean(all_logloss)), 5),
        "accuracy_mean": round(float(np.mean(all_accuracy)), 4),
        "test_distribution": {
            "H": test_pct.get(0, 0),
            "D": test_pct.get(1, 0),
            "A": test_pct.get(2, 0),
        },
    }
    result.update(diag)
    return result


def evaluate_two_stage(df_universe, feature_names, test_name,
                       lockbox_mode=False):
    """Train & evaluate a Two-Stage model (Shadow architecture).

    Same pattern as evaluate_feature_set but uses train_two_stage/predict_two_stage.
    """
    prep = _prepare_dataset(df_universe, feature_names, test_name,
                            lockbox_mode=lockbox_mode)
    if "error" in prep:
        return {"test": test_name, **prep}

    df_train, df_test, df_sorted = prep["df_train"], prep["df_test"], prep["df_sorted"]
    X_tr, y_tr, X_te, y_te = prep["X_tr"], prep["y_tr"], prep["X_te"], prep["y_te"]

    all_briers, all_logloss, all_accuracy = [], [], []
    all_y_probs = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 42 + 7
        s1, s2, s2_cols = train_two_stage(X_tr, y_tr, feature_names, seed=seed)
        y_prob = predict_two_stage(s1, s2, X_te, s2_cols, feature_names)

        all_briers.append(multiclass_brier(y_te, y_prob))
        all_logloss.append(log_loss(y_te, y_prob, labels=[0, 1, 2]))
        y_pred = np.argmax(y_prob, axis=1)
        all_accuracy.append(float(np.mean(y_pred == y_te)))
        all_y_probs.append(y_prob)

    ensemble_prob = np.mean(all_y_probs, axis=0)
    brier_ensemble = multiclass_brier(y_te, ensemble_prob)
    ci_lo, ci_hi = bootstrap_ci(y_te, ensemble_prob)

    # Draw-specific metrics (Two-Stage diagnostic)
    draw_pred = np.argmax(ensemble_prob, axis=1) == 1
    draw_real = y_te == 1
    n_draw_pred = int(draw_pred.sum())
    n_draw_correct = int((draw_pred & draw_real).sum())
    draw_acc = round(n_draw_correct / n_draw_pred, 4) if n_draw_pred > 0 else 0.0

    test_dist = df_test["result"].value_counts().sort_index()
    test_pct = (test_dist / len(df_test) * 100).round(1).to_dict()

    return {
        "test": test_name,
        "architecture": "two_stage",
        "universe": classify_test_universe(feature_names),
        "n_features": len(feature_names),
        "features": feature_names,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "date_range": [str(df_sorted["date"].min()), str(df_sorted["date"].max())],
        "split_date": str(df_test["date"].min()),
        "brier_ensemble": round(brier_ensemble, 5),
        "brier_ci95": [round(ci_lo, 5), round(ci_hi, 5)],
        "brier_seed_mean": round(float(np.mean(all_briers)), 5),
        "brier_seed_std": round(float(np.std(all_briers)), 5),
        "brier_seed_range": [round(min(all_briers), 5), round(max(all_briers), 5)],
        "logloss_mean": round(float(np.mean(all_logloss)), 5),
        "accuracy_mean": round(float(np.mean(all_accuracy)), 4),
        "draw_predicted": n_draw_pred,
        "draw_correct": n_draw_correct,
        "draw_accuracy": draw_acc,
        "test_distribution": {
            "H": test_pct.get(0, 0),
            "D": test_pct.get(1, 0),
            "A": test_pct.get(2, 0),
        },
    }


def evaluate_feature_set_residual(df_universe, feature_names, test_name,
                                  lockbox_mode=False):
    """Train & evaluate a residual model (base_margin from market odds).

    Same pattern as evaluate_feature_set but uses train_xgb_residual()
    and reports delta vs market baseline.
    """
    prep = _prepare_dataset(df_universe, feature_names, test_name,
                            lockbox_mode=lockbox_mode)
    if "error" in prep:
        return {"test": test_name, **prep}

    df_train, df_test, df_sorted = prep["df_train"], prep["df_test"], prep["df_sorted"]
    X_tr, y_tr, X_te, y_te = prep["X_tr"], prep["y_tr"], prep["X_te"], prep["y_te"]

    # Compute market logits for train and test
    logits_train = compute_market_logits(df_train)
    logits_test = compute_market_logits(df_test)

    # Market baseline on same test set
    inv_h = 1.0 / df_test["odds_home"].values
    inv_d = 1.0 / df_test["odds_draw"].values
    inv_a = 1.0 / df_test["odds_away"].values
    total = inv_h + inv_d + inv_a
    market_probs = np.column_stack([inv_h / total, inv_d / total, inv_a / total])
    brier_market = multiclass_brier(y_te, market_probs)

    # Multi-seed evaluation with residual model
    all_briers, all_logloss, all_accuracy = [], [], []
    all_y_probs = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 42 + 7
        model = train_xgb_residual(X_tr, y_tr, logits_train, seed=seed)
        y_prob = predict_with_base_margin(model, X_te, logits_test)

        all_briers.append(multiclass_brier(y_te, y_prob))
        all_logloss.append(log_loss(y_te, y_prob, labels=[0, 1, 2]))
        all_accuracy.append(float(np.mean(np.argmax(y_prob, axis=1) == y_te)))
        all_y_probs.append(y_prob)

    # Ensemble
    ensemble_prob = np.mean(all_y_probs, axis=0)
    brier_ensemble = multiclass_brier(y_te, ensemble_prob)
    ci_lo, ci_hi = bootstrap_ci(y_te, ensemble_prob)

    # Paired delta CI: ensemble vs market
    delta = brier_ensemble - brier_market
    delta_ci_lo, delta_ci_hi = bootstrap_paired_delta(
        y_te, ensemble_prob, market_probs)

    # Diagnostic decomposition (FS-01 / FS-02)
    diag = {}
    if _DECOMPOSE:
        decomp_10 = brier_decomposition(y_te, ensemble_prob, n_bins=10)
        decomp_20 = brier_decomposition(y_te, ensemble_prob, n_bins=20)
        ece = multiclass_ece(y_te, ensemble_prob, n_bins=15)
        diag = {
            "brier_decomposition_10": decomp_10,
            "brier_decomposition_20": decomp_20,
            "ece": {k: v for k, v in ece.items() if k != "curve_data"},
            "diagnosis": _diagnose_cal_vs_res(decomp_10, ece),
        }

    result = {
        "test": test_name,
        "universe": "residual",
        "n_features": len(feature_names),
        "features": feature_names,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "split_date": str(df_test["date"].min()),
        "brier_ensemble": round(brier_ensemble, 5),
        "brier_ci95": [round(ci_lo, 5), round(ci_hi, 5)],
        "brier_seed_mean": round(float(np.mean(all_briers)), 5),
        "brier_seed_std": round(float(np.std(all_briers)), 5),
        "brier_market": round(brier_market, 5),
        "delta_vs_market": round(delta, 5),
        "delta_ci95": [round(delta_ci_lo, 5), round(delta_ci_hi, 5)],
        "logloss_mean": round(float(np.mean(all_logloss)), 5),
        "accuracy_mean": round(float(np.mean(all_accuracy)), 4),
    }
    result.update(diag)
    return result


# ─── Walk-Forward Multi-Window (FS-04) ──────────────────────

_WALK_FORWARD = False


def walk_forward_evaluate(df_universe, feature_names, test_name,
                          params=None, test_window_months=6,
                          min_train_months=12, min_test=30):
    """Expanding window walk-forward evaluation (FS-04).

    Instead of a single 80/20 split, uses multiple temporal windows:
      Window 1: Train [min_date → min_date+12m] → Test [next 6m]
      Window 2: Train [min_date → min_date+18m] → Test [next 6m]
      ...
    Expanding window: train grows, test is always ~6 months.

    Args:
        df_universe: Pre-filtered universe DataFrame, sorted by date.
        feature_names: Feature columns for model training.
        test_name: Label for the champion being evaluated.
        params: Optional Optuna best_params. If None, uses PROD_HYPERPARAMS.
        test_window_months: Length of each test window in months (default 6).
        min_train_months: Minimum training period in months (default 12).
        min_test: Minimum test samples per window.

    Returns dict with per-window results + aggregated stats, or None.
    """
    missing = [f for f in feature_names if f not in df_universe.columns]
    if missing:
        return {"status": "ERROR", "error": "missing_features", "detail": missing}

    nan_rows = df_universe[feature_names].isna().any(axis=1).sum()
    if nan_rows > 0:
        return {"status": "ERROR", "error": "nan_in_features", "count": int(nan_rows)}

    if len(df_universe) < 100:
        return {"status": "INSUFFICIENT_DATA", "n": len(df_universe)}

    dates = df_universe["date"]
    min_date = dates.min()
    max_date = dates.max()

    # Generate window boundaries (expanding train, fixed test length)
    windows = []
    current_test_start = min_date + pd.DateOffset(months=min_train_months)
    while current_test_start + pd.DateOffset(months=test_window_months) <= max_date + pd.Timedelta(days=1):
        test_end = current_test_start + pd.DateOffset(months=test_window_months)
        windows.append((min_date, current_test_start, test_end))
        current_test_start = test_end

    if len(windows) < 2:
        return {
            "status": "INSUFFICIENT_WINDOWS",
            "n_windows": len(windows),
            "date_range": [str(min_date.date()), str(max_date.date())],
            "months_span": round((max_date - min_date).days / 30.4, 1),
        }

    results_per_window = []
    for train_start, test_start, test_end in windows:
        df_train = df_universe[(dates >= train_start) & (dates < test_start)]
        df_test = df_universe[(dates >= test_start) & (dates < test_end)]

        if len(df_test) < min_test:
            continue

        X_tr = df_train[feature_names].values.astype(np.float32)
        y_tr = df_train["result"].values.astype(int)
        X_te = df_test[feature_names].values.astype(np.float32)
        y_te = df_test["result"].values.astype(int)

        # Train model (single seed per window for speed)
        if params:
            merged = {**PROD_HYPERPARAMS, **params, "random_state": 42}
            model = xgb.XGBClassifier(**merged)
            sw = np.ones(len(y_tr), dtype=np.float32)
            sw[y_tr == 1] = DRAW_WEIGHT
            model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        else:
            model = train_xgb(X_tr, y_tr, seed=42)

        y_prob = model.predict_proba(X_te)
        brier_val = multiclass_brier(y_te, y_prob)

        # Market baseline on same window (if odds available)
        mkt_brier = None
        delta = None
        if all(c in df_test.columns for c in ("odds_home", "odds_draw", "odds_away")):
            odds_valid = (df_test["odds_home"].notna() &
                          (df_test["odds_home"] > 1.0))
            if odds_valid.sum() >= min_test:
                df_mkt = df_test[odds_valid]
                y_mkt = df_mkt["result"].values.astype(int)
                inv_h = 1.0 / df_mkt["odds_home"].values
                inv_d = 1.0 / df_mkt["odds_draw"].values
                inv_a = 1.0 / df_mkt["odds_away"].values
                total = inv_h + inv_d + inv_a
                mkt_probs = np.column_stack([inv_h/total, inv_d/total, inv_a/total])
                mkt_brier = round(multiclass_brier(y_mkt, mkt_probs), 5)
                # Model brier on same subset for fair comparison
                y_prob_subset = model.predict_proba(
                    df_mkt[feature_names].values.astype(np.float32))
                model_brier_fair = multiclass_brier(y_mkt, y_prob_subset)
                delta = round(model_brier_fair - mkt_brier, 5)

        results_per_window.append({
            "window": "%s \u2192 %s" % (test_start.date(), test_end.date()),
            "n_train": len(df_train),
            "n_test": len(df_test),
            "brier": round(brier_val, 5),
            "market_brier": mkt_brier,
            "delta": delta,
        })

    if len(results_per_window) < 2:
        return {
            "status": "INSUFFICIENT_WINDOWS",
            "n_windows": len(results_per_window),
            "date_range": [str(min_date.date()), str(max_date.date())],
        }

    briers = [r["brier"] for r in results_per_window]
    deltas = [r["delta"] for r in results_per_window if r["delta"] is not None]

    # Stability assessment
    brier_std = float(np.std(briers))
    if brier_std < 0.03:
        stability = "STABLE"
    elif brier_std < 0.05:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"

    # Direction consistency: how many windows agree on sign of delta?
    direction = None
    if deltas:
        n_positive = sum(1 for d in deltas if d > 0)
        n_negative = sum(1 for d in deltas if d < 0)
        direction = {
            "model_worse": n_positive,
            "model_better": n_negative,
            "consistent": n_positive == 0 or n_negative == 0,
        }

    return {
        "status": "OK",
        "test": test_name,
        "n_windows": len(results_per_window),
        "windows": results_per_window,
        "brier_mean": round(float(np.mean(briers)), 5),
        "brier_std": round(brier_std, 5),
        "brier_min": round(min(briers), 5),
        "brier_max": round(max(briers), 5),
        "delta_mean": round(float(np.mean(deltas)), 5) if deltas else None,
        "delta_std": round(float(np.std(deltas)), 5) if deltas else None,
        "stability": stability,
        "direction": direction,
    }


# ─── Calibration Test (FS-03) ────────────────────────────────

_CALIBRATE = False


def calibration_test(df_universe, feature_names, test_name,
                     params=None, method="isotonic"):
    """Evaluate model with optional post-hoc calibration (FS-03).

    Compares 'none' (raw XGBoost) vs 'isotonic' (or 'platt') calibration.
    CRITICAL: Calibration uses a held-out slice of TRAIN data (80/20 inner
    split), NEVER touches the test set. This prevents information leakage.

    Args:
        df_universe: Pre-filtered universe DataFrame.
        feature_names: Feature columns.
        test_name: Label for the test.
        params: Optional Optuna hyperparams.
        method: 'isotonic' or 'platt' (sigmoid).

    Returns dict comparing none vs calibrated model on same test set.
    """
    from sklearn.calibration import CalibratedClassifierCV

    missing = [f for f in feature_names if f not in df_universe.columns]
    if missing:
        return {"status": "ERROR", "error": "missing_features", "detail": missing}

    if len(df_universe) < 150:
        return {"status": "INSUFFICIENT_DATA", "n": len(df_universe)}

    # Standard 80/20 split
    split_idx = int(len(df_universe) * (1 - TEST_FRACTION))
    df_train = df_universe.iloc[:split_idx]
    df_test = df_universe.iloc[split_idx:]

    if len(df_test) < 50:
        return {"status": "INSUFFICIENT_DATA", "n_test": len(df_test)}

    X_te = df_test[feature_names].values.astype(np.float32)
    y_te = df_test["result"].values.astype(int)

    # Inner split of train: 80% for model, 20% for calibration
    cal_split = int(len(df_train) * 0.80)
    df_tr_inner = df_train.iloc[:cal_split]
    df_cal = df_train.iloc[cal_split:]

    if len(df_cal) < 30:
        return {"status": "INSUFFICIENT_CAL_DATA", "n_cal": len(df_cal)}

    X_tr_inner = df_tr_inner[feature_names].values.astype(np.float32)
    y_tr_inner = df_tr_inner["result"].values.astype(int)
    X_cal = df_cal[feature_names].values.astype(np.float32)
    y_cal = df_cal["result"].values.astype(int)
    X_tr_full = df_train[feature_names].values.astype(np.float32)
    y_tr_full = df_train["result"].values.astype(int)

    # === None (raw XGBoost, trained on full train) ===
    if params:
        merged = {**PROD_HYPERPARAMS, **params, "random_state": 42}
        model_none = xgb.XGBClassifier(**merged)
        sw = np.ones(len(y_tr_full), dtype=np.float32)
        sw[y_tr_full == 1] = DRAW_WEIGHT
        model_none.fit(X_tr_full, y_tr_full, sample_weight=sw, verbose=False)
    else:
        model_none = train_xgb(X_tr_full, y_tr_full, seed=42)
    y_prob_none = model_none.predict_proba(X_te)
    brier_none = multiclass_brier(y_te, y_prob_none)
    ece_none = multiclass_ece(y_te, y_prob_none, n_bins=15)

    # === Calibrated (trained on inner, calibrated on cal, predict on test) ===
    if params:
        merged_inner = {**PROD_HYPERPARAMS, **params, "random_state": 42}
        model_inner = xgb.XGBClassifier(**merged_inner)
        sw_inner = np.ones(len(y_tr_inner), dtype=np.float32)
        sw_inner[y_tr_inner == 1] = DRAW_WEIGHT
        model_inner.fit(X_tr_inner, y_tr_inner, sample_weight=sw_inner,
                        verbose=False)
    else:
        model_inner = train_xgb(X_tr_inner, y_tr_inner, seed=42)

    cal_model = CalibratedClassifierCV(model_inner, method=method, cv="prefit")
    cal_model.fit(X_cal, y_cal)
    y_prob_cal = cal_model.predict_proba(X_te)
    brier_cal = multiclass_brier(y_te, y_prob_cal)
    ece_cal = multiclass_ece(y_te, y_prob_cal, n_bins=15)

    # Paired delta
    ci_lo, ci_hi = bootstrap_paired_delta(y_te, y_prob_cal, y_prob_none)

    # Diagnosis
    ece_improved = ece_cal["ece_avg"] < ece_none["ece_avg"]
    brier_improved = brier_cal < brier_none

    if ece_improved and not brier_improved:
        conclusion = "CALIBRATION_ISSUE_CONFIRMED"
    elif brier_improved and ece_improved:
        conclusion = "CALIBRATION_HELPS"
    elif not ece_improved and not brier_improved:
        conclusion = "RESOLUTION_ISSUE_CONFIRMED"
    else:
        conclusion = "MIXED"

    return {
        "status": "OK",
        "test": test_name,
        "method": method,
        "n_train_inner": len(df_tr_inner),
        "n_cal": len(df_cal),
        "n_test": len(df_test),
        "none": {
            "brier": round(brier_none, 5),
            "ece_avg": ece_none["ece_avg"],
            "ece_home": ece_none["ece_home"],
            "ece_draw": ece_none["ece_draw"],
            "ece_away": ece_none["ece_away"],
        },
        "calibrated": {
            "brier": round(brier_cal, 5),
            "ece_avg": ece_cal["ece_avg"],
            "ece_home": ece_cal["ece_home"],
            "ece_draw": ece_cal["ece_draw"],
            "ece_away": ece_cal["ece_away"],
        },
        "delta_brier": round(brier_cal - brier_none, 5),
        "delta_ci95": [round(ci_lo, 5), round(ci_hi, 5)],
        "ece_improved": ece_improved,
        "brier_improved": brier_improved,
        "conclusion": conclusion,
    }


# ─── Opening vs Closing Test (FS-07) ────────────────────────

_OPENING_TEST = False


def opening_vs_closing_test(df_universe, feature_names, test_name,
                            params=None, lockbox_mode=False):
    """Compare model vs TRUE opening odds AND closing odds (FS-07).

    CRITICAL: Filters by opening_odds_kind to separate real openings from
    closing proxies. Only rows where opening != closing AND opening_odds_kind
    is a true opening type are included.

    Args:
        df_universe: Pre-filtered universe DataFrame with odds columns.
        feature_names: Champion feature columns.
        test_name: Label for the champion.
        params: Optional Optuna hyperparams.
        lockbox_mode: If True, use 70/15/15 split.

    Returns dict with model vs opening, model vs closing, line movement value.
    """
    from app.ml.devig import devig_proportional

    # Check required columns exist
    required = ["odds_home_close", "odds_draw_close", "odds_away_close",
                "odds_home_open", "odds_draw_open", "odds_away_open",
                "opening_odds_kind"]
    missing_cols = [c for c in required if c not in df_universe.columns]
    if missing_cols:
        return {
            "status": "MISSING_COLUMNS",
            "missing": missing_cols,
            "hint": "Re-extract with --extract to include opening_odds columns",
        }

    # Step 0: Report opening_odds_kind distribution
    kind_dist = df_universe["opening_odds_kind"].value_counts(dropna=False).to_dict()

    # Split train/test
    if len(df_universe) < 100:
        return {"status": "INSUFFICIENT_DATA", "n": len(df_universe)}

    if lockbox_mode:
        n = len(df_universe)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        df_train = df_universe.iloc[:train_end]
        df_test = df_universe.iloc[train_end:val_end]
    else:
        split_idx = int(len(df_universe) * (1 - TEST_FRACTION))
        df_train = df_universe.iloc[:split_idx]
        df_test = df_universe.iloc[split_idx:]

    if len(df_test) < 30:
        return {"status": "INSUFFICIENT_DATA", "n_test": len(df_test)}

    # TRUE OPENINGS: only rows where opening_odds_kind is a real opening
    true_opening_kinds = {"opening", "true_opening", "earliest_available"}

    is_true_opening = df_test["opening_odds_kind"].isin(true_opening_kinds)
    has_closing = (df_test["odds_home_close"].notna() &
                   (df_test["odds_home_close"] > 1.0))
    has_true_opening = (df_test["odds_home_open"].notna() &
                        (df_test["odds_home_open"] > 1.0) &
                        is_true_opening)

    # Odds must differ (if identical, no line movement information)
    odds_differ = (
        (df_test["odds_home_open"] != df_test["odds_home_close"]) |
        (df_test["odds_draw_open"] != df_test["odds_draw_close"]) |
        (df_test["odds_away_open"] != df_test["odds_away_close"])
    )

    has_both = has_closing & has_true_opening & odds_differ
    n_both = int(has_both.sum())

    if n_both < 50:
        return {
            "status": "INSUFFICIENT_DATA",
            "n_both": n_both,
            "opening_odds_kind_distribution": kind_dist,
            "reason": "Only %d matches with true opening + closing + different" % n_both,
        }

    df_both = df_test[has_both]
    y_true = df_both["result"].values.astype(int)

    # De-vig closing and opening odds
    oh_c = df_both["odds_home_close"].values
    od_c = df_both["odds_draw_close"].values
    oa_c = df_both["odds_away_close"].values
    closing_probs = np.array([devig_proportional(h, d, a)
                              for h, d, a in zip(oh_c, od_c, oa_c)])

    oh_o = df_both["odds_home_open"].values
    od_o = df_both["odds_draw_open"].values
    oa_o = df_both["odds_away_open"].values
    opening_probs = np.array([devig_proportional(h, d, a)
                              for h, d, a in zip(oh_o, od_o, oa_o)])

    # Train model on full training set
    fmissing = [f for f in feature_names if f not in df_train.columns]
    if fmissing:
        return {"status": "ERROR", "error": "missing_features", "detail": fmissing}

    X_tr = df_train[feature_names].values.astype(np.float32)
    y_tr = df_train["result"].values.astype(int)

    if params:
        merged = {**PROD_HYPERPARAMS, **params, "random_state": 42}
        model = xgb.XGBClassifier(**merged)
        sw = np.ones(len(y_tr), dtype=np.float32)
        sw[y_tr == 1] = DRAW_WEIGHT
        model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    else:
        model = train_xgb(X_tr, y_tr, seed=42)

    X_both = df_both[feature_names].values.astype(np.float32)
    y_prob_model = model.predict_proba(X_both)

    # Brier scores
    brier_model = multiclass_brier(y_true, y_prob_model)
    brier_closing = multiclass_brier(y_true, closing_probs)
    brier_opening = multiclass_brier(y_true, opening_probs)

    # Paired delta CIs
    ci_vs_closing = bootstrap_paired_delta(y_true, y_prob_model, closing_probs)
    ci_vs_opening = bootstrap_paired_delta(y_true, y_prob_model, opening_probs)

    return {
        "status": "OK",
        "test": test_name,
        "n_both": n_both,
        "n_test_total": len(df_test),
        "opening_odds_kind_distribution": kind_dist,
        "brier_model": round(brier_model, 5),
        "brier_closing": round(brier_closing, 5),
        "brier_opening": round(brier_opening, 5),
        "delta_vs_closing": round(brier_model - brier_closing, 5),
        "delta_vs_opening": round(brier_model - brier_opening, 5),
        "line_movement_value": round(brier_opening - brier_closing, 5),
        "ci_vs_closing": [round(ci_vs_closing[0], 5), round(ci_vs_closing[1], 5)],
        "ci_vs_opening": [round(ci_vs_opening[0], 5), round(ci_vs_opening[1], 5)],
    }


# ─── Market Baseline ─────────────────────────────────────────

def devig_sensitivity_test(df_odds_universe, lockbox_mode=False):
    """Compare market Brier under different de-vig methods (FS-06).

    Tests proportional, power, and Shin methods on the same odds universe.
    Returns a dict with per-method Brier and deltas.
    """
    from app.ml.devig import devig_proportional, devig_power, devig_shin

    if len(df_odds_universe) < 50:
        return None

    if lockbox_mode:
        n = len(df_odds_universe)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        df_test = df_odds_universe.iloc[train_end:val_end]
    else:
        split_idx = int(len(df_odds_universe) * (1 - TEST_FRACTION))
        df_test = df_odds_universe.iloc[split_idx:]

    if len(df_test) < 30:
        return None

    y_true = df_test["result"].values.astype(int)
    oh = df_test["odds_home"].values
    od = df_test["odds_draw"].values
    oa = df_test["odds_away"].values

    methods = {
        "proportional": devig_proportional,
        "power": devig_power,
        "shin": devig_shin,
    }

    results = {}
    for name, fn in methods.items():
        probs = np.array([fn(h, d, a) for h, d, a in zip(oh, od, oa)])
        brier = multiclass_brier(y_true, probs)
        ci_lo, ci_hi = bootstrap_ci(y_true, probs)
        results[name] = {
            "brier": round(brier, 6),
            "ci95": [round(ci_lo, 5), round(ci_hi, 5)],
        }

    # Deltas (relative to proportional baseline)
    base = results["proportional"]["brier"]
    for name in ("power", "shin"):
        results[name]["delta_vs_prop"] = round(
            results[name]["brier"] - base, 6)

    results["n_test"] = len(df_test)
    return results


def market_brier(df_odds_universe: pd.DataFrame,
                 lockbox_mode: bool = False) -> Optional[dict]:
    """Compute Brier from devigged market odds as probabilistic baseline.

    Receives the odds universe (pre-filtered) so the split is identical
    to all other tests in the same universe.

    In lockbox mode uses 70/15/15 split (evaluates on val slice, aligned
    with model tests). Standard mode uses 80/20.
    """
    if len(df_odds_universe) < 50:
        return None

    if lockbox_mode:
        n = len(df_odds_universe)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        df_test = df_odds_universe.iloc[train_end:val_end]
    else:
        split_idx = int(len(df_odds_universe) * (1 - TEST_FRACTION))
        df_test = df_odds_universe.iloc[split_idx:]

    if len(df_test) < 30:
        return None

    # De-vig: normalize implied probs to sum=1
    inv_h = 1.0 / df_test["odds_home"].values
    inv_d = 1.0 / df_test["odds_draw"].values
    inv_a = 1.0 / df_test["odds_away"].values
    total = inv_h + inv_d + inv_a

    market_probs = np.column_stack([inv_h / total, inv_d / total, inv_a / total])
    y_true = df_test["result"].values.astype(int)

    brier = multiclass_brier(y_true, market_probs)
    ci_lo, ci_hi = bootstrap_ci(y_true, market_probs)

    # Diagnostic decomposition (FS-01 / FS-02) — market baseline
    diag = {}
    if _DECOMPOSE:
        decomp_10 = brier_decomposition(y_true, market_probs, n_bins=10)
        decomp_20 = brier_decomposition(y_true, market_probs, n_bins=20)
        ece = multiclass_ece(y_true, market_probs, n_bins=15)
        diag = {
            "brier_decomposition_10": decomp_10,
            "brier_decomposition_20": decomp_20,
            "ece": {k: v for k, v in ece.items() if k != "curve_data"},
        }

    result = {
        "test": "MKT_market",
        "universe": "odds",
        "n_test": len(df_test),
        "brier_ensemble": round(brier, 5),
        "brier_ci95": [round(ci_lo, 5), round(ci_hi, 5)],
    }
    result.update(diag)
    return result


def fair_model_vs_market(df_odds_universe: pd.DataFrame, feature_names: list,
                         test_name: str,
                         params: dict | None = None,
                         lockbox_mode: bool = False) -> Optional[dict]:
    """Compare model vs market on the SAME test subset (ABE P1-C).

    Both are evaluated on the odds universe test set, so N is identical.
    Now includes paired bootstrap CI for the delta.

    In lockbox mode uses 70/15/15 split (evaluates on val slice, aligned
    with model tests). Standard mode uses 80/20.

    Args:
        df_odds_universe: Pre-filtered odds universe DataFrame.
        params: Optional XGBoost hyperparams (e.g. from Optuna). If None,
                uses production hyperparams via train_xgb().
        lockbox_mode: If True, use 70/15/15 split (train/val/lockbox).
    """
    # Verify model features exist
    missing = [f for f in feature_names if f not in df_odds_universe.columns]
    if missing:
        return None

    # Check for NaN in model features within odds universe
    nan_rows = df_odds_universe[feature_names].isna().any(axis=1).sum()
    if nan_rows > 0:
        return None

    if len(df_odds_universe) < 100:
        return None

    if lockbox_mode:
        n = len(df_odds_universe)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        df_train = df_odds_universe.iloc[:train_end]
        df_test = df_odds_universe.iloc[train_end:val_end]
    else:
        split_idx = int(len(df_odds_universe) * (1 - TEST_FRACTION))
        df_train = df_odds_universe.iloc[:split_idx]
        df_test = df_odds_universe.iloc[split_idx:]

    if len(df_test) < 50:
        return None

    X_tr = df_train[feature_names].values.astype(np.float32)
    y_tr = df_train["result"].values.astype(int)
    X_te = df_test[feature_names].values.astype(np.float32)
    y_te = df_test["result"].values.astype(int)

    # Model prediction — use tuned params if provided, else production defaults
    if params:
        merged = {**PROD_HYPERPARAMS, **params, "random_state": 42}
        model = xgb.XGBClassifier(**merged)
        sample_weight = np.ones(len(y_tr), dtype=np.float32)
        sample_weight[y_tr == 1] = DRAW_WEIGHT
        model.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)
    else:
        model = train_xgb(X_tr, y_tr, seed=42)
    y_prob_model = model.predict_proba(X_te)
    model_brier = multiclass_brier(y_te, y_prob_model)

    # Market prediction (de-vig)
    inv_h = 1.0 / df_test["odds_home"].values
    inv_d = 1.0 / df_test["odds_draw"].values
    inv_a = 1.0 / df_test["odds_away"].values
    total = inv_h + inv_d + inv_a
    market_probs = np.column_stack([inv_h / total, inv_d / total, inv_a / total])
    market_brier_val = multiclass_brier(y_te, market_probs)

    # Fix 2: paired bootstrap CI for delta
    delta_ci_lo, delta_ci_hi = bootstrap_paired_delta(y_te, y_prob_model, market_probs)

    # FS-05: alignment check — model and market use identical test matches
    alignment = {
        "n_model": len(y_te),
        "n_market": len(market_probs),
        "match_ids_n": len(df_test),
        "date_min": str(df_test["date"].min()),
        "date_max": str(df_test["date"].max()),
    }
    if "match_id" in df_test.columns:
        alignment["test_match_ids"] = sorted(df_test["match_id"].tolist())

    return {
        "test": test_name,
        "n_test_fair": len(df_test),
        "model_brier_fair": round(model_brier, 5),
        "market_brier_fair": round(market_brier_val, 5),
        "delta": round(model_brier - market_brier_val, 5),
        "delta_ci95": [round(delta_ci_lo, 5), round(delta_ci_hi, 5)],
        "market_wins": model_brier > market_brier_val,
        "alignment_check": alignment,
    }


# ─── Run All Tests for One League ────────────────────────────

def run_league_tests(df: pd.DataFrame, league_id: int,
                     lockbox_mode: bool = False,
                     run_residual: bool = False,
                     run_two_stage: bool = False) -> dict:
    """Run all feature set tests for one league using fixed universes."""
    league_name = LEAGUE_NAMES.get(league_id, f"league_{league_id}")

    print(f"\n{'=' * 70}")
    print(f"  FEATURE LAB: {league_name} (id={league_id})")
    print(f"  Matches: {len(df)} | Seeds: {N_SEEDS} | Bootstrap: {N_BOOTSTRAP}")
    if lockbox_mode:
        print(f"  Mode: LOCKBOX (70/15/15)")
    print(f"{'=' * 70}")

    # Fix 1: compute universes once (include Two-Stage tests for proper universe coverage)
    all_tests_dict = dict(TESTS)
    if run_two_stage:
        all_tests_dict.update(TWO_STAGE_TESTS)
    universes = compute_universes(df, all_tests_dict)

    # Run anchor tests (A0, A1) in each non-empty universe for intra-universe deltas
    anchor_tests = {"A0_baseline_17": TESTS["A0_baseline_17"],
                    "A1_only_elo_k32": TESTS["A1_only_elo_k32"]}
    anchors_by_universe = {}
    for uid, udf in universes.items():
        if udf.empty:
            continue
        for anchor_name, anchor_feats in anchor_tests.items():
            key = f"{anchor_name}@{uid}"
            res = evaluate_feature_set(udf, anchor_feats, key,
                                       lockbox_mode=lockbox_mode)
            if res and "error" not in res:
                anchors_by_universe[key] = res

    results = []
    for test_name, features in TESTS.items():
        uid = classify_test_universe(features)
        udf = universes.get(uid, pd.DataFrame())

        print(f"\n  [{test_name}] {len(features)} features (universe={uid})...")
        if udf.empty:
            res = {"test": test_name, "universe": uid,
                   "error": f"empty_universe:{uid}"}
            print(f"    SKIP: {res['error']}")
            results.append(res)
            continue

        res = evaluate_feature_set(udf, features, test_name,
                                   lockbox_mode=lockbox_mode)
        if res and "error" not in res:
            print(f"    Brier: {res['brier_ensemble']:.5f} "
                  f"(seeds: {res['brier_seed_mean']:.5f} ± {res['brier_seed_std']:.5f}) "
                  f"CI95[{res['brier_ci95'][0]:.5f}, {res['brier_ci95'][1]:.5f}] "
                  f"Acc: {res['accuracy_mean']:.3f} "
                  f"(N_train={res['n_train']}, N_test={res['n_test']})")
        elif res:
            print(f"    SKIP: {res['error']}")
        results.append(res)

    # ─── Section W: Two-Stage architecture tests ──────────────
    two_stage_results = []
    if run_two_stage:
        print(f"\n  {'─' * 60}")
        print(f"  SECTION W: Two-Stage Architecture (Shadow B)")
        print(f"  {'─' * 60}")

        for test_name, features in TWO_STAGE_TESTS.items():
            uid = classify_test_universe(features)
            udf = universes.get(uid, pd.DataFrame())

            print(f"\n  [{test_name}] {len(features)} features (universe={uid})...")
            if udf.empty:
                res = {"test": test_name, "universe": uid,
                       "error": "empty_universe:%s" % uid}
                print(f"    SKIP: {res['error']}")
                two_stage_results.append(res)
                continue

            res = evaluate_two_stage(udf, features, test_name,
                                     lockbox_mode=lockbox_mode)
            if res and "error" not in res:
                print(f"    Brier: {res['brier_ensemble']:.5f} "
                      f"(seeds: {res['brier_seed_mean']:.5f} "
                      f"\u00b1 {res['brier_seed_std']:.5f}) "
                      f"CI95[{res['brier_ci95'][0]:.5f}, {res['brier_ci95'][1]:.5f}] "
                      f"Acc: {res['accuracy_mean']:.3f} "
                      f"Draw: {res['draw_predicted']}/{res['draw_correct']} "
                      f"({res['draw_accuracy']:.3f}) "
                      f"(N_train={res['n_train']}, N_test={res['n_test']})")
            elif res:
                print(f"    SKIP: {res['error']}")
            two_stage_results.append(res)

        # Two-Stage vs One-Stage head-to-head summary
        valid_ts = [r for r in two_stage_results if r and "error" not in r]
        if valid_ts:
            print(f"\n  {'─' * 70}")
            print(f"  TWO-STAGE vs ONE-STAGE HEAD-TO-HEAD")
            delta_sym = "\u0394"
            print(f"  {'Test':<28} {'TS Brier':>10} {'OS Brier':>10} "
                  f"{delta_sym:>8} {'Draw Acc':>9}")
            print(f"  {'─' * 28} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 9}")
            # Map W tests to their One-Stage equivalents
            h2h_map = {
                "W0_ts_baseline": "A0_baseline_17",
                "W2_ts_baseline_elo": "H1_defense_elo",
                "W5_ts_full_odds": "J2_full_odds",
                "W10_ts_defense_elo": "H1_defense_elo",
            }
            for ts_r in valid_ts:
                ts_name = ts_r["test"]
                os_name = h2h_map.get(ts_name)
                os_brier = None
                if os_name:
                    for r in results:
                        if r and r.get("test") == os_name and "error" not in r:
                            os_brier = r["brier_ensemble"]
                            break
                delta_str = ""
                os_str = "N/A"
                if os_brier is not None:
                    os_str = "%.5f" % os_brier
                    delta = ts_r["brier_ensemble"] - os_brier
                    delta_str = "%+.5f" % delta
                print(f"  {ts_name:<28} {ts_r['brier_ensemble']:>10.5f} "
                      f"{os_str:>10} {delta_str:>8} "
                      f"{ts_r['draw_accuracy']:>9.3f}")

    # Market baseline (uses odds universe, lockbox-aware split)
    print(f"\n  [MKT_market] De-vigged odds baseline...")
    mkt = market_brier(universes["odds"], lockbox_mode=lockbox_mode)
    if mkt:
        print(f"    Brier: {mkt['brier_ensemble']:.5f} "
              f"CI95[{mkt['brier_ci95'][0]:.5f}, {mkt['brier_ci95'][1]:.5f}] "
              f"(N_test={mkt['n_test']})")
    else:
        print(f"    SKIP: insufficient odds data")

    # Fair model-vs-market comparison with paired delta CI (lockbox-aware)
    fair_comparisons = {}
    if mkt:
        # Best result per universe that has odds
        for uid in ("odds", "odds_xg"):
            valid = [r for r in results
                     if r and "error" not in r and r.get("universe") == uid]
            if not valid:
                continue
            best = min(valid, key=lambda r: r["brier_ensemble"])
            fair = fair_model_vs_market(universes[uid], best["features"],
                                        best["test"],
                                        lockbox_mode=lockbox_mode)
            if fair:
                fair_comparisons[best["test"]] = fair
                delta_ci = fair.get("delta_ci95", [None, None])
                print(f"\n  [FAIR] {best['test']} vs market (N={fair['n_test_fair']}): "
                      f"model={fair['model_brier_fair']:.5f} "
                      f"market={fair['market_brier_fair']:.5f} "
                      f"Δ={fair['delta']:+.5f} "
                      f"CI95[{delta_ci[0]:+.5f}, {delta_ci[1]:+.5f}]")

    # FS-05: Cross-check alignment between FAIR, market, and model test sets
    alignment_report = []
    if mkt and fair_comparisons:
        for test_name, fair in fair_comparisons.items():
            ac = fair.get("alignment_check", {})
            fair_ids = set(ac.get("test_match_ids", []))
            n_fair = ac.get("n_model", 0)
            n_market = mkt.get("n_test", 0)

            # FAIR must have model_N == market_N within same universe
            ids_aligned = n_fair == ac.get("n_market", -1)
            check = {
                "test": test_name,
                "n_fair_model": n_fair,
                "n_fair_market": ac.get("n_market", 0),
                "n_market_baseline": n_market,
                "model_market_aligned": ids_aligned,
                "date_min": ac.get("date_min"),
                "date_max": ac.get("date_max"),
            }
            if not ids_aligned:
                print(f"  [FS-05 WARNING] Alignment mismatch for {test_name}: "
                      f"model={n_fair}, market={ac.get('n_market', '?')}")
            alignment_report.append(check)

    # Lockbox evaluation (Fix 4) — champion selected PER UNIVERSE
    lockbox_results = []
    if lockbox_mode:
        for uid, udf in universes.items():
            if udf.empty:
                continue
            # Find best test within THIS universe only
            valid_in_uid = [r for r in results
                           if r and "error" not in r
                           and r.get("universe") == uid]
            if not valid_in_uid:
                continue
            champion = min(valid_in_uid, key=lambda r: r["brier_ensemble"])
            prep = _prepare_dataset(udf, champion["features"],
                                    champion["test"], lockbox_mode=True)
            if "error" in prep:
                continue
            # Train on train+val, evaluate on lockbox (one-shot)
            X_tv = np.vstack([prep["X_tr"], prep["X_te"]])
            y_tv = np.concatenate([prep["y_tr"], prep["y_te"]])
            model = train_xgb(X_tv, y_tv, seed=42)
            y_prob_lock = model.predict_proba(prep["X_lock"])
            lock_brier = multiclass_brier(prep["y_lock"], y_prob_lock)
            lock_ci = bootstrap_ci(prep["y_lock"], y_prob_lock)
            lock_entry = {
                "champion": champion["test"],
                "universe": uid,
                "n_lockbox": len(prep["df_lockbox"]),
                "lockbox_brier": round(lock_brier, 5),
                "lockbox_ci95": [round(lock_ci[0], 5), round(lock_ci[1], 5)],
                "split_dates": {
                    "train_end": str(prep["df_train"]["date"].max()),
                    "val_end": str(prep["df_test"]["date"].max()),
                    "lockbox_start": str(prep["df_lockbox"]["date"].min()),
                },
            }
            lockbox_results.append(lock_entry)
            print(f"\n  [LOCKBOX] {champion['test']} (universe={uid}): "
                  f"Brier={lock_brier:.5f} "
                  f"CI95[{lock_ci[0]:.5f}, {lock_ci[1]:.5f}] "
                  f"(N={len(prep['df_lockbox'])})")

    # ─── Section R: Market Residual tests ────────────────────
    residual_results = []
    if run_residual:
        print(f"\n  {'─' * 60}")
        print(f"  SECTION R: Market Residual (correction over market prior)")
        print(f"  {'─' * 60}")

        for test_name, features in RESIDUAL_TESTS.items():
            # Determine universe: needs odds always, check if also needs xG
            feats_set = set(features)
            needs_xg = bool(feats_set & _XG_SET)
            uid = "odds_xg" if needs_xg else "odds"
            udf = universes.get(uid, pd.DataFrame())

            print(f"\n  [{test_name}] {len(features)} features (universe={uid})...")
            if udf.empty:
                res = {"test": test_name, "universe": uid,
                       "error": f"empty_universe:{uid}"}
                print(f"    SKIP: {res['error']}")
                residual_results.append(res)
                continue

            res = evaluate_feature_set_residual(udf, features, test_name,
                                                lockbox_mode=lockbox_mode)
            if res and "error" not in res:
                print(f"    Brier: {res['brier_ensemble']:.5f} "
                      f"(seeds: {res['brier_seed_mean']:.5f} ± {res['brier_seed_std']:.5f}) "
                      f"CI95[{res['brier_ci95'][0]:.5f}, {res['brier_ci95'][1]:.5f}] "
                      f"Acc: {res['accuracy_mean']:.3f} "
                      f"(N_train={res['n_train']}, N_test={res['n_test']})")
                print(f"    vs Market: {res['brier_market']:.5f} | "
                      f"Δ={res['delta_vs_market']:+.5f} "
                      f"CI95[{res['delta_ci95'][0]:+.5f}, {res['delta_ci95'][1]:+.5f}]")
            elif res:
                print(f"    SKIP: {res['error']}")
            residual_results.append(res)

        # Summary table
        valid_r = [r for r in residual_results if r and "error" not in r]
        if valid_r:
            print(f"\n  {'─' * 70}")
            print(f"  SECTION R SUMMARY — Market Residual vs Market Baseline")
            print(f"  {'─' * 70}")
            print(f"  {'Test':<28} {'Brier_Res':>10} {'Brier_Mkt':>10} "
                  f"{'Delta':>8} {'CI95':>22}")
            for r in valid_r:
                dci = r["delta_ci95"]
                sig = "*" if dci[1] < 0 else (" " if dci[0] > 0 else " ")
                print(f"  {r['test']:<28} {r['brier_ensemble']:>10.5f} "
                      f"{r['brier_market']:>10.5f} "
                      f"{r['delta_vs_market']:>+8.5f}{sig} "
                      f"[{dci[0]:+.5f}, {dci[1]:+.5f}]")

    # ─── Devig sensitivity (FS-06) ────────────────────────────
    devig_sensitivity = None
    if _DEVIG_SENSITIVITY and universes.get("odds") is not None:
        devig_sensitivity = devig_sensitivity_test(
            universes["odds"], lockbox_mode=lockbox_mode)
        if devig_sensitivity:
            print(f"\n  {'─' * 70}")
            print(f"  DEVIG SENSITIVITY — Market Brier by de-vig method")
            print(f"  {'─' * 70}")
            for method in ("proportional", "power", "shin"):
                m = devig_sensitivity[method]
                delta_str = ""
                if "delta_vs_prop" in m:
                    delta_str = f"  Δ_vs_prop={m['delta_vs_prop']:+.6f}"
                print(f"  {method:<15} Brier={m['brier']:.6f}  "
                      f"CI95[{m['ci95'][0]:.5f}, {m['ci95'][1]:.5f}]"
                      f"{delta_str}")

    # ─── Walk-forward multi-window (FS-04) ────────────────────
    walk_forward_results = []
    if _WALK_FORWARD:
        print(f"\n  {'─' * 70}")
        print(f"  WALK-FORWARD — Expanding window evaluation")
        print(f"  {'─' * 70}")

        # Collect candidates: best per universe + fixed baseline
        wf_candidates = []

        # Fixed baseline (base universe)
        wf_candidates.append({
            "test": "FIXED_baseline",
            "features": BASELINE_FEATURES,
            "universe": "base",
            "params": None,
        })

        # Best test per universe (from standard results)
        for uid in universes:
            valid_in_uid = [r for r in results
                           if r and "error" not in r
                           and r.get("universe") == uid]
            if not valid_in_uid:
                continue
            best = min(valid_in_uid, key=lambda r: r["brier_ensemble"])
            # Skip if same as fixed baseline
            if best["test"] == "FIXED_baseline" or best["test"].startswith("A0_"):
                continue
            wf_candidates.append({
                "test": best["test"],
                "features": best["features"],
                "universe": uid,
                "params": best.get("optuna_params"),
            })

        for cand in wf_candidates:
            uid = cand["universe"]
            udf = universes.get(uid, pd.DataFrame())
            if udf.empty:
                continue
            print(f"\n  [WF] {cand['test']} (universe={uid})...")
            wf = walk_forward_evaluate(
                udf, cand["features"], cand["test"],
                params=cand["params"])

            if wf and wf.get("status") == "OK":
                wf["universe"] = uid
                walk_forward_results.append(wf)
                # Print per-window results
                for w in wf["windows"]:
                    delta_str = f"  Δ={w['delta']:+.5f}" if w["delta"] is not None else ""
                    mkt_str = f"  mkt={w['market_brier']:.5f}" if w["market_brier"] is not None else ""
                    print(f"    {w['window']}  N={w['n_test']:>4}  "
                          f"Brier={w['brier']:.5f}{mkt_str}{delta_str}")
                # Aggregate
                d = wf.get("direction", {})
                dir_str = ""
                if d:
                    dir_str = (f"  direction: model_worse={d['model_worse']}"
                               f" model_better={d['model_better']}")
                print(f"    SUMMARY: mean={wf['brier_mean']:.5f} "
                      f"std={wf['brier_std']:.5f} [{wf['stability']}] "
                      f"n_windows={wf['n_windows']}"
                      f"{dir_str}")
                if wf.get("delta_mean") is not None:
                    print(f"    Δ_mean={wf['delta_mean']:+.5f} "
                          f"Δ_std={wf['delta_std']:.5f}")
            elif wf:
                print(f"    {wf.get('status', 'ERROR')}: "
                      f"{wf.get('error', wf.get('n_windows', ''))}")

    # ─── Opening vs Closing test (FS-07) ──────────────────────
    opening_test_result = None
    if _OPENING_TEST:
        print(f"\n  {'─' * 70}")
        print(f"  OPENING vs CLOSING — Commercial edge test")
        print(f"  {'─' * 70}")

        # Use best test from odds universe
        valid_odds = [r for r in results
                      if r and "error" not in r
                      and r.get("universe") in ("odds", "odds_xg")]
        if valid_odds:
            best = min(valid_odds, key=lambda r: r["brier_ensemble"])
            uid = best.get("universe", "odds")
            udf = universes.get(uid, pd.DataFrame())
            if not udf.empty:
                print(f"  Champion: {best['test']} (universe={uid})")
                opening_test_result = opening_vs_closing_test(
                    udf, best["features"], best["test"],
                    params=best.get("optuna_params"),
                    lockbox_mode=lockbox_mode)

                if opening_test_result.get("status") == "OK":
                    ot = opening_test_result
                    print(f"  N_both={ot['n_both']} (of {ot['n_test_total']} test)")
                    print(f"  Brier model:   {ot['brier_model']:.5f}")
                    print(f"  Brier closing: {ot['brier_closing']:.5f}")
                    print(f"  Brier opening: {ot['brier_opening']:.5f}")
                    print(f"  Δ_vs_closing:  {ot['delta_vs_closing']:+.5f}  "
                          f"CI95[{ot['ci_vs_closing'][0]:+.5f}, "
                          f"{ot['ci_vs_closing'][1]:+.5f}]")
                    print(f"  Δ_vs_opening:  {ot['delta_vs_opening']:+.5f}  "
                          f"CI95[{ot['ci_vs_opening'][0]:+.5f}, "
                          f"{ot['ci_vs_opening'][1]:+.5f}]")
                    print(f"  Line movement: {ot['line_movement_value']:+.5f} "
                          f"(opening - closing)")
                    if ot["delta_vs_opening"] < 0 and ot["ci_vs_opening"][1] < 0:
                        print(f"  >>> COMMERCIAL EDGE CANDIDATE (model beats opening)")
                else:
                    print(f"  {opening_test_result.get('status')}: "
                          f"{opening_test_result.get('reason', opening_test_result.get('error', ''))}")
                    kind_dist = opening_test_result.get("opening_odds_kind_distribution")
                    if kind_dist:
                        print(f"  opening_odds_kind dist: {kind_dist}")

    # ─── Calibration test (FS-03) ──────────────────────────────
    calibration_result = None
    if _CALIBRATE:
        print(f"\n  {'─' * 70}")
        print(f"  CALIBRATION TEST — none vs {_CALIBRATE}")
        print(f"  {'─' * 70}")

        # Use best test per universe
        for uid in universes:
            valid_in_uid = [r for r in results
                           if r and "error" not in r
                           and r.get("universe") == uid]
            if not valid_in_uid:
                continue
            best = min(valid_in_uid, key=lambda r: r["brier_ensemble"])
            udf = universes.get(uid, pd.DataFrame())
            if udf.empty:
                continue
            print(f"\n  [CAL] {best['test']} (universe={uid})...")
            cal = calibration_test(
                udf, best["features"], best["test"],
                params=best.get("optuna_params"),
                method=_CALIBRATE)

            if cal and cal.get("status") == "OK":
                calibration_result = cal
                n_ = cal["none"]
                c_ = cal["calibrated"]
                print(f"    None:       Brier={n_['brier']:.5f}  "
                      f"ECE_avg={n_['ece_avg']:.5f}")
                print(f"    {_CALIBRATE.title():10s}: Brier={c_['brier']:.5f}  "
                      f"ECE_avg={c_['ece_avg']:.5f}")
                print(f"    Δ_Brier={cal['delta_brier']:+.5f}  "
                      f"CI95[{cal['delta_ci95'][0]:+.5f}, "
                      f"{cal['delta_ci95'][1]:+.5f}]")
                print(f"    Conclusion: {cal['conclusion']}")
            elif cal:
                print(f"    {cal.get('status')}: "
                      f"{cal.get('error', cal.get('n', ''))}")

    # ─── Decomposition summary (FS-01/FS-02) ─────────────────
    if _DECOMPOSE:
        print(f"\n  {'─' * 70}")
        print(f"  DECOMPOSITION SUMMARY — Brier = REL - RES + UNC")
        print(f"  {'─' * 70}")

        # Market baseline
        if mkt and "brier_decomposition_10" in mkt:
            md = mkt["brier_decomposition_10"]
            me = mkt.get("ece", {})
            recon_err = abs(md["brier_reconstructed"] - mkt["brier_ensemble"])
            print(f"  MKT_market  REL={md['reliability']:.5f}  "
                  f"RES={md['resolution']:.5f}  UNC={md['uncertainty']:.5f}  "
                  f"recon_err={recon_err:.5f}  "
                  f"ECE_avg={me.get('ece_avg', 'N/A')}")

        # Top-5 tests by Brier
        valid = [r for r in results
                 if r and "error" not in r and "brier_decomposition_10" in r]
        valid.sort(key=lambda r: r["brier_ensemble"])
        for r in valid[:5]:
            d = r["brier_decomposition_10"]
            e = r.get("ece", {})
            recon_err = abs(d["brier_reconstructed"] - r["brier_ensemble"])
            print(f"  {r['test']:<25} REL={d['reliability']:.5f}  "
                  f"RES={d['resolution']:.5f}  UNC={d['uncertainty']:.5f}  "
                  f"recon_err={recon_err:.5f}  "
                  f"ECE_avg={e.get('ece_avg', 'N/A')}  "
                  f"dx={r.get('diagnosis', '-')}")

    return {
        "league_id": league_id,
        "league_name": league_name,
        "n_matches": len(df),
        "universes": {uid: len(udf) for uid, udf in universes.items()},
        "anchors_by_universe": anchors_by_universe,
        "tests": [r for r in results if r is not None],
        "two_stage_tests": [r for r in two_stage_results if r is not None],
        "market_baseline": mkt,
        "market_brier": mkt["brier_ensemble"] if mkt and "brier_ensemble" in mkt else None,
        "fair_comparisons": {k: {kk: vv for kk, vv in v.items()
                                  if kk != "alignment_check"}
                              for k, v in fair_comparisons.items()},
        "alignment_checks": alignment_report,
        "lockbox": lockbox_results if lockbox_mode else None,
        "residual_tests": residual_results if run_residual else None,
        "devig_sensitivity": devig_sensitivity,
        "walk_forward": walk_forward_results if walk_forward_results else None,
        "opening_test": opening_test_result,
        "calibration_test": calibration_result,
    }


# ─── Comparison Table ────────────────────────────────────────

def print_comparison(all_results: list[dict]):
    """Print cross-league comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  CROSS-LEAGUE COMPARISON")
    print(f"{'=' * 90}")

    # Header
    leagues = [r["league_name"] for r in all_results]
    header = f"  {'Test':<20}"
    for lg in leagues:
        header += f" {'Brier ' + lg:>25}"
    print(header)
    print(f"  {'─' * 20}" + f" {'─' * 25}" * len(leagues))

    # Collect all test names (preserving order)
    all_tests = list(TESTS.keys()) + ["MKT_market"]

    for test_name in all_tests:
        row = f"  {test_name:<20}"
        for lg_result in all_results:
            found = None
            if test_name == "MKT_market":
                found = lg_result.get("market_baseline")
            else:
                for t in lg_result["tests"]:
                    if t and t.get("test") == test_name:
                        found = t
                        break

            if found and "error" not in found:
                brier = found.get("brier_ensemble", found.get("brier_mean", 0))
                if "brier_ci95" in found:
                    ci_lo, ci_hi = found["brier_ci95"]
                    row += f"  {brier:.5f} [{ci_lo:.4f},{ci_hi:.4f}]"
                else:
                    row += f"  {brier:.5f}                 "
            else:
                err = found.get("error", "no data") if found else "no data"
                row += f"  {'—':>25}"
        print(row)

    # Delta from baseline
    print(f"\n  DELTA vs T0 (negative = better):")
    print(f"  {'Test':<20}", end="")
    for lg in leagues:
        print(f" {'Δ ' + lg:>15}", end="")
    print()
    print(f"  {'─' * 20}" + f" {'─' * 15}" * len(leagues))

    for test_name in all_tests:
        if test_name == "A0_baseline_17":
            continue
        row = f"  {test_name:<20}"
        for lg_result in all_results:
            # Get baseline brier
            base_brier = None
            for t in lg_result["tests"]:
                if t and t.get("test") == "A0_baseline_17" and "error" not in t:
                    base_brier = t.get("brier_ensemble", t.get("brier_mean"))
                    break

            # Get this test's brier
            found = None
            if test_name == "MKT_market":
                found = lg_result.get("market_baseline")
            else:
                for t in lg_result["tests"]:
                    if t and t.get("test") == test_name:
                        found = t
                        break

            if found and "error" not in found and base_brier is not None:
                delta = found.get("brier_ensemble", found.get("brier_mean", 0)) - base_brier
                sign = "+" if delta > 0 else ""
                row += f"  {sign}{delta:.5f}      "
            else:
                row += f"  {'—':>15}"
        print(row)


# ─── Optuna Runner ───────────────────────────────────────────

def run_optuna_tests(df: pd.DataFrame, league_id: int) -> dict:
    """Run Optuna-tuned evaluation on champion feature sets using universes."""
    league_name = LEAGUE_NAMES.get(league_id, f"league_{league_id}")

    print(f"\n{'=' * 70}")
    print(f"  OPTUNA LAB: {league_name} (id={league_id})")
    print(f"  Matches: {len(df)} | Candidates: {len(OPTUNA_CANDIDATES)}")
    print(f"  Trials: {OPTUNA_N_TRIALS} | CV Folds: {OPTUNA_CV_FOLDS}")
    print(f"{'=' * 70}")

    # Compute universes for optuna candidates
    universes = compute_universes(df, OPTUNA_CANDIDATES)

    results = []
    for test_name, features in OPTUNA_CANDIDATES.items():
        uid = classify_test_universe(features)
        udf = universes.get(uid, pd.DataFrame())
        print(f"\n  [{test_name}] {len(features)} features (universe={uid})...")
        if udf.empty:
            res = {"test": test_name, "universe": uid, "error": f"empty_universe:{uid}"}
            print(f"    SKIP: {res['error']}")
            results.append(res)
            continue
        res = evaluate_feature_set_optuna(udf, features, test_name)
        if res and "error" not in res:
            print(f"    TUNED Brier: {res['brier_ensemble']:.5f} "
                  f"(seeds: {res['brier_seed_mean']:.5f} ± {res['brier_seed_std']:.5f}) "
                  f"CI95[{res['brier_ci95'][0]:.5f}, {res['brier_ci95'][1]:.5f}] "
                  f"Acc: {res['accuracy_mean']:.3f}")
        elif res:
            print(f"    SKIP: {res['error']}")
        results.append(res)

    # Also run fixed-params baseline for comparison (base universe)
    print(f"\n  [FIXED_baseline] A0 with prod hyperparams (control)...")
    fixed_res = evaluate_feature_set(universes["base"], BASELINE_FEATURES, "FIXED_baseline")
    if fixed_res and "error" not in fixed_res:
        print(f"    FIXED Brier: {fixed_res['brier_ensemble']:.5f} "
              f"(seeds: {fixed_res['brier_seed_mean']:.5f} ± {fixed_res['brier_seed_std']:.5f})")
    results.append(fixed_res)

    # Market baseline (odds universe)
    print(f"\n  [MKT_market] De-vigged odds baseline...")
    mkt = market_brier(universes["odds"])
    if mkt:
        print(f"    Brier: {mkt['brier_ensemble']:.5f} "
              f"CI95[{mkt['brier_ci95'][0]:.5f}, {mkt['brier_ci95'][1]:.5f}] "
              f"(N_test={mkt['n_test']})")
    else:
        print(f"    SKIP: insufficient odds data")

    # Fair model-vs-market comparison with paired delta CI
    fair_comparisons = {}
    if mkt:
        valid = [r for r in results if r and "error" not in r and r.get("test") != "FIXED_baseline"]
        if valid:
            best = min(valid, key=lambda r: r["brier_ensemble"])
            best_uid = best.get("universe", "odds")
            # Use odds universe (or odds_xg if that's the best's universe)
            fair_udf = universes.get(best_uid if "odds" in best_uid else "odds",
                                     universes["odds"])
            fair = fair_model_vs_market(
                fair_udf, best["features"], best["test"],
                params=best.get("optuna_params"),
            )
            if fair:
                fair_comparisons[best["test"]] = fair
                delta_ci = fair.get("delta_ci95", [None, None])
                print(f"\n  [FAIR] {best['test']} vs market (N={fair['n_test_fair']}): "
                      f"model={fair['model_brier_fair']:.5f} market={fair['market_brier_fair']:.5f} "
                      f"Δ={fair['delta']:+.5f} "
                      f"CI95[{delta_ci[0]:+.5f}, {delta_ci[1]:+.5f}]")

    # Summary table
    print(f"\n  {'─' * 65}")
    print(f"  {'Test':<25} {'Brier':>10} {'CV Brier':>10} {'Depth':>6} {'LR':>8} {'N_est':>6}")
    print(f"  {'─' * 65}")
    for r in results:
        if r and "error" not in r:
            name = r["test"]
            brier = r.get("brier_ensemble", r.get("brier_mean", 0))
            cv = r.get("optuna_cv_brier", "—")
            p = r.get("optuna_params", {})
            depth = p.get("max_depth", "—")
            lr = p.get("learning_rate", "—")
            n_est = p.get("n_estimators", "—")
            cv_str = f"{cv:.5f}" if isinstance(cv, float) else str(cv)
            lr_str = f"{lr:.4f}" if isinstance(lr, float) else str(lr)
            print(f"  {name:<25} {brier:>10.5f} {cv_str:>10} {depth!s:>6} {lr_str:>8} {n_est!s:>6}")
    if mkt:
        print(f"  {'MKT_market':<25} {mkt['brier_ensemble']:>10.5f}")
    print(f"  {'─' * 65}")

    return {
        "league_id": league_id,
        "league_name": league_name,
        "n_matches": len(df),
        "mode": "optuna",
        "tests": [r for r in results if r is not None],
        "market_baseline": mkt,
        "fair_comparisons": fair_comparisons,
    }


def print_optuna_comparison(all_results: list[dict]):
    """Print cross-league comparison for Optuna results."""
    print(f"\n{'=' * 90}")
    print(f"  OPTUNA CROSS-LEAGUE COMPARISON")
    print(f"{'=' * 90}")

    leagues = [r["league_name"] for r in all_results]
    header = f"  {'Test':<25}"
    for lg in leagues:
        header += f" {lg:>20}"
    print(header)
    print(f"  {'─' * 25}" + f" {'─' * 20}" * len(leagues))

    all_tests = list(OPTUNA_CANDIDATES.keys()) + ["FIXED_baseline", "MKT_market"]

    for test_name in all_tests:
        row = f"  {test_name:<25}"
        for lg_result in all_results:
            found = None
            if test_name == "MKT_market":
                found = lg_result.get("market_baseline")
            else:
                for t in lg_result["tests"]:
                    if t and t.get("test") == test_name:
                        found = t
                        break
            if found and "error" not in found:
                brier = found.get("brier_ensemble", found.get("brier_mean", 0))
                row += f"  {brier:.5f}             "
            else:
                row += f"  {'—':>20}"
        print(row)


# ─── SHAP Analysis ────────────────────────────────────────────

# Feature sets to analyze with SHAP (key tests that answer specific questions)
SHAP_TESTS = {
    "S0_baseline_17":       ("Baseline v1.0.1", BASELINE_FEATURES),
    "S1_baseline_odds":     ("Baseline + Odds", BASELINE_FEATURES + ODDS_FEATURES),
    "S2_elo_odds":          ("Elo + Odds", ELO_FEATURES + ODDS_FEATURES),
    "S3_defense_elo":       ("Defense + Elo", DEFENSE_PAIR + ELO_FEATURES),
    "S4_m2_interactions":   ("ARG Signal + Elo + Interactions", ARG_SIGNAL + ELO_FEATURES + INTERACTION_FEATURES),
    "S5_xg_elo":            ("xG + Elo", XG_CORE + ELO_FEATURES),
    "S6_power_5":           ("Power 5", ["elo_diff", "opp_rating_diff", "overperf_diff",
                                          "home_goals_conceded_avg", "draw_elo_interaction"]),
    "S7_abe_elo":           ("ABE + Elo", OPP_ADJ_FEATURES + OVERPERF_FEATURES + DRAW_AWARE_FEATURES + HOME_BIAS_FEATURES + ELO_FEATURES),
    "S8_abe_elo_odds":      ("ABE + Elo + Odds", OPP_ADJ_FEATURES + OVERPERF_FEATURES + DRAW_AWARE_FEATURES + HOME_BIAS_FEATURES + ELO_FEATURES + ODDS_FEATURES),
    # MTV SHAP tests (GDT Mandato 3)
    "S9_mtv_elo":           ("MTV + Elo", MTV_FEATURES + ELO_FEATURES),
    "S10_mtv_elo_odds":     ("MTV + Elo + Odds", MTV_FEATURES + ELO_FEATURES + ODDS_FEATURES),
    "S11_mtv_xg_elo_odds":  ("MTV + xG + Elo + Odds", MTV_FEATURES + XG_CORE + ELO_FEATURES + ODDS_FEATURES),
    "S12_mtv_full":         ("MTV + DEF + Elo + FORM + Odds", MTV_FEATURES + DEFENSE_PAIR + ELO_FEATURES + FORM_CORE + ODDS_FEATURES),
}


def run_shap_analysis(df: pd.DataFrame, league_id: int, output_dir: str) -> dict:
    """Run SHAP TreeExplainer on key feature sets using fixed universes.

    Answers:
    - Which features drive predictions per class (H/D/A)?
    - How do odds interact with the model?
    - Are there non-linear interactions?
    """
    league_name = LEAGUE_NAMES.get(league_id, f"league_{league_id}")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  SHAP ANALYSIS: {league_name} (id={league_id})")
    print(f"  Tests: {len(SHAP_TESTS)} key feature sets")
    print(f"{'=' * 70}")

    # Compute universes for SHAP tests
    shap_feats_dict = {k: v[1] for k, v in SHAP_TESTS.items()}
    universes = compute_universes(df, shap_feats_dict)

    all_shap_results = []

    for test_key, (label, feature_names) in SHAP_TESTS.items():
        uid = classify_test_universe(feature_names)
        udf = universes.get(uid, pd.DataFrame())
        print(f"\n  [{test_key}] {label} ({len(feature_names)} features, universe={uid})...")

        if udf.empty:
            print(f"    SKIP: empty_universe:{uid}")
            all_shap_results.append({"test": test_key, "label": label,
                                     "error": f"empty_universe:{uid}"})
            continue

        prep = _prepare_dataset(udf, feature_names, test_key)
        if "error" in prep:
            print(f"    SKIP: {prep['error']}")
            all_shap_results.append({"test": test_key, "label": label, **prep})
            continue

        df_train, df_test = prep["df_train"], prep["df_test"]
        X_tr, y_tr, X_te, y_te = prep["X_tr"], prep["y_tr"], prep["X_te"], prep["y_te"]
        needs_odds = any(f in ODDS_FEATURES for f in feature_names)

        # Train model (single seed for SHAP — deterministic enough)
        model = train_xgb(X_tr, y_tr, seed=42)
        y_prob = model.predict_proba(X_te)
        brier = multiclass_brier(y_te, y_prob)

        # SHAP TreeExplainer
        print(f"    Computing SHAP values (N_test={len(df_test)})...")
        explainer = shap.TreeExplainer(model)
        shap_raw = explainer.shap_values(X_te)

        # SHAP 0.49+: ndarray (n_samples, n_features, n_classes)
        # Older SHAP: list of 3 arrays [H, D, A], each (n_samples, n_features)
        if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
            # (n_samples, n_features, n_classes) → per-class: shap_raw[:, :, cls]
            shap_per_class = [shap_raw[:, :, c] for c in range(shap_raw.shape[2])]
        else:
            shap_per_class = shap_raw  # list of arrays

        class_names = ["Home", "Draw", "Away"]
        result = {
            "test": test_key,
            "label": label,
            "n_features": len(feature_names),
            "features": feature_names,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "brier": round(brier, 5),
        }

        # Per-class SHAP importance (mean absolute SHAP value per feature)
        per_class = {}
        for cls_idx, cls_name in enumerate(class_names):
            sv = shap_per_class[cls_idx]  # (n_test, n_features)
            mean_abs = np.mean(np.abs(sv), axis=0)

            # Rank features by importance
            ranking = sorted(zip(feature_names, mean_abs.tolist()),
                             key=lambda x: x[1], reverse=True)

            per_class[cls_name] = {
                "ranking": [{"feature": f, "mean_abs_shap": round(v, 5)} for f, v in ranking],
                "top3": [f for f, _ in ranking[:3]],
            }

        result["per_class"] = per_class

        # Global importance (mean across all classes)
        global_importance = np.zeros(len(feature_names))
        for cls_idx in range(3):
            global_importance += np.mean(np.abs(shap_per_class[cls_idx]), axis=0)
        global_importance /= 3.0

        global_ranking = sorted(zip(feature_names, global_importance.tolist()),
                                key=lambda x: x[1], reverse=True)
        result["global_ranking"] = [{"feature": f, "mean_abs_shap": round(v, 5)} for f, v in global_ranking]

        # SHAP interaction detection: mean SHAP value (signed) per feature per class
        # Positive mean = pushes towards this class, Negative = pushes away
        signed_means = {}
        for cls_idx, cls_name in enumerate(class_names):
            sv = shap_per_class[cls_idx]
            means = np.mean(sv, axis=0)
            signed_means[cls_name] = {f: round(float(v), 5) for f, v in zip(feature_names, means)}
        result["signed_mean_shap"] = signed_means

        # For odds tests: analyze how odds features interact with model
        if needs_odds:
            odds_idx = [feature_names.index(f) for f in ODDS_FEATURES if f in feature_names]
            non_odds_idx = [i for i in range(len(feature_names)) if i not in odds_idx]

            odds_contrib = sum(global_importance[i] for i in odds_idx)
            non_odds_contrib = sum(global_importance[i] for i in non_odds_idx)
            total = odds_contrib + non_odds_contrib

            result["odds_analysis"] = {
                "odds_share_pct": round(odds_contrib / total * 100, 1) if total > 0 else 0,
                "non_odds_share_pct": round(non_odds_contrib / total * 100, 1) if total > 0 else 0,
                "odds_dominance": odds_contrib > non_odds_contrib,
            }

        all_shap_results.append(result)

        # Print summary
        print(f"    Brier: {brier:.5f}")
        print(f"    Global top-5:")
        for rank_item in result["global_ranking"][:5]:
            print(f"      {rank_item['feature']:<35} {rank_item['mean_abs_shap']:.5f}")

        if needs_odds and "odds_analysis" in result:
            oa = result["odds_analysis"]
            print(f"    Odds share: {oa['odds_share_pct']}% | Non-odds: {oa['non_odds_share_pct']}%")

    # Cross-test summary table
    print(f"\n{'=' * 70}")
    print(f"  SHAP SUMMARY: {league_name}")
    print(f"{'=' * 70}")
    print(f"  {'Test':<25} {'Brier':>8} {'#1 Feature':<30} {'#2 Feature':<25}")
    print(f"  {'─' * 88}")
    for r in all_shap_results:
        if "error" in r:
            print(f"  {r['test']:<25} {'—':>8} {r['error']}")
            continue
        gr = r["global_ranking"]
        f1 = gr[0]["feature"] if len(gr) > 0 else "—"
        f2 = gr[1]["feature"] if len(gr) > 1 else "—"
        print(f"  {r['test']:<25} {r['brier']:>8.5f} {f1:<30} {f2:<25}")

    # Per-class comparison: which features matter most for DRAWS?
    print(f"\n  DRAW-CLASS Top-3 per test:")
    for r in all_shap_results:
        if "error" in r:
            continue
        draw_top3 = r["per_class"]["Draw"]["top3"]
        print(f"    {r['test']:<25} {', '.join(draw_top3)}")

    # Save results
    shap_output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "league_id": league_id,
            "league_name": league_name,
            "n_matches": len(df),
            "universes": {uid: len(udf) for uid, udf in universes.items()},
        },
        "shap_results": all_shap_results,
    }

    shap_path = out_path / f"shap_analysis_{league_id}.json"
    with open(shap_path, "w") as f:
        json.dump(shap_output, f, indent=2, default=str)
    print(f"\n  SHAP results saved to: {shap_path}")

    return shap_output


# ─── Signal Map (marginal contribution per feature category) ──

# Feature categories for signal map — maps category name to
# (test_with_category, test_without_category) for delta computation
SIGNAL_MAP_CATEGORIES = {
    "Odds":         ("J2_full_odds",        "A0_baseline_17"),
    "Elo":          ("A1_only_elo_k32",     "A0_baseline_17"),
    "Elo+Baseline": ("H1_defense_elo",      "A0_baseline_17"),
    "Form":         ("E3_form_elo",         "A1_only_elo_k32"),
    "Matchups":     ("F1_matchup_elo",      "A1_only_elo_k32"),
    "xG":           ("P2_xg_elo",           "A1_only_elo_k32"),
    "Interactions":  ("L1_interactions_elo", "A1_only_elo_k32"),
    "Opp-Adjusted": ("K1_opp_adj_elo",      "A1_only_elo_k32"),
    "Draw-Aware":   ("K5_draw_aware_elo",   "A1_only_elo_k32"),
    "H2H":          ("F4_h2h_elo",          "A1_only_elo_k32"),
    "MTV":          ("S1_mtv_elo",          "A1_only_elo_k32"),
    "XI Continuity": ("Q1_xi_elo",          "A1_only_elo_k32"),
    "Geo":          ("U1_geo_elo",          "A1_only_elo_k32"),
    "Standings":    ("T1_standings_elo",     "A1_only_elo_k32"),
    "Precip":       ("X1_precip_elo",       "A1_only_elo_k32"),
    # Two-Stage architecture effect
    "TwoStage (arch)":     ("W2_ts_baseline_elo",  "A0_baseline_17"),
    "TwoStage + Odds":     ("W5_ts_full_odds",     "J2_full_odds"),
    "TwoStage + Implied":  ("W8_ts_implied_odds",  "J2_full_odds"),
}


def generate_signal_map(all_results):
    """Generate per-league signal map showing marginal contribution of each feature category.

    Returns dict: {league_name: {category: {delta, with_brier, without_brier, signal}}}
    """
    signal_maps = []

    for league_result in all_results:
        league_name = league_result.get("league_name", "unknown")
        tests = league_result.get("tests", [])
        ts_tests = league_result.get("two_stage_tests", [])

        # Build lookup: test_name -> brier_ensemble
        lookup = {}
        for t in tests:
            if t and "error" not in t:
                lookup[t["test"]] = t["brier_ensemble"]
        for t in ts_tests:
            if t and "error" not in t:
                lookup[t["test"]] = t["brier_ensemble"]

        # Market baseline
        market_brier = league_result.get("market_brier")

        categories = {}
        for cat_name, (with_test, without_test) in SIGNAL_MAP_CATEGORIES.items():
            b_with = lookup.get(with_test)
            b_without = lookup.get(without_test)
            if b_with is not None and b_without is not None:
                delta = round(b_without - b_with, 5)  # Positive = category helps
                categories[cat_name] = {
                    "delta": delta,
                    "with_brier": b_with,
                    "without_brier": b_without,
                    "signal": "STRONG" if delta > 0.005 else
                              "MODERATE" if delta > 0.001 else
                              "WEAK" if delta > 0 else "NOISE",
                }

        signal_maps.append({
            "league": league_name,
            "league_id": league_result.get("league_id"),
            "market_brier": market_brier,
            "categories": categories,
        })

    return signal_maps


def print_signal_map(signal_maps):
    """Pretty-print the signal map across all leagues."""
    if not signal_maps:
        return

    print(f"\n{'=' * 100}")
    print(f"  SIGNAL MAP — Marginal Contribution by Feature Category")
    print(f"  (delta = Brier improvement when adding category; positive = helps)")
    print(f"{'=' * 100}")

    # Collect all categories across all leagues
    all_cats = []
    for sm in signal_maps:
        all_cats.extend(sm["categories"].keys())
    all_cats = sorted(set(all_cats), key=lambda c: list(SIGNAL_MAP_CATEGORIES.keys()).index(c)
                      if c in SIGNAL_MAP_CATEGORIES else 999)

    # Header
    leagues = [sm["league"] for sm in signal_maps]
    header = f"  {'Category':<22}"
    for lg in leagues:
        header += f" {lg[:12]:>12}"
    print(header)
    print(f"  {'─' * 22}" + f" {'─' * 12}" * len(leagues))

    # Market baseline row
    mkt_row = f"  {'Market Brier':<22}"
    for sm in signal_maps:
        mb = sm.get("market_brier")
        mkt_row += f" {mb:>12.5f}" if mb else f" {'N/A':>12}"
    print(mkt_row)
    print(f"  {'─' * 22}" + f" {'─' * 12}" * len(leagues))

    # Category rows
    for cat in all_cats:
        row = f"  {cat:<22}"
        for sm in signal_maps:
            info = sm["categories"].get(cat)
            if info:
                d = info["delta"]
                sig = info["signal"][0]  # S/M/W/N
                if d > 0.005:
                    row += f"  {d:>+.5f} *"
                elif d > 0.001:
                    row += f"  {d:>+.5f} ."
                elif d > 0:
                    row += f"  {d:>+.5f}  "
                else:
                    row += f"  {d:>+.5f} -"
            else:
                row += f" {'---':>12}"
        print(row)

    # Summary: best category per league
    print(f"\n  {'─' * 22}" + f" {'─' * 12}" * len(leagues))
    best_row = f"  {'BEST CATEGORY':<22}"
    for sm in signal_maps:
        cats = sm["categories"]
        if cats:
            best = max(cats.items(), key=lambda x: x[1]["delta"])
            best_row += f" {best[0][:12]:>12}"
        else:
            best_row += f" {'N/A':>12}"
    print(best_row)

    invest_row = f"  {'INVEST IN':<22}"
    for sm in signal_maps:
        cats = sm["categories"]
        # Find categories with STRONG or MODERATE signal
        invest = [c for c, v in cats.items()
                  if v["signal"] in ("STRONG", "MODERATE")
                  and c not in ("Elo", "Elo+Baseline")]  # Elo is always available
        if invest:
            invest_row += f" {invest[0][:12]:>12}"
        else:
            invest_row += f" {'(nothing)':>12}"
    print(invest_row)

    print()


# ─── Ablation: Competitiveness Features (14 vs 17) ───────────

def run_competitiveness_ablation(df, league_id, lockbox_mode=False):
    """Ablation test: BASELINE_14 vs BASELINE_17 (competitiveness features).

    Uses a single dataset preparation (BASELINE_17 superset) and derives
    BASELINE_14 subset from the same df_train/df_test for apples-to-apples.
    """
    league_name = LEAGUE_NAMES.get(league_id, f"league_{league_id}")

    # Use "base" universe (no odds/xg/mtv needed)
    tests_for_universe = {"ABLATE_baseline_17": BASELINE_FEATURES}
    universes = compute_universes(df, tests_for_universe)
    udf = universes.get("base")
    if udf is None or udf.empty:
        return {"league_id": league_id, "league_name": league_name,
                "error": "empty_universe"}

    # Single preparation with BASELINE_17 (superset)
    prep = _prepare_dataset(udf, BASELINE_FEATURES, "ABLATE_baseline_17",
                            lockbox_mode=lockbox_mode)
    if "error" in prep:
        return {"league_id": league_id, "league_name": league_name, **prep}

    df_train, df_test = prep["df_train"], prep["df_test"]
    y_tr, y_te = prep["y_tr"], prep["y_te"]
    X_tr_17, X_te_17 = prep["X_tr"], prep["X_te"]

    # Derive BASELINE_14 from same DataFrames (apples-to-apples)
    X_tr_14 = df_train[BASELINE_14_FEATURES].fillna(0).values.astype(np.float32)
    X_te_14 = df_test[BASELINE_14_FEATURES].fillna(0).values.astype(np.float32)

    split_date = str(df_test["date"].min())

    # Train/eval both with N_SEEDS ensemble
    def _train_eval(X_tr, X_te):
        all_probs = []
        for seed_i in range(N_SEEDS):
            seed = seed_i * 42 + 7
            model = train_xgb(X_tr, y_tr, seed=seed)
            y_prob = model.predict_proba(X_te)
            all_probs.append(y_prob)
        ensemble = np.mean(all_probs, axis=0)
        brier = multiclass_brier(y_te, ensemble)
        ci = bootstrap_ci(y_te, ensemble)
        return ensemble, brier, ci

    prob_14, brier_14, ci_14 = _train_eval(X_tr_14, X_te_14)
    prob_17, brier_17, ci_17 = _train_eval(X_tr_17, X_te_17)

    # Paired delta: Δ = brier_17 - brier_14
    delta = brier_17 - brier_14
    delta_ci = bootstrap_paired_delta(y_te, prob_17, prob_14)

    return {
        "league_id": league_id,
        "league_name": league_name,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "split_date": split_date,
        "brier_14": round(brier_14, 5),
        "brier_14_ci95": [round(ci_14[0], 5), round(ci_14[1], 5)],
        "brier_17": round(brier_17, 5),
        "brier_17_ci95": [round(ci_17[0], 5), round(ci_17[1], 5)],
        "delta": round(delta, 5),
        "delta_ci95": [round(delta_ci[0], 5), round(delta_ci[1], 5)],
    }


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature Lab — Experimental Feature Testing")
    parser.add_argument("--extract", action="store_true",
                        help="Extract fresh data from DB")
    parser.add_argument("--league", type=int, action="append", default=None,
                        help="League ID(s) to test (can repeat). Default: 128, 94")
    parser.add_argument("--data-dir", type=str, default="scripts/output/lab",
                        help="Directory for cached data")
    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna tuning on champion feature sets (Section O)")
    parser.add_argument("--shap", action="store_true",
                        help="Run SHAP explainability analysis on key feature sets")
    parser.add_argument("--lockbox", action="store_true",
                        help="Lockbox mode: 70/15/15 split, one-shot champion eval")
    parser.add_argument("--residual", action="store_true",
                        help="Run Section R: Market Residual tests")
    parser.add_argument("--min-date", type=str, default=None,
                        help="Min date filter YYYY-MM-DD (applied after load)")
    parser.add_argument("--decompose", action="store_true",
                        help="Add Brier decomposition (cal/res/unc) + ECE per class")
    parser.add_argument("--devig-sensitivity", action="store_true",
                        help="Compare market Brier under proportional/power/shin devig")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Walk-forward multi-window evaluation on champions")
    parser.add_argument("--opening-test", action="store_true",
                        help="Opening vs closing odds test (commercial edge)")
    parser.add_argument("--calibrate", type=str, default=None,
                        choices=["isotonic", "platt"],
                        help="Calibration test: isotonic or platt vs none")
    parser.add_argument("--strict-mtv", action="store_true",
                        help="GDT strict mode: drop matches without MTV data before universes "
                             "(forces apples-to-apples temporal overlap)")
    parser.add_argument("--two-stage", action="store_true",
                        help="Run Section W: Two-Stage architecture tests (Shadow B)")
    parser.add_argument("--signal-map", action="store_true",
                        help="Generate signal map showing marginal contribution per feature "
                             "category per league")
    parser.add_argument("--cross-league", action="store_true",
                        help="Merge all specified leagues into one dataset for cross-league "
                             "training experiment")
    parser.add_argument("--all-leagues", action="store_true",
                        help="Run on ALL 25 active domestic leagues")
    parser.add_argument("--ablate-competitiveness", action="store_true",
                        help="Run ablation BASELINE_14 vs BASELINE_17 (competitiveness features)")
    args = parser.parse_args()

    data_dir = args.data_dir
    run_optuna = args.optuna
    run_shap = args.shap
    lockbox_mode = args.lockbox
    run_residual = args.residual
    run_two_stage = args.two_stage
    run_signal_map = args.signal_map
    run_cross_league = args.cross_league

    # --all-leagues: run on all 23 primary league IDs (25 leagues via split)
    if args.all_leagues:
        league_ids = sorted(k for k in LEAGUE_NAMES.keys() if k > 0)
    else:
        league_ids = args.league or [128, 94]

    # signal-map implies two-stage (need W tests for TwoStage categories)
    if run_signal_map:
        run_two_stage = True

    # Set diagnostic flags (FS-01/FS-02/FS-03/FS-04/FS-06/FS-07)
    global _DECOMPOSE, _DEVIG_SENSITIVITY, _WALK_FORWARD, _OPENING_TEST, _CALIBRATE
    if args.decompose:
        _DECOMPOSE = True
    if args.devig_sensitivity:
        _DEVIG_SENSITIVITY = True
    if args.walk_forward:
        _WALK_FORWARD = True
    if args.opening_test:
        _OPENING_TEST = True
    if args.calibrate:
        _CALIBRATE = args.calibrate

    if run_optuna and not HAS_OPTUNA:
        print("  ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    if run_shap and not HAS_SHAP:
        print("  ERROR: shap not installed. Run: pip install shap")
        sys.exit(1)

    run_ablation = args.ablate_competitiveness

    if run_ablation:
        mode_label = "ABLATION_COMPETITIVENESS"
    elif run_shap:
        mode_label = "SHAP"
    elif run_optuna:
        mode_label = "OPTUNA"
    elif lockbox_mode:
        mode_label = "LOCKBOX"
    else:
        mode_label = "STANDARD"
    if run_residual:
        mode_label += "+RESIDUAL"
    if _DECOMPOSE:
        mode_label += "+DECOMPOSE"
    if _DEVIG_SENSITIVITY:
        mode_label += "+DEVIG"
    if _WALK_FORWARD:
        mode_label += "+WALKFWD"
    if _OPENING_TEST:
        mode_label += "+OPENING"
    if _CALIBRATE:
        mode_label += "+CAL_%s" % _CALIBRATE.upper()
    if args.strict_mtv:
        mode_label += "+STRICT_MTV"
    if run_two_stage:
        mode_label += "+TWO_STAGE"
    if run_signal_map:
        mode_label += "+SIGNAL_MAP"
    if run_cross_league:
        mode_label += "+CROSS_LEAGUE"

    print(f"\n  FEATURE LAB ({mode_label})")
    print(f"  {'=' * 50}")
    print(f"  Leagues: {[LEAGUE_NAMES.get(l, l) for l in league_ids]} ({len(league_ids)})")
    if run_ablation:
        print(f"  Ablation: BASELINE_14 ({len(BASELINE_14_FEATURES)}f) vs "
              f"BASELINE_17 ({len(BASELINE_FEATURES)}f)")
        print(f"  Competitiveness: {COMPETITIVENESS}")
    elif run_shap:
        print(f"  SHAP tests: {len(SHAP_TESTS)} key feature sets")
    elif run_optuna:
        print(f"  Candidates: {len(OPTUNA_CANDIDATES)} champion sets")
        print(f"  Optuna: {OPTUNA_N_TRIALS} trials x {OPTUNA_CV_FOLDS}-fold temporal CV")
    else:
        n_tests = len(TESTS)
        if run_two_stage:
            n_tests += len(TWO_STAGE_TESTS)
        print(f"  Tests: {n_tests} feature sets + market baseline")
    if run_residual:
        print(f"  Residual: {len(RESIDUAL_TESTS)} market-residual tests (Section R)")
    if run_cross_league:
        print(f"  Cross-league: merge all leagues into one training set")
    print(f"  Seeds: {N_SEEDS} | Bootstrap: {N_BOOTSTRAP}")

    all_results = []
    all_league_dfs = {}  # league_id -> DataFrame (for cross-league merge)

    for league_id in league_ids:
        csv_path = Path(data_dir) / f"lab_data_{league_id}.csv"

        if args.extract or not csv_path.exists():
            df = extract_league_data(league_id, output_dir=data_dir)
        else:
            print(f"\n  Loading cached: {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["date"])
            # Recompute all sequential features (Elo + variants + form + matchup)
            df = compute_elo_goals(df)
            df = compute_all_experimental_features(df)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        # Compute implied_draw from odds (for Two-Stage tests)
        df = compute_implied_draw(df)
        n_implied = int(df["implied_draw"].notna().sum())
        print(f"  Implied draw computed: {n_implied}/{len(df)} ({100*n_implied/len(df):.1f}%)")

        # MTV: Merge historical talent_delta features from pre-computed parquet
        mtv_path = Path("data/historical_mtv_features.parquet")
        if mtv_path.exists():
            mtv_df = pd.read_parquet(mtv_path, columns=[
                "match_id", "home_talent_delta", "away_talent_delta",
                "talent_delta_diff", "shock_magnitude", "talent_delta_missing",
            ])
            df["match_id"] = df["match_id"].astype("Int64")
            mtv_df["match_id"] = mtv_df["match_id"].astype("Int64")
            df = df.merge(mtv_df, on="match_id", how="left")
            n_mtv = int(df["home_talent_delta"].notna().sum())
            print(f"  MTV merged: {n_mtv}/{len(df)} ({100*n_mtv/len(df):.1f}%)")
        else:
            print(f"  MTV: parquet not found at {mtv_path}, skipping")
            for col in MTV_FEATURES + ["talent_delta_missing"]:
                df[col] = None

        # GDT Mandato 2: Strict MTV mode — force apples-to-apples temporal overlap
        if args.strict_mtv:
            initial_len = len(df)
            df = df.dropna(subset=["home_talent_delta"]).reset_index(drop=True)
            print(f"  [GDT STRICT MODE] Dropped {initial_len - len(df)} matches "
                  f"to force exact temporal overlap ({len(df)} remain, 2023+)")

        if args.min_date:
            cutoff = pd.Timestamp(args.min_date)
            before = len(df)
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            print(f"  Date filter >= {args.min_date}: {before} → {len(df)} rows")

        if df.empty:
            print(f"  SKIP: no data for league {league_id}")
            continue

        # Store for cross-league merge (before running tests)
        if run_cross_league:
            all_league_dfs[league_id] = df.copy()

        if run_ablation:
            result = run_competitiveness_ablation(df, league_id,
                                                  lockbox_mode=lockbox_mode)
        elif run_shap:
            result = run_shap_analysis(df, league_id, data_dir)
        elif run_optuna:
            result = run_optuna_tests(df, league_id)
        else:
            result = run_league_tests(df, league_id, lockbox_mode=lockbox_mode,
                                      run_residual=run_residual,
                                      run_two_stage=run_two_stage)
        all_results.append(result)

    # ─── Cross-league merge mode ──────────────────────────────
    cross_league_result = None
    if run_cross_league and len(all_league_dfs) > 1:
        print(f"\n{'=' * 70}")
        print(f"  CROSS-LEAGUE MERGE — {len(all_league_dfs)} leagues combined")
        print(f"{'=' * 70}")

        # Concatenate all league DataFrames, add league_id column for tracking
        merged_dfs = []
        for lid, ldf in all_league_dfs.items():
            ldf_copy = ldf.copy()
            ldf_copy["source_league_id"] = lid
            merged_dfs.append(ldf_copy)
        df_merged = pd.concat(merged_dfs, ignore_index=True)
        df_merged = df_merged.sort_values(["date", "match_id"]).reset_index(drop=True)

        # Re-compute Elo on merged dataset (cross-league Elo)
        print(f"  Merged: {len(df_merged)} matches from {len(all_league_dfs)} leagues")
        print(f"  Re-computing Elo on merged dataset...")
        df_merged = compute_elo_goals(df_merged)
        df_merged = compute_all_experimental_features(df_merged)
        df_merged = compute_implied_draw(df_merged)

        cross_league_result = run_league_tests(
            df_merged, league_id=0,
            lockbox_mode=lockbox_mode,
            run_residual=run_residual,
            run_two_stage=run_two_stage)
        cross_league_result["league_name"] = "CROSS-LEAGUE"
        cross_league_result["league_id"] = 0
        cross_league_result["source_leagues"] = list(all_league_dfs.keys())
        all_results.append(cross_league_result)

    # ─── Ablation output ─────────────────────────────────────
    if run_ablation and all_results:
        valid = [r for r in all_results if "error" not in r]
        errors = [r for r in all_results if "error" in r]
        aggregate = None

        if valid:
            print(f"\n{'=' * 85}")
            print(f"  ABLATION: BASELINE_14 vs BASELINE_17 (Competitiveness Features)")
            print(f"{'=' * 85}")
            print(f"  {'League':<18} {'Brier14':>8} {'Brier17':>8} "
                  f"{'Delta':>8} {'CI95_lo':>8} {'CI95_hi':>8} {'N_test':>7}")
            print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

            for r in valid:
                sig = ""
                if r["delta_ci95"][1] < 0:
                    sig = " **"  # 17 significantly better
                elif r["delta_ci95"][0] > 0:
                    sig = " !!"  # 14 significantly better
                print(f"  {r['league_name']:<18} {r['brier_14']:>8.5f} {r['brier_17']:>8.5f} "
                      f"{r['delta']:>+8.5f} {r['delta_ci95'][0]:>+8.5f} "
                      f"{r['delta_ci95'][1]:>+8.5f} {r['n_test']:>7d}{sig}")

            # Weighted aggregate
            total_n = sum(r["n_test"] for r in valid)
            delta_weighted = sum(r["delta"] * r["n_test"] for r in valid) / total_n
            n_better_17 = sum(1 for r in valid if r["delta"] < 0)
            n_better_14 = sum(1 for r in valid if r["delta"] > 0)
            n_neutral = sum(1 for r in valid if r["delta"] == 0)

            print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
            print(f"  {'WEIGHTED AVG':<18} {'':>8} {'':>8} {delta_weighted:>+8.5f} "
                  f"{'':>8} {'':>8} {total_n:>7d}")
            print(f"\n  17 better (D<0): {n_better_17} | 14 better (D>0): {n_better_14}"
                  f" | neutral: {n_neutral}")
            print(f"  ** = CI95 entirely <0 (17 sig. better)")
            print(f"  !! = CI95 entirely >0 (14 sig. better)")

            # Aggregate to results
            aggregate = {
                "delta_weighted": round(delta_weighted, 5),
                "total_n_test": total_n,
                "n_leagues": len(valid),
                "n_better_17": n_better_17,
                "n_better_14": n_better_14,
                "n_neutral": n_neutral,
            }

        if errors:
            print(f"\n  Errors ({len(errors)}):")
            for r in errors:
                print(f"    {r['league_name']}: {r['error']}")

        # Save JSON
        output_path = Path(data_dir) / "feature_lab_ablation_competitiveness.json"
        save_payload = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "mode": mode_label,
                "leagues": league_ids,
                "n_seeds": N_SEEDS,
                "n_bootstrap": N_BOOTSTRAP,
                "test_fraction": TEST_FRACTION,
                "prod_hyperparams": PROD_HYPERPARAMS,
                "baseline_14_features": BASELINE_14_FEATURES,
                "baseline_17_features": BASELINE_FEATURES,
                "competitiveness_features": COMPETITIVENESS,
            },
            "results": all_results,
            "aggregate": aggregate if valid else None,
        }
        with open(output_path, "w") as f:
            json.dump(save_payload, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    # Cross-league comparison (standard/optuna modes only)
    elif not run_shap and len(all_results) > 1:
        if run_optuna:
            print_optuna_comparison(all_results)
        else:
            print_comparison(all_results)

    # ─── Signal Map ───────────────────────────────────────────
    if not run_ablation and run_signal_map and all_results:
        signal_maps = generate_signal_map(all_results)
        print_signal_map(signal_maps)

    # Save results (standard/optuna modes — SHAP saves its own file)
    if not run_ablation and not run_shap:
        suffix = "_optuna" if run_optuna else ""
        if run_signal_map:
            suffix += "_signal_map"
        if run_cross_league:
            suffix += "_cross_league"
        output_path = Path(data_dir) / f"feature_lab_results{suffix}.json"
        save_payload = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "mode": mode_label,
                "leagues": league_ids,
                "n_seeds": N_SEEDS,
                "n_bootstrap": N_BOOTSTRAP,
                "elo_k": ELO_K,
                "elo_home_adv": ELO_HOME_ADV,
                "elo_initial": ELO_INITIAL,
                "test_fraction": TEST_FRACTION,
                "rolling_window": ROLLING_WINDOW,
                "time_decay_lambda": TIME_DECAY_LAMBDA,
                "draw_weight": DRAW_WEIGHT,
                **({"optuna_n_trials": OPTUNA_N_TRIALS,
                    "optuna_cv_folds": OPTUNA_CV_FOLDS} if run_optuna else
                   {"prod_hyperparams": PROD_HYPERPARAMS}),
            },
            "results": all_results,
        }
        if run_signal_map and all_results:
            save_payload["signal_maps"] = signal_maps
        with open(output_path, "w") as f:
            json.dump(save_payload, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
