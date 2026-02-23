#!/usr/bin/env python3
"""
VORP Calibration Lab — Sprint 2, Camino B "El Francotirador T-60"

GDT directive: calibrate β for log-shift-softmax adjustment using
talent_delta_diff as the lineup shock signal on LATAM v1.3.0 base probs.

Formula (GDT exact):
  Z = log(P_base)              # logit transform (clipped)
  Z_home += β * d              # shift home by talent advantage
  Z_away -= β * d              # shift away inversely
  Z_draw  unchanged            # draw absorbs via softmax renorm
  P_adj = softmax(Z_adj)       # back to probability space

Usage:
  source .env

  # Phase 1: materialize talent_delta_diff for all LATAM FT matches (run once, ~2 min)
  python scripts/calibrate_vorp_lab.py --materialize

  # Phase 2: optimize β (fast, repeatable, ~10 sec)
  python scripts/calibrate_vorp_lab.py

  # Custom date filter
  python scripts/calibrate_vorp_lab.py --min-date 2024-01-01

Output:
  scripts/output/lab/vorp_talent_delta_cache.csv   (Phase 1 cache)
  scripts/output/lab/vorp_lab_results.json          (β*, Brier, Skill Score)
  scripts/output/lab/vorp_lab_detail.csv            (per-match detail)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ─── Project root ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vorp_lab")

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

LATAM_LEAGUE_IDS = [128, 71, 239, 242, 262, 265, 268, 281, 299, 344, 250]
MIN_DATE_DEFAULT = "2023-01-01"
TRAIN_FRAC = 0.80
EPS = 1e-7
BETA_MAX = 10.0
N_BOOTSTRAP = 2000

FEATURES_14 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "home_rest_days", "away_rest_days",
    "home_matches_played", "away_matches_played",
    "goal_diff_avg", "rest_diff",
]
MOMENTUM_2 = ["elo_k10_diff", "elo_momentum_diff"]
GEO_2 = ["altitude_diff_m", "travel_distance_km"]
FEATURES_LATAM = FEATURES_14 + MOMENTUM_2 + GEO_2  # 18f

LATAM_VERSION_PATTERN = "v1.3.0-latam%"

OUTPUT_DIR = Path("scripts/output/lab")
CACHE_FILE = OUTPUT_DIR / "vorp_talent_delta_cache.csv"
RESULT_FILE = OUTPUT_DIR / "vorp_lab_results.json"
DETAIL_FILE = OUTPUT_DIR / "vorp_lab_detail.csv"

LEAGUE_NAMES = {
    128: "Argentina", 71: "Brazil", 239: "Colombia", 242: "Ecuador",
    262: "Mexico", 265: "Chile", 268: "Uruguay", 281: "Peru",
    299: "Venezuela", 344: "Bolivia", 250: "Paraguay",
}


# ═══════════════════════════════════════════════════════════════════════════
# Math utilities
# ═══════════════════════════════════════════════════════════════════════════

def stable_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. z: (N, 3) or (3,)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def brier_multiclass(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score for 3-class (lower = better). Standard: mean(sum((p - y)^2))."""
    n = len(outcomes)
    if n == 0:
        return float("nan")
    y = np.zeros((n, 3), dtype=np.float64)
    for i, o in enumerate(outcomes):
        y[i, int(o)] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def adjust_probs(p_base: np.ndarray, d: np.ndarray, beta: float) -> np.ndarray:
    """
    Log-shift-softmax adjustment (GDT exact formula).

    p_base: (N, 3) [home, draw, away]
    d: (N,) talent_delta_diff (positive = home advantage)
    beta: scalar >= 0

    Returns: (N, 3) adjusted probabilities summing to 1.
    """
    p_clipped = np.clip(p_base, EPS, 1.0 - EPS)
    z = np.log(p_clipped)
    z[:, 0] += beta * d     # home gets boost
    z[:, 2] -= beta * d     # away gets penalty
    # z[:, 1] unchanged      # draw absorbs via softmax
    return stable_softmax(z)


def odds_to_probs(h: float, d: float, a: float):
    """Proportional de-vig from decimal odds."""
    inv_sum = 1.0 / h + 1.0 / d + 1.0 / a
    return 1.0 / (h * inv_sum), 1.0 / (d * inv_sum), 1.0 / (a * inv_sum)


def skill_score(brier_model: float, brier_market: float) -> float:
    """1 - model/market. Positive = alpha over market."""
    if brier_market <= 0:
        return 0.0
    return 1.0 - brier_model / brier_market


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Materialize talent_delta_diff (batch SQL + Python)
# ═══════════════════════════════════════════════════════════════════════════

class TalentDeltaComputer:
    """
    Batch compute talent_delta_diff using pre-loaded data in memory.

    Replicates the logic of compute_match_talent_delta_features() but
    operates on pre-loaded DataFrames instead of per-match DB queries.
    """

    def __init__(self, mps_df: pd.DataFrame, lineups_df: pd.DataFrame):
        """
        mps_df: match_player_stats (match_id, player_external_id, team_id,
                 match_date, rating, minutes, position, is_substitute)
        lineups_df: match_lineups (match_id, team_id, starting_xi_ids)
        """
        self._build_lineup_index(lineups_df)
        self._build_team_history(mps_df)

    def _build_lineup_index(self, lineups_df: pd.DataFrame):
        """Index lineups by (match_id, team_id) for O(1) lookup."""
        self.lineups = {}
        for _, row in lineups_df.iterrows():
            key = (int(row["match_id"]), int(row["team_id"]))
            xi = row["starting_xi_ids"]
            if xi is None:
                continue
            if isinstance(xi, str):
                try:
                    xi = json.loads(xi)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(xi, (list, np.ndarray)) and len(xi) >= 7:
                clean = [int(x) for x in xi if x is not None]
                if len(clean) >= 7:
                    self.lineups[key] = clean

    def _build_team_history(self, mps_df: pd.DataFrame):
        """
        Build per-team match history sorted by date.
        Structure: team_id -> list of {match_id, match_date, players: {pid: {rating, is_sub}}}
        """
        self.team_history = {}

        # Group by team + match for efficiency
        grouped = mps_df.groupby(["team_id", "match_id"])
        for (team_id, match_id), group in grouped:
            team_id = int(team_id)
            match_id = int(match_id)
            match_date = group["match_date"].iloc[0]

            players = {}
            for _, r in group.iterrows():
                pid = int(r["player_external_id"])
                players[pid] = {
                    "rating": float(r["rating"]) if pd.notna(r["rating"]) else None,
                    "is_sub": bool(r["is_substitute"]) if pd.notna(r["is_substitute"]) else True,
                }

            self.team_history.setdefault(team_id, []).append({
                "match_id": match_id,
                "match_date": pd.Timestamp(match_date),
                "players": players,
            })

        # Sort each team's history by (date, match_id) — deterministic
        for team_id in self.team_history:
            self.team_history[team_id].sort(
                key=lambda x: (x["match_date"], x["match_id"])
            )
        logger.info(
            f"Team history built: {len(self.team_history)} teams, "
            f"{sum(len(v) for v in self.team_history.values())} team-match entries"
        )

    def _get_history_before(self, team_id: int, before_date, limit: int):
        """Get last `limit` matches strictly before `before_date`."""
        history = self.team_history.get(team_id, [])
        before_ts = pd.Timestamp(before_date)
        recent = [m for m in history if m["match_date"] < before_ts]
        return recent[-limit:]

    def _expected_xi(self, team_id: int, before_date, window: int = 15) -> list | None:
        """Top 11 players by start frequency in last `window` matches."""
        recent = self._get_history_before(team_id, before_date, window)
        if len(recent) < 3:
            return None

        start_counts: dict[int, int] = {}
        for match in recent:
            for pid, pdata in match["players"].items():
                if not pdata.get("is_sub", True):
                    start_counts[pid] = start_counts.get(pid, 0) + 1

        if len(start_counts) < 7:
            return None

        sorted_players = sorted(start_counts.items(), key=lambda x: (-x[1], x[0]))
        return [pid for pid, _ in sorted_players[:11]]

    def _compute_pts(self, player_ids: list, team_id: int, before_date,
                     limit: int = 10, min_ratings: int = 3) -> dict[int, float]:
        """
        PTS for each player: avg(rating) over last `limit` matches.
        Cascade fallback: player rolling → team P25 → global 6.50
        """
        recent = self._get_history_before(team_id, before_date, limit)

        # Collect all ratings for P25 fallback
        player_ratings: dict[int, list[float]] = {}
        all_ratings: list[float] = []
        for match in recent:
            for pid, pdata in match["players"].items():
                r = pdata.get("rating")
                if r is not None and r > 0:
                    player_ratings.setdefault(pid, []).append(r)
                    all_ratings.append(r)

        p25_team = float(np.percentile(all_ratings, 25)) if all_ratings else 6.5

        pts: dict[int, float] = {}
        for pid in player_ids:
            ratings = player_ratings.get(pid, [])
            if len(ratings) >= min_ratings:
                pts[pid] = float(np.mean(ratings))
            else:
                pts[pid] = p25_team
        return pts

    def compute(self, match_id: int, home_team_id: int, away_team_id: int,
                match_date) -> dict | None:
        """
        Compute talent_delta_diff for a single match.
        Returns dict with keys or None if data insufficient.
        """
        # XI real from lineups
        xi_real_home = self.lineups.get((match_id, home_team_id))
        xi_real_away = self.lineups.get((match_id, away_team_id))
        if not xi_real_home or not xi_real_away:
            return None

        # XI expected from start frequency
        xi_exp_home = self._expected_xi(home_team_id, match_date)
        xi_exp_away = self._expected_xi(away_team_id, match_date)
        if not xi_exp_home or not xi_exp_away:
            return None

        # PTS for both lineups
        pts_real_home = self._compute_pts(xi_real_home, home_team_id, match_date)
        pts_exp_home = self._compute_pts(xi_exp_home, home_team_id, match_date)
        pts_real_away = self._compute_pts(xi_real_away, away_team_id, match_date)
        pts_exp_away = self._compute_pts(xi_exp_away, away_team_id, match_date)

        mean_real_h = np.mean([pts_real_home[p] for p in xi_real_home])
        mean_exp_h = np.mean([pts_exp_home[p] for p in xi_exp_home])
        mean_real_a = np.mean([pts_real_away[p] for p in xi_real_away])
        mean_exp_a = np.mean([pts_exp_away[p] for p in xi_exp_away])

        home_delta = float(mean_real_h - mean_exp_h)
        away_delta = float(mean_real_a - mean_exp_a)

        return {
            "home_talent_delta": round(home_delta, 4),
            "away_talent_delta": round(away_delta, 4),
            "talent_delta_diff": round(home_delta - away_delta, 4),
            "shock_magnitude": round(max(abs(home_delta), abs(away_delta)), 4),
        }


def _get_db_conn():
    """Get a psycopg2 connection from DATABASE_URL."""
    import psycopg2
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set. Run: source .env")
        sys.exit(1)
    return psycopg2.connect(db_url)


def materialize(min_date: str):
    """Phase 1: batch compute talent_delta_diff and save to CSV."""
    conn = _get_db_conn()
    latam_ids = ",".join(str(x) for x in LATAM_LEAGUE_IDS)

    logger.info(f"Loading LATAM FT matches since {min_date}...")
    matches_df = pd.read_sql(
        f"""SELECT id AS match_id, date, league_id, home_team_id, away_team_id,
                   home_goals, away_goals, status
            FROM matches
            WHERE league_id IN ({latam_ids})
              AND status IN ('FT', 'AET', 'PEN')
              AND date >= '{min_date}'
            ORDER BY date, id""",
        conn,
    )
    logger.info(f"  {len(matches_df)} matches loaded")

    match_ids_str = ",".join(str(x) for x in matches_df["match_id"].tolist())

    logger.info("Loading lineups...")
    lineups_df = pd.read_sql(
        f"""SELECT match_id, team_id, starting_xi_ids
            FROM match_lineups
            WHERE match_id IN ({match_ids_str})""",
        conn,
    )
    logger.info(f"  {len(lineups_df)} lineup rows loaded")

    logger.info("Loading match_player_stats (this may take a moment)...")
    mps_df = pd.read_sql(
        f"""SELECT mps.match_id, mps.player_external_id, mps.team_id,
                   mps.match_date, mps.rating, mps.minutes,
                   mps.position, mps.is_substitute
            FROM match_player_stats mps
            WHERE mps.team_id IN (
                SELECT DISTINCT unnest(ARRAY[home_team_id, away_team_id])
                FROM matches WHERE league_id IN ({latam_ids})
            )
            AND mps.match_date >= '{min_date}'::date - INTERVAL '6 months'
            AND mps.rating IS NOT NULL""",
        conn,
    )
    logger.info(f"  {len(mps_df)} MPS rows loaded ({mps_df['team_id'].nunique()} teams)")

    conn.close()

    # Build computer and process
    computer = TalentDeltaComputer(mps_df, lineups_df)

    records = []
    n_ok, n_fail = 0, 0
    t0 = time.time()

    for i, row in matches_df.iterrows():
        result = computer.compute(
            match_id=int(row["match_id"]),
            home_team_id=int(row["home_team_id"]),
            away_team_id=int(row["away_team_id"]),
            match_date=row["date"],
        )
        rec = {
            "match_id": int(row["match_id"]),
            "date": str(pd.Timestamp(row["date"]).date()),
            "league_id": int(row["league_id"]),
            "home_team_id": int(row["home_team_id"]),
            "away_team_id": int(row["away_team_id"]),
            "home_goals": int(row["home_goals"]) if pd.notna(row["home_goals"]) else None,
            "away_goals": int(row["away_goals"]) if pd.notna(row["away_goals"]) else None,
        }
        if result:
            rec.update(result)
            n_ok += 1
        else:
            rec["talent_delta_diff"] = None
            n_fail += 1
        records.append(rec)

        if (i + 1) % 1000 == 0:
            logger.info(f"  Progress: {i + 1}/{len(matches_df)} ({n_ok} ok, {n_fail} fail)")

    elapsed = time.time() - t0
    df = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)

    logger.info(
        f"Materialized: {n_ok} with delta, {n_fail} without "
        f"({n_ok / (n_ok + n_fail) * 100:.1f}% coverage) in {elapsed:.1f}s"
    )
    logger.info(f"Saved to {CACHE_FILE}")

    # Per-league summary
    valid = df[df["talent_delta_diff"].notna()]
    for lid in sorted(LATAM_LEAGUE_IDS):
        n = len(valid[valid["league_id"] == lid])
        total = len(df[df["league_id"] == lid])
        name = LEAGUE_NAMES.get(lid, str(lid))
        logger.info(f"  {name:12s} ({lid}): {n:4d} / {total:4d} ({n / total * 100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Optimize β
# ═══════════════════════════════════════════════════════════════════════════

def load_latam_model():
    """Load LATAM v1.3.0 TwoStageEngine from model_snapshots."""
    from sqlalchemy import create_engine, text
    from app.ml.engine import TwoStageEngine

    db_url = os.environ.get("DATABASE_URL")
    db_engine = create_engine(db_url)

    with db_engine.connect() as conn:
        row = conn.execute(text("""
            SELECT model_blob, model_version, brier_score, id
            FROM model_snapshots
            WHERE model_version LIKE :pattern
            ORDER BY created_at DESC LIMIT 1
        """), {"pattern": LATAM_VERSION_PATTERN}).fetchone()

    db_engine.dispose()

    if not row or not row.model_blob:
        logger.error(f"No model snapshot found matching {LATAM_VERSION_PATTERN}")
        sys.exit(1)

    model = TwoStageEngine(model_version=row.model_version)
    if not model.load_from_bytes(row.model_blob):
        logger.error("Failed to deserialize model blob")
        sys.exit(1)

    logger.info(
        f"LATAM model loaded: {row.model_version}, brier={row.brier_score:.4f}, "
        f"snapshot_id={row.id}"
    )
    return model


def load_lab_csvs(min_date: str) -> pd.DataFrame:
    """Load and concatenate lab CSVs for all LATAM leagues."""
    dfs = []
    for lid in LATAM_LEAGUE_IDS:
        path = OUTPUT_DIR / f"lab_data_{lid}.csv"
        if not path.exists():
            logger.warning(f"Lab CSV missing: {path}")
            continue
        df = pd.read_csv(path)
        if "league_id" not in df.columns:
            df["league_id"] = lid
        df = df[df["date"] >= min_date]
        dfs.append(df)
        logger.info(f"  {LEAGUE_NAMES.get(lid, lid):12s}: {len(df)} rows")

    if not dfs:
        logger.error("No lab CSVs found!")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["date", "match_id"]).reset_index(drop=True)
    logger.info(f"Lab data: {len(combined)} rows across {len(dfs)} leagues")
    return combined


def compute_outcome(df: pd.DataFrame) -> np.ndarray:
    """0=home, 1=draw, 2=away from home_goals/away_goals."""
    outcomes = np.full(len(df), -1, dtype=int)
    for i, (hg, ag) in enumerate(zip(df["home_goals"], df["away_goals"])):
        if pd.isna(hg) or pd.isna(ag):
            outcomes[i] = -1
        elif hg > ag:
            outcomes[i] = 0
        elif hg == ag:
            outcomes[i] = 1
        else:
            outcomes[i] = 2
    return outcomes


def bootstrap_brier_ci(probs: np.ndarray, outcomes: np.ndarray,
                       n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float]:
    """Bootstrap 95% CI for Brier score."""
    n = len(outcomes)
    briers = np.empty(n_bootstrap)
    rng = np.random.default_rng(42)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        briers[b] = brier_multiclass(probs[idx], outcomes[idx])
    return float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))


def optimize_beta(min_date: str):
    """Phase 2: load data, optimize β, evaluate, report."""

    # ── Load cached talent_delta ──
    if not CACHE_FILE.exists():
        logger.error(f"Cache not found: {CACHE_FILE}. Run with --materialize first.")
        sys.exit(1)

    delta_df = pd.read_csv(CACHE_FILE)
    logger.info(f"Loaded delta cache: {len(delta_df)} rows")

    # ── Load lab CSVs ──
    logger.info("Loading lab CSVs...")
    lab_df = load_lab_csvs(min_date)

    # ── Load LATAM model ──
    model = load_latam_model()

    # ── Merge: lab features + talent_delta on match_id ──
    merged = lab_df.merge(
        delta_df[["match_id", "talent_delta_diff", "shock_magnitude",
                   "home_talent_delta", "away_talent_delta"]],
        on="match_id",
        how="inner",
        suffixes=("", "_delta"),
    )

    # Filter: talent_delta_diff not null + valid outcomes
    merged = merged[merged["talent_delta_diff"].notna()].copy()
    merged["outcome"] = compute_outcome(merged)
    merged = merged[merged["outcome"] >= 0].copy()

    # Sort deterministically (GDT SC2)
    merged = merged.sort_values(["date", "match_id"]).reset_index(drop=True)

    logger.info(
        f"Merged dataset: {len(merged)} matches with talent_delta_diff + outcomes"
    )

    # ── Generate base probs with LATAM v1.3.0 model ──
    feature_df = merged[FEATURES_LATAM].fillna(0).copy()
    base_probs = model.predict_proba(feature_df)

    merged["p_base_home"] = base_probs[:, 0]
    merged["p_base_draw"] = base_probs[:, 1]
    merged["p_base_away"] = base_probs[:, 2]

    # ── Compute market probs from odds (for Skill Score) ──
    mkt_probs = np.zeros((len(merged), 3))
    has_odds = np.ones(len(merged), dtype=bool)
    for i, (oh, od, oa) in enumerate(
        zip(merged["odds_home"], merged["odds_draw"], merged["odds_away"])
    ):
        if pd.notna(oh) and pd.notna(od) and pd.notna(oa) and oh > 1 and od > 1 and oa > 1:
            mkt_probs[i] = odds_to_probs(oh, od, oa)
        else:
            has_odds[i] = False

    # ── Temporal split 80/20 (ABE P0) ──
    n = len(merged)
    n_train = int(n * TRAIN_FRAC)
    train = merged.iloc[:n_train].copy()
    test = merged.iloc[n_train:].copy()

    train_probs = base_probs[:n_train]
    test_probs = base_probs[n_train:]
    train_d = merged["talent_delta_diff"].values[:n_train]
    test_d = merged["talent_delta_diff"].values[n_train:]
    train_y = merged["outcome"].values[:n_train]
    test_y = merged["outcome"].values[n_train:]
    train_mkt = mkt_probs[:n_train]
    test_mkt = mkt_probs[n_train:]
    train_has_odds = has_odds[:n_train]
    test_has_odds = has_odds[n_train:]

    logger.info(
        f"Split: train={n_train} ({train['date'].min()} → {train['date'].max()}) | "
        f"test={n - n_train} ({test['date'].min()} → {test['date'].max()})"
    )

    # ── Optimize β on train (scipy bounded) ──
    def objective(beta):
        p_adj = adjust_probs(train_probs.copy(), train_d, beta)
        return brier_multiclass(p_adj, train_y)

    result = minimize_scalar(objective, bounds=(0, BETA_MAX), method="bounded")
    beta_star = result.x
    brier_train_adj = result.fun

    # ── Sensitivity: scan β grid ──
    betas = np.linspace(0, min(5.0, BETA_MAX), 51)
    sensitivity = []
    for b in betas:
        p = adjust_probs(train_probs.copy(), train_d, b)
        sensitivity.append({"beta": round(b, 3), "brier_train": round(brier_multiclass(p, train_y), 6)})

    # ── Evaluate on all splits ──
    brier_train_base = brier_multiclass(train_probs, train_y)
    brier_test_base = brier_multiclass(test_probs, test_y)
    p_test_adj = adjust_probs(test_probs.copy(), test_d, beta_star)
    brier_test_adj = brier_multiclass(p_test_adj, test_y)

    # Market Brier (where odds available)
    brier_train_mkt = brier_multiclass(train_mkt[train_has_odds], train_y[train_has_odds])
    brier_test_mkt = brier_multiclass(test_mkt[test_has_odds], test_y[test_has_odds])

    # Bootstrap CI for test delta
    test_delta = brier_test_adj - brier_test_base
    ci_lo, ci_hi = bootstrap_brier_ci(p_test_adj, test_y)
    ci_lo_base, ci_hi_base = bootstrap_brier_ci(test_probs, test_y)

    # ── Per-league breakdown ──
    league_results = []
    for lid in sorted(LATAM_LEAGUE_IDS):
        mask_test = (test["league_id"].values == lid)
        mask_test_odds = mask_test & test_has_odds
        n_test_lid = mask_test.sum()
        if n_test_lid < 5:
            continue

        b_base = brier_multiclass(test_probs[mask_test], test_y[mask_test])
        b_adj = brier_multiclass(p_test_adj[mask_test], test_y[mask_test])
        b_mkt = brier_multiclass(test_mkt[mask_test_odds], test_y[mask_test_odds]) if mask_test_odds.sum() > 0 else float("nan")

        league_results.append({
            "league_id": lid,
            "name": LEAGUE_NAMES.get(lid, str(lid)),
            "n_test": int(n_test_lid),
            "brier_base": round(b_base, 4),
            "brier_adj": round(b_adj, 4),
            "brier_market": round(b_mkt, 4),
            "delta_brier": round(b_adj - b_base, 4),
            "skill_base_vs_mkt": round(skill_score(b_base, b_mkt) * 100, 2),
            "skill_adj_vs_mkt": round(skill_score(b_adj, b_mkt) * 100, 2),
        })

    # ── Console output ──
    print("\n" + "=" * 78)
    print("  VORP CALIBRATION LAB — Sprint 2, Camino B")
    print(f"  Model: {model.model_version} | β* = {beta_star:.4f}")
    print("=" * 78)

    print(f"\n  N total = {n} | train = {n_train} | test = {n - n_train}")
    print(f"  talent_delta_diff: mean={merged['talent_delta_diff'].mean():.4f}, "
          f"std={merged['talent_delta_diff'].std():.4f}, "
          f"min={merged['talent_delta_diff'].min():.4f}, "
          f"max={merged['talent_delta_diff'].max():.4f}")

    print(f"\n  {'':20s} {'TRAIN':>10s} {'TEST (OOS)':>12s}")
    print(f"  {'─' * 44}")
    print(f"  {'Brier base':20s} {brier_train_base:10.4f} {brier_test_base:12.4f}")
    print(f"  {'Brier adjusted':20s} {brier_train_adj:10.4f} {brier_test_adj:12.4f}")
    print(f"  {'Δ (adj - base)':20s} {brier_train_adj - brier_train_base:+10.4f} {test_delta:+12.4f}")
    print(f"  {'Brier market':20s} {brier_train_mkt:10.4f} {brier_test_mkt:12.4f}")
    print(f"  {'Skill base vs mkt':20s} {skill_score(brier_train_base, brier_train_mkt) * 100:+9.2f}% "
          f"{skill_score(brier_test_base, brier_test_mkt) * 100:+11.2f}%")
    print(f"  {'Skill adj vs mkt':20s} {skill_score(brier_train_adj, brier_train_mkt) * 100:+9.2f}% "
          f"{skill_score(brier_test_adj, brier_test_mkt) * 100:+11.2f}%")

    print(f"\n  Bootstrap 95% CI (test):")
    print(f"    Brier base: [{ci_lo_base:.4f}, {ci_hi_base:.4f}]")
    print(f"    Brier adj:  [{ci_lo:.4f}, {ci_hi:.4f}]")

    print(f"\n  {'─' * 78}")
    print(f"  PER-LEAGUE TEST BREAKDOWN (β* = {beta_star:.4f})")
    print(f"  {'─' * 78}")
    print(f"  {'Liga':<14s} {'N':>4s} {'B_base':>7s} {'B_adj':>7s} {'Δ':>7s} {'B_mkt':>7s} "
          f"{'Sk_base':>8s} {'Sk_adj':>8s}")
    for lr in league_results:
        marker = " **" if lr["skill_adj_vs_mkt"] > 0 else ""
        print(f"  {lr['name']:<14s} {lr['n_test']:4d} {lr['brier_base']:7.4f} "
              f"{lr['brier_adj']:7.4f} {lr['delta_brier']:+7.4f} {lr['brier_market']:7.4f} "
              f"{lr['skill_base_vs_mkt']:+7.2f}% {lr['skill_adj_vs_mkt']:+7.2f}%{marker}")

    print(f"\n  ** = Skill Score positivo (alpha sobre mercado)")
    print("=" * 78)

    # ── Detail CSV ──
    # Add adjusted probs to full dataset for export
    all_adj_probs = adjust_probs(base_probs.copy(), merged["talent_delta_diff"].values, beta_star)
    merged["p_adj_home"] = all_adj_probs[:, 0]
    merged["p_adj_draw"] = all_adj_probs[:, 1]
    merged["p_adj_away"] = all_adj_probs[:, 2]
    merged["split"] = ["train"] * n_train + ["test"] * (n - n_train)

    detail_cols = [
        "match_id", "date", "league_id", "outcome",
        "p_base_home", "p_base_draw", "p_base_away",
        "talent_delta_diff", "shock_magnitude",
        "home_talent_delta", "away_talent_delta",
        "p_adj_home", "p_adj_draw", "p_adj_away",
        "odds_home", "odds_draw", "odds_away",
        "split",
    ]
    # Only export columns that exist
    export_cols = [c for c in detail_cols if c in merged.columns]
    merged[export_cols].to_csv(DETAIL_FILE, index=False)
    logger.info(f"Detail CSV saved to {DETAIL_FILE}")

    # ── JSON results ──
    results = {
        "beta_star": round(beta_star, 6),
        "beta_max_bound": BETA_MAX,
        "train_frac": TRAIN_FRAC,
        "n_total": n,
        "n_train": n_train,
        "n_test": n - n_train,
        "model_version": model.model_version,
        "talent_delta_stats": {
            "mean": round(merged["talent_delta_diff"].mean(), 4),
            "std": round(merged["talent_delta_diff"].std(), 4),
            "min": round(merged["talent_delta_diff"].min(), 4),
            "max": round(merged["talent_delta_diff"].max(), 4),
        },
        "train": {
            "brier_base": round(brier_train_base, 6),
            "brier_adj": round(brier_train_adj, 6),
            "brier_market": round(brier_train_mkt, 6),
            "delta": round(brier_train_adj - brier_train_base, 6),
            "skill_base_pct": round(skill_score(brier_train_base, brier_train_mkt) * 100, 2),
            "skill_adj_pct": round(skill_score(brier_train_adj, brier_train_mkt) * 100, 2),
        },
        "test": {
            "brier_base": round(brier_test_base, 6),
            "brier_adj": round(brier_test_adj, 6),
            "brier_market": round(brier_test_mkt, 6),
            "delta": round(test_delta, 6),
            "skill_base_pct": round(skill_score(brier_test_base, brier_test_mkt) * 100, 2),
            "skill_adj_pct": round(skill_score(brier_test_adj, brier_test_mkt) * 100, 2),
            "bootstrap_ci_base": [round(ci_lo_base, 6), round(ci_hi_base, 6)],
            "bootstrap_ci_adj": [round(ci_lo, 6), round(ci_hi, 6)],
        },
        "per_league": league_results,
        "sensitivity": sensitivity,
    }

    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results JSON saved to {RESULT_FILE}")

    # ── Gating recommendation ──
    if abs(test_delta) < 0.001:
        print(f"\n  ⚠ OOS delta es ~0 ({test_delta:+.4f}). Considerar gating por shock_magnitude:")
        high_shock = merged[merged["shock_magnitude"] > merged["shock_magnitude"].quantile(0.75)]
        if len(high_shock) > 50:
            hs_test = high_shock[high_shock["split"] == "test"]
            if len(hs_test) > 10:
                hs_probs = base_probs[high_shock.index.intersection(range(n_train, n))]
                # Simplified: just report the observation
                print(f"    Top-quartile shock (>{merged['shock_magnitude'].quantile(0.75):.3f}): "
                      f"N={len(hs_test)} matches — posible gating target")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VORP Calibration Lab — Sprint 2")
    parser.add_argument("--materialize", action="store_true",
                        help="Phase 1: compute talent_delta_diff (run once)")
    parser.add_argument("--min-date", default=MIN_DATE_DEFAULT,
                        help=f"Minimum match date (default: {MIN_DATE_DEFAULT})")
    args = parser.parse_args()

    if args.materialize:
        materialize(args.min_date)
    else:
        optimize_beta(args.min_date)
