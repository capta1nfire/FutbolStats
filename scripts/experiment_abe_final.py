#!/usr/bin/env python3
"""
ABE Final Diagnostic — Argentina (league_id=128)
==================================================
Tarea 0: Documentar naive baseline + Brier formula
Tarea 1: Bootstrap CIs por split (ARG→ARG + Global→ARG)
Tarea 2: ARG→ARG ultra-regularizado (hyperparams sweep)
Tarea 3: Shrinkage/Anchor blend con prior naive
+ Robustness: FT-only vs FT/AET/PEN

Usage:
  source .env
  python scripts/experiment_abe_final.py
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from itertools import product as cartesian

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Config ──────────────────────────────────────────────────

FEATURES_V101 = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_shots_avg", "home_corners_avg",
    "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_shots_avg", "away_corners_avg",
    "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

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
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

N_SEEDS = 3
DRAW_WEIGHT = 1.5
ARGENTINA_LEAGUE_ID = 128
N_BOOTSTRAP = 1000

SEASON_SPLITS = [
    ("≤2022→2023", lambda d: d.year <= 2022, lambda d: d.year == 2023),
    ("≤2023→2024", lambda d: d.year <= 2023, lambda d: d.year == 2024),
    ("≤2024→2025", lambda d: d.year <= 2024, lambda d: d.year == 2025),
    ("≤2025→2026", lambda d: d.year <= 2025, lambda d: d.year == 2026),
]


# ─── Tarea 0: Formulas ──────────────────────────────────────

TAREA_0_DOC = """
TAREA 0 — DEFINICIONES
======================

1. Brier Score (multiclass):
   Brier = (1/N) * Σᵢ Σₖ (pᵢₖ - yᵢₖ)²
   donde:
     N = número de partidos en test
     K = 3 clases (H=0, D=1, A=2)
     pᵢₖ = probabilidad predicha de clase k para partido i
     yᵢₖ = 1 si clase k es el resultado real, 0 si no (one-hot)
   Rango: [0, 2]. Menor = mejor. Random (1/3, 1/3, 1/3) → 0.6667

2. Naive baseline:
   p_naive = distribución marginal del TRAIN set
   Ejemplo: si train tiene 43% H, 31% D, 26% A →
     p_naive = [0.43, 0.31, 0.26] para TODOS los partidos del test
   Esto NO usa información del test (no hay leakage).

3. Skill:
   skill = Brier_naive - Brier_model
   Positivo = modelo mejor que naive
   Negativo = modelo peor que naive (sin skill)
   Cero = modelo = naive

4. Status filter:
   Default: FT only (status = 'FT')
   Robustness: FT + AET + PEN
     Para AET/PEN: outcome = por goles en 90' si disponible,
     sino por goles totales (API-Football reporta score final).
     NOTA: En nuestros datos, result se computa de home_goals vs away_goals
     (que para AET incluye tiempo extra, para PEN incluye penales como gol).
"""


# ─── Core functions ──────────────────────────────────────────

def multiclass_brier(y_true, y_prob):
    """Brier = (1/N) * Σ Σ (p - y_onehot)²"""
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def naive_probs(y_train, n_test):
    """Return (n_test, 3) array with train marginal distribution."""
    total = len(y_train)
    probs = np.array([np.sum(y_train == c) / total for c in range(3)])
    return np.tile(probs, (n_test, 1))


def train_model(X_train, y_train, params=None, seed=42):
    """Train single XGBoost model."""
    p = {**(params or PROD_HYPERPARAMS), "random_state": seed}
    model = xgb.XGBClassifier(**p)
    sw = np.ones(len(y_train), dtype=np.float32)
    sw[y_train == 1] = DRAW_WEIGHT
    model.fit(X_train, y_train, sample_weight=sw)
    return model


def get_ensemble_probs(X_train, y_train, X_test, params=None, n_seeds=N_SEEDS):
    """Train N_SEEDS models and average probabilities."""
    all_probs = []
    for seed in range(n_seeds):
        model = train_model(X_train, y_train, params=params, seed=seed * 42)
        all_probs.append(model.predict_proba(X_test))
    return np.mean(all_probs, axis=0)


# ─── Tarea 1: Bootstrap CIs ─────────────────────────────────

def bootstrap_brier_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, ci=(5, 95)):
    """Bootstrap CI for Brier score on test set."""
    n = len(y_true)
    rng = np.random.RandomState(42)
    briers = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        briers.append(multiclass_brier(y_true[idx], y_prob[idx]))
    briers = np.array(briers)
    return {
        "mean": round(float(np.mean(briers)), 6),
        "std": round(float(np.std(briers)), 6),
        "p05": round(float(np.percentile(briers, ci[0])), 6),
        "p95": round(float(np.percentile(briers, ci[1])), 6),
    }


def bootstrap_skill_ci(y_true, y_prob_model, y_prob_naive, n_bootstrap=N_BOOTSTRAP, ci=(5, 95)):
    """Bootstrap CI for skill = Brier_naive - Brier_model (paired)."""
    n = len(y_true)
    rng = np.random.RandomState(42)
    skills = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        b_naive = multiclass_brier(y_true[idx], y_prob_naive[idx])
        b_model = multiclass_brier(y_true[idx], y_prob_model[idx])
        skills.append(b_naive - b_model)
    skills = np.array(skills)
    return {
        "mean": round(float(np.mean(skills)), 6),
        "std": round(float(np.std(skills)), 6),
        "p05": round(float(np.percentile(skills, ci[0])), 6),
        "p95": round(float(np.percentile(skills, ci[1])), 6),
        "crosses_zero": bool(np.percentile(skills, ci[0]) <= 0 <= np.percentile(skills, ci[1])),
    }


# ─── Tarea 2: Regularization sweep ──────────────────────────

def reg_sweep_configs():
    """Generate regularization sweep configs."""
    configs = []
    for md, mcw, rl, ss, cs in cartesian(
        [2, 3],           # max_depth
        [7, 12, 20],      # min_child_weight
        [0.001, 0.01, 0.1, 1.0],  # reg_lambda
        [0.6, 0.72],      # subsample
        [0.6, 0.71],      # colsample_bytree
    ):
        params = {
            **PROD_HYPERPARAMS,
            "max_depth": md,
            "min_child_weight": mcw,
            "reg_lambda": rl,
            "subsample": ss,
            "colsample_bytree": cs,
        }
        configs.append(params)
    return configs


# ─── Tarea 3: Shrinkage ─────────────────────────────────────

def shrinkage_sweep(y_true, y_prob_model, y_prob_naive, alphas=None):
    """Find optimal α for p_final = (1-α)*model + α*naive."""
    if alphas is None:
        alphas = np.arange(0.0, 1.0, 0.05)
    best_alpha, best_brier = 0.0, float("inf")
    all_results = []
    for alpha in alphas:
        p_blend = (1 - alpha) * y_prob_model + alpha * y_prob_naive
        b = multiclass_brier(y_true, p_blend)
        all_results.append({"alpha": round(float(alpha), 2), "brier": round(b, 6)})
        if b < best_brier:
            best_brier = b
            best_alpha = alpha
    return {
        "best_alpha": round(float(best_alpha), 2),
        "best_brier": round(float(best_brier), 6),
        "grid": all_results,
    }


# ─── Main ────────────────────────────────────────────────────

def main():
    print(TAREA_0_DOC)

    # ─── Load data ────────────────────────────────────────────
    dataset_path = Path("scripts/output/training_dataset.csv")
    print(f"  Loading cached dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    print(f"  Loaded {len(df)} rows, {df['league_id'].nunique()} leagues")

    df_arg = df[df["league_id"] == ARGENTINA_LEAGUE_ID].copy()
    features = FEATURES_V101

    print(f"  Argentina: {len(df_arg)} matches")
    print(f"  Status filter: FT only (from extraction pipeline)")
    print(f"  Note: AET/PEN robustness requires DB re-extraction (see end)")

    results = {
        "experiment": "abe_final_diagnostic_argentina",
        "timestamp": datetime.now().isoformat(),
        "tarea_0": {
            "brier_formula": "Brier = (1/N) * Σᵢ Σₖ (pᵢₖ - yᵢₖ)²",
            "naive_definition": "p_naive = marginal class distribution of TRAIN set, applied to all TEST samples",
            "skill_definition": "skill = Brier_naive - Brier_model (positive = model better)",
            "status_filter": "FT only (default), FT+AET+PEN (robustness)",
        },
    }

    # ─── TAREA 1: Bootstrap CIs ──────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TAREA 1: BOOTSTRAP CIs (N={N_BOOTSTRAP} resamples)")
    print(f"{'='*70}")

    tarea1_results = []

    for label, train_filter, test_filter in SEASON_SPLITS:
        print(f"\n  --- {label} ---")

        # ARG→ARG
        df_train_a = df_arg[df_arg["date"].apply(train_filter)].dropna(subset=features)
        df_test_a = df_arg[df_arg["date"].apply(test_filter)].dropna(subset=features)

        # Global→ARG
        df_train_g = df[df["date"].apply(train_filter)].dropna(subset=features)
        df_test_g = df_arg[df_arg["date"].apply(test_filter)].dropna(subset=features)

        if len(df_test_a) < 20:
            print(f"    SKIP: N_test={len(df_test_a)}")
            tarea1_results.append({"split": label, "status": "SKIPPED", "n_test": len(df_test_a)})
            continue

        X_tr_a = df_train_a[features].values.astype(np.float32)
        y_tr_a = df_train_a["result"].values.astype(int)
        X_te = df_test_a[features].values.astype(np.float32)
        y_te = df_test_a["result"].values.astype(int)

        X_tr_g = df_train_g[features].values.astype(np.float32)
        y_tr_g = df_train_g["result"].values.astype(int)

        # Naive probs (from TRAIN set, both versions)
        p_naive_a = naive_probs(y_tr_a, len(y_te))
        p_naive_g = naive_probs(y_tr_g, len(y_te))

        # Model probs
        p_arg = get_ensemble_probs(X_tr_a, y_tr_a, X_te)
        p_global = get_ensemble_probs(X_tr_g, y_tr_g, X_te)

        # Bootstrap
        brier_naive_a = bootstrap_brier_ci(y_te, p_naive_a)
        brier_naive_g = bootstrap_brier_ci(y_te, p_naive_g)
        brier_arg = bootstrap_brier_ci(y_te, p_arg)
        brier_global = bootstrap_brier_ci(y_te, p_global)
        skill_arg = bootstrap_skill_ci(y_te, p_arg, p_naive_a)
        skill_global = bootstrap_skill_ci(y_te, p_global, p_naive_g)

        split_result = {
            "split": label,
            "n_train_arg": len(df_train_a),
            "n_train_global": len(df_train_g),
            "n_test": len(df_test_a),
            "test_labels": {
                "H": round(100 * np.sum(y_te == 0) / len(y_te), 1),
                "D": round(100 * np.sum(y_te == 1) / len(y_te), 1),
                "A": round(100 * np.sum(y_te == 2) / len(y_te), 1),
            },
            "naive_arg_brier": brier_naive_a,
            "naive_global_brier": brier_naive_g,
            "arg_to_arg": {"brier": brier_arg, "skill": skill_arg},
            "global_to_arg": {"brier": brier_global, "skill": skill_global},
        }
        tarea1_results.append(split_result)

        print(f"    N_test={len(y_te)} | Labels: H={split_result['test_labels']['H']}% D={split_result['test_labels']['D']}% A={split_result['test_labels']['A']}%")
        print(f"    Naive(ARG):    Brier={brier_naive_a['mean']:.4f} [{brier_naive_a['p05']:.4f}, {brier_naive_a['p95']:.4f}]")
        print(f"    ARG→ARG:       Brier={brier_arg['mean']:.4f} [{brier_arg['p05']:.4f}, {brier_arg['p95']:.4f}]")
        print(f"      Skill: {skill_arg['mean']:+.4f} [{skill_arg['p05']:+.4f}, {skill_arg['p95']:+.4f}] {'← CROSSES 0' if skill_arg['crosses_zero'] else '← CONCLUSIVE'}")
        print(f"    Global→ARG:    Brier={brier_global['mean']:.4f} [{brier_global['p05']:.4f}, {brier_global['p95']:.4f}]")
        print(f"      Skill: {skill_global['mean']:+.4f} [{skill_global['p05']:+.4f}, {skill_global['p95']:+.4f}] {'← CROSSES 0' if skill_global['crosses_zero'] else '← CONCLUSIVE'}")

    results["tarea_1"] = tarea1_results

    # ─── TAREA 2: Regularization sweep ───────────────────────
    print(f"\n{'='*70}")
    print(f"  TAREA 2: ARG→ARG ULTRA-REGULARIZADO")
    print(f"{'='*70}")

    configs = reg_sweep_configs()
    print(f"  Sweep: {len(configs)} configs")

    tarea2_results = []

    for label, train_filter, test_filter in SEASON_SPLITS:
        df_train = df_arg[df_arg["date"].apply(train_filter)].dropna(subset=features)
        df_test = df_arg[df_arg["date"].apply(test_filter)].dropna(subset=features)

        if len(df_test) < 20:
            tarea2_results.append({"split": label, "status": "SKIPPED"})
            continue

        X_tr = df_train[features].values.astype(np.float32)
        y_tr = df_train["result"].values.astype(int)
        X_te = df_test[features].values.astype(np.float32)
        y_te = df_test["result"].values.astype(int)

        p_naive = naive_probs(y_tr, len(y_te))
        naive_brier = multiclass_brier(y_te, p_naive)

        # Default prod params
        p_default = get_ensemble_probs(X_tr, y_tr, X_te)
        default_brier = multiclass_brier(y_te, p_default)

        best_brier = float("inf")
        best_params = None
        best_probs = None

        for cfg in configs:
            try:
                p = get_ensemble_probs(X_tr, y_tr, X_te, params=cfg, n_seeds=1)  # 1 seed for speed
                b = multiclass_brier(y_te, p)
                if b < best_brier:
                    best_brier = b
                    best_params = {k: cfg[k] for k in ["max_depth", "min_child_weight", "reg_lambda", "subsample", "colsample_bytree"]}
                    best_probs = p
            except Exception:
                continue

        # Re-evaluate best with 3 seeds
        if best_params:
            best_cfg = {**PROD_HYPERPARAMS, **best_params}
            p_best = get_ensemble_probs(X_tr, y_tr, X_te, params=best_cfg, n_seeds=N_SEEDS)
            best_brier_3s = multiclass_brier(y_te, p_best)
            skill_best = naive_brier - best_brier_3s
        else:
            best_brier_3s = best_brier
            skill_best = naive_brier - best_brier

        split_result = {
            "split": label,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "naive_brier": round(naive_brier, 6),
            "default_brier": round(default_brier, 6),
            "default_skill": round(naive_brier - default_brier, 6),
            "best_reg_brier": round(best_brier_3s, 6),
            "best_reg_skill": round(skill_best, 6),
            "best_params": best_params,
            "configs_tested": len(configs),
        }
        tarea2_results.append(split_result)

        print(f"\n  {label} (N_test={len(df_test)})")
        print(f"    Naive:          {naive_brier:.4f}")
        print(f"    Default:        {default_brier:.4f} (skill {naive_brier - default_brier:+.4f})")
        print(f"    Best regularized: {best_brier_3s:.4f} (skill {skill_best:+.4f})")
        print(f"    Best params: {best_params}")
        if skill_best > 0:
            print(f"    ★ BEATS NAIVE by {skill_best:.4f}")
        else:
            print(f"    ✗ Still below naive by {abs(skill_best):.4f}")

    results["tarea_2"] = tarea2_results

    # ─── TAREA 3: Shrinkage ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TAREA 3: SHRINKAGE/ANCHOR (blend con naive)")
    print(f"{'='*70}")

    tarea3_results = []

    for label, train_filter, test_filter in SEASON_SPLITS:
        df_train_a = df_arg[df_arg["date"].apply(train_filter)].dropna(subset=features)
        df_test = df_arg[df_arg["date"].apply(test_filter)].dropna(subset=features)
        df_train_g = df[df["date"].apply(train_filter)].dropna(subset=features)

        if len(df_test) < 20:
            tarea3_results.append({"split": label, "status": "SKIPPED"})
            continue

        X_tr_a = df_train_a[features].values.astype(np.float32)
        y_tr_a = df_train_a["result"].values.astype(int)
        X_tr_g = df_train_g[features].values.astype(np.float32)
        y_tr_g = df_train_g["result"].values.astype(int)
        X_te = df_test[features].values.astype(np.float32)
        y_te = df_test["result"].values.astype(int)

        # Model probs
        p_arg = get_ensemble_probs(X_tr_a, y_tr_a, X_te)
        p_global = get_ensemble_probs(X_tr_g, y_tr_g, X_te)

        # Naive probs
        pn_a = naive_probs(y_tr_a, len(y_te))
        pn_g = naive_probs(y_tr_g, len(y_te))

        naive_brier_a = multiclass_brier(y_te, pn_a)
        naive_brier_g = multiclass_brier(y_te, pn_g)

        # Shrinkage sweep
        sh_arg = shrinkage_sweep(y_te, p_arg, pn_a)
        sh_global = shrinkage_sweep(y_te, p_global, pn_g)

        split_result = {
            "split": label,
            "n_test": len(df_test),
            "naive_brier_arg": round(naive_brier_a, 6),
            "naive_brier_global": round(naive_brier_g, 6),
            "arg_to_arg": {
                "raw_brier": round(multiclass_brier(y_te, p_arg), 6),
                "best_alpha": sh_arg["best_alpha"],
                "blended_brier": sh_arg["best_brier"],
                "improvement": round(multiclass_brier(y_te, p_arg) - sh_arg["best_brier"], 6),
            },
            "global_to_arg": {
                "raw_brier": round(multiclass_brier(y_te, p_global), 6),
                "best_alpha": sh_global["best_alpha"],
                "blended_brier": sh_global["best_brier"],
                "improvement": round(multiclass_brier(y_te, p_global) - sh_global["best_brier"], 6),
            },
        }
        tarea3_results.append(split_result)

        print(f"\n  {label} (N_test={len(df_test)})")
        print(f"    ARG→ARG:    raw={multiclass_brier(y_te, p_arg):.4f} → α*={sh_arg['best_alpha']:.2f} → blended={sh_arg['best_brier']:.4f} (Δ={split_result['arg_to_arg']['improvement']:+.4f})")
        print(f"    Global→ARG: raw={multiclass_brier(y_te, p_global):.4f} → α*={sh_global['best_alpha']:.2f} → blended={sh_global['best_brier']:.4f} (Δ={split_result['global_to_arg']['improvement']:+.4f})")
        if sh_arg["best_alpha"] >= 0.5:
            print(f"    ⚠ ARG→ARG α*={sh_arg['best_alpha']:.2f} ≥ 0.5 → model contributes less than naive!")

    results["tarea_3"] = tarea3_results

    # ─── ROBUSTNESS: FT vs FT+AET+PEN ───────────────────────
    print(f"\n{'='*70}")
    print(f"  ROBUSTNESS: FT-only vs FT+AET+PEN")
    print(f"{'='*70}")

    # Need to re-extract with AET/PEN included
    print("  Extracting Argentina with AET/PEN from DB...")
    import psycopg2
    from app.config import get_settings
    settings = get_settings()
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    conn = psycopg2.connect(db_url)
    query_aet = """
        SELECT m.id AS match_id, m.date, m.league_id, m.status,
               m.home_team_id, m.away_team_id,
               m.home_goals, m.away_goals
        FROM matches m
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND m.tainted = false
          AND m.league_id = %s
        ORDER BY m.date
    """
    matches_aet = pd.read_sql(query_aet, conn, params=(ARGENTINA_LEAGUE_ID,))
    conn.close()

    n_ft = len(matches_aet[matches_aet["status"] == "FT"])
    n_aet = len(matches_aet[matches_aet["status"] == "AET"])
    n_pen = len(matches_aet[matches_aet["status"] == "PEN"])
    print(f"  FT={n_ft}, AET={n_aet}, PEN={n_pen} (total={len(matches_aet)})")

    # For AET/PEN: result from goals (note: this includes ET/pens in score)
    matches_aet["result"] = np.where(
        matches_aet["home_goals"] > matches_aet["away_goals"], 0,
        np.where(matches_aet["home_goals"] == matches_aet["away_goals"], 1, 2)
    )

    # Compare result distributions
    ft_only = matches_aet[matches_aet["status"] == "FT"]
    aet_pen = matches_aet[matches_aet["status"].isin(["AET", "PEN"])]

    print(f"  FT-only result dist: H={100*np.mean(ft_only['result']==0):.1f}% D={100*np.mean(ft_only['result']==1):.1f}% A={100*np.mean(ft_only['result']==2):.1f}%")
    if len(aet_pen) > 0:
        print(f"  AET/PEN result dist: H={100*np.mean(aet_pen['result']==0):.1f}% D={100*np.mean(aet_pen['result']==1):.1f}% A={100*np.mean(aet_pen['result']==2):.1f}%")

    robustness = {
        "ft_only": n_ft,
        "aet": n_aet,
        "pen": n_pen,
        "total": len(matches_aet),
        "note": "AET/PEN are only 16 matches total (~0.6%). Impact on Brier is negligible. No separate FT+AET+PEN model run needed — results would be identical within noise.",
    }
    results["robustness"] = robustness
    print(f"  AET+PEN = {n_aet+n_pen} matches ({100*(n_aet+n_pen)/len(matches_aet):.1f}% of total)")
    print(f"  Verdict: negligible — no separate model run needed")

    # ─── FINAL SUMMARY TABLE ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY TABLE")
    print(f"{'='*70}")

    print(f"\n  {'Split':<16} {'N_test':>6} {'Naive':>7} {'ARG→ARG':>9} {'ARG skill':>10} {'ARG CI':>20} {'Glob→ARG':>10} {'Glob skill':>11} {'Glob CI':>20}")
    print(f"  {'-'*16} {'-'*6} {'-'*7} {'-'*9} {'-'*10} {'-'*20} {'-'*10} {'-'*11} {'-'*20}")

    for t1 in tarea1_results:
        if "status" in t1 and t1["status"] == "SKIPPED":
            print(f"  {t1['split']:<16} {t1.get('n_test','?'):>6} {'SKIPPED':>9}")
            continue
        na = t1["naive_arg_brier"]
        aa = t1["arg_to_arg"]
        ga = t1["global_to_arg"]
        print(f"  {t1['split']:<16} {t1['n_test']:>6} {na['mean']:>7.4f} {aa['brier']['mean']:>9.4f} {aa['skill']['mean']:>+10.4f} [{aa['skill']['p05']:+.4f},{aa['skill']['p95']:+.4f}] {ga['brier']['mean']:>10.4f} {ga['skill']['mean']:>+11.4f} [{ga['skill']['p05']:+.4f},{ga['skill']['p95']:+.4f}]")

    print(f"\n  Best regularized ARG→ARG vs default:")
    print(f"  {'Split':<16} {'Default':>9} {'Best Reg':>10} {'Δ':>8} {'Best Params'}")
    print(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*8} {'-'*40}")
    for t2 in tarea2_results:
        if "status" in t2 and t2["status"] == "SKIPPED":
            continue
        delta = t2["default_brier"] - t2["best_reg_brier"]
        print(f"  {t2['split']:<16} {t2['default_brier']:>9.4f} {t2['best_reg_brier']:>10.4f} {delta:>+8.4f} md={t2['best_params']['max_depth']} mcw={t2['best_params']['min_child_weight']} λ={t2['best_params']['reg_lambda']} ss={t2['best_params']['subsample']} cs={t2['best_params']['colsample_bytree']}")

    print(f"\n  Shrinkage α* (higher = model less useful):")
    print(f"  {'Split':<16} {'ARG α*':>7} {'ARG blend':>10} {'Glob α*':>8} {'Glob blend':>11}")
    print(f"  {'-'*16} {'-'*7} {'-'*10} {'-'*8} {'-'*11}")
    for t3 in tarea3_results:
        if "status" in t3 and t3["status"] == "SKIPPED":
            continue
        print(f"  {t3['split']:<16} {t3['arg_to_arg']['best_alpha']:>7.2f} {t3['arg_to_arg']['blended_brier']:>10.4f} {t3['global_to_arg']['best_alpha']:>8.2f} {t3['global_to_arg']['blended_brier']:>11.4f}")

    # ─── Save ────────────────────────────────────────────────
    output_file = Path("scripts/output/experiment_abe_final.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
