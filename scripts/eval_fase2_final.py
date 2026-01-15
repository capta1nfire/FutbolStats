#!/usr/bin/env python3
"""
FASE 2: Final Evaluation with Optimal Configuration

Based on threshold sweep findings:
- threshold=0.32 gives precision=0.309 > prevalence, but recall=4.5%
- threshold=0.30 gives precision=0.262 > prevalence, recall=30.9%

This script finds the optimal threshold balancing precision and recall,
with the constraint that calibration (brier/logloss) must not degrade.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["DATABASE_URL"] = "postgresql://postgres:hzvozcXijUpblVrQshuowYcEGwZnMrfO@maglev.proxy.rlwy.net:24997/railway"

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from app.database import get_async_session
from app.features.engineering import FeatureEngineer


BASE_FEATURES = [
    "home_goals_scored_avg", "home_goals_conceded_avg", "home_shots_avg",
    "home_corners_avg", "home_rest_days", "home_matches_played",
    "away_goals_scored_avg", "away_goals_conceded_avg", "away_shots_avg",
    "away_corners_avg", "away_rest_days", "away_matches_played",
    "goal_diff_avg", "rest_diff",
    "abs_attack_diff", "abs_defense_diff", "abs_strength_gap",
]

PARAMS_STAGE1 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "random_state": 42,
}

PARAMS_STAGE2 = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

PARAMS_BASELINE = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.0283,
    "n_estimators": 114,
    "min_child_weight": 7,
    "subsample": 0.72,
    "colsample_bytree": 0.71,
    "random_state": 42,
}


def prepare_features(df: pd.DataFrame, features: list) -> np.ndarray:
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features].fillna(0).values


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "odds_draw" in df.columns:
        df["implied_draw_raw"] = 1 / df["odds_draw"].replace(0, np.nan)
        df["implied_home_raw"] = 1 / df["odds_home"].replace(0, np.nan)
        df["implied_away_raw"] = 1 / df["odds_away"].replace(0, np.nan)
        total = df["implied_draw_raw"] + df["implied_home_raw"] + df["implied_away_raw"]
        df["implied_draw"] = df["implied_draw_raw"] / total
        df["implied_draw"] = df["implied_draw"].fillna(0.25)
    else:
        df["implied_draw"] = 0.25
    return df


class TwoStageModel:
    def __init__(self, draw_weight: float = 1.2):
        self.stage1 = None
        self.stage2 = None
        self.draw_weight = draw_weight
        self.features_s1 = BASE_FEATURES + ["implied_draw"]
        self.features_s2 = BASE_FEATURES

    def fit(self, df_train: pd.DataFrame):
        y_train = df_train["result"].values
        y_draw = (y_train == 1).astype(int)
        nondraw_mask = y_train != 1
        y_home = (y_train[nondraw_mask] == 0).astype(int)

        X_s1 = prepare_features(df_train, self.features_s1)
        X_s2 = prepare_features(df_train[nondraw_mask], self.features_s2)

        sample_weight = np.ones(len(y_draw), dtype=np.float32)
        sample_weight[y_draw == 1] = self.draw_weight

        self.stage1 = xgb.XGBClassifier(**PARAMS_STAGE1)
        self.stage1.fit(X_s1, y_draw, sample_weight=sample_weight, verbose=False)

        self.stage2 = xgb.XGBClassifier(**PARAMS_STAGE2)
        self.stage2.fit(X_s2, y_home, verbose=False)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_s1 = prepare_features(df, self.features_s1)
        X_s2 = prepare_features(df, self.features_s2)

        p_draw = self.stage1.predict_proba(X_s1)[:, 1]
        p_home_given_nondraw = self.stage2.predict_proba(X_s2)[:, 1]

        p_home = (1 - p_draw) * p_home_given_nondraw
        p_away = (1 - p_draw) * (1 - p_home_given_nondraw)

        return np.column_stack([p_home, p_draw, p_away])

    def predict_with_threshold(self, df: pd.DataFrame, draw_threshold: float = None) -> tuple:
        proba = self.predict_proba(df)
        if draw_threshold is None:
            y_pred = np.argmax(proba, axis=1)
        else:
            y_pred = np.argmax(proba, axis=1)
            draw_override = proba[:, 1] > draw_threshold
            y_pred[draw_override] = 1
        return y_pred, proba


async def main():
    print("=" * 80)
    print("FASE 2: FINAL EVALUATION")
    print("=" * 80)

    # Load data
    async for session in get_async_session():
        fe = FeatureEngineer(session)
        df = await fe.build_training_dataset()
        break

    df = df.sort_values("date").reset_index(drop=True)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = add_derived_features(df)

    # Temporal splits
    cutoff_90 = datetime.utcnow() - timedelta(days=90)
    cutoff_30 = datetime.utcnow() - timedelta(days=30)

    df_train = df[df["date_dt"] < cutoff_90].reset_index(drop=True)
    df_test_90 = df[df["date_dt"] >= cutoff_90].reset_index(drop=True)
    df_test_30 = df[df["date_dt"] >= cutoff_30].reset_index(drop=True)

    print(f"\nDataset: {len(df)} samples")
    print(f"Train: {len(df_train)} samples")
    print(f"Test 90d: {len(df_test_90)} samples")
    print(f"Test 30d: {len(df_test_30)} samples")

    # Train baseline
    X_train_base = prepare_features(df_train, BASE_FEATURES)
    y_train = df_train["result"].values

    baseline = xgb.XGBClassifier(**PARAMS_BASELINE)
    baseline.fit(X_train_base, y_train, verbose=False)

    # Train two-stage
    two_stage = TwoStageModel(draw_weight=1.2)
    two_stage.fit(df_train)

    # Optimal threshold from previous analysis
    OPTIMAL_THRESHOLD = 0.30  # Balance between precision and recall

    # ========================================
    # EVALUATION ON 90-DAY HOLDOUT
    # ========================================
    print("\n" + "=" * 80)
    print("EVALUACIÓN: HOLDOUT 90 DÍAS")
    print("=" * 80)

    y_test = df_test_90["result"].values
    X_test_base = prepare_features(df_test_90, BASE_FEATURES)

    # Baseline predictions
    baseline_proba = baseline.predict_proba(X_test_base)
    baseline_pred = np.argmax(baseline_proba, axis=1)

    # Two-stage predictions (with threshold)
    ts_pred, ts_proba = two_stage.predict_with_threshold(df_test_90, OPTIMAL_THRESHOLD)

    # Metrics
    def full_metrics(name, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        ll = log_loss(y_true, y_proba)
        brier = np.mean([
            brier_score_loss((y_true == c).astype(int), y_proba[:, c])
            for c in range(3)
        ])

        draw_true = (y_true == 1).astype(int)
        draw_pred = (y_pred == 1)

        pr_auc_draw = average_precision_score(draw_true, y_proba[:, 1])
        roc_auc_draw = roc_auc_score(draw_true, y_proba[:, 1])

        n_draw_pred = draw_pred.sum()
        correct_draw = (draw_pred & (y_true == 1)).sum()
        precision_draw = correct_draw / n_draw_pred if n_draw_pred > 0 else 0
        recall_draw = correct_draw / draw_true.sum() if draw_true.sum() > 0 else 0

        return {
            "name": name,
            "accuracy": acc,
            "log_loss": ll,
            "brier": brier,
            "draw_pred_n": n_draw_pred,
            "draw_pred_pct": n_draw_pred / len(y_pred),
            "precision_draw": precision_draw,
            "recall_draw": recall_draw,
            "pr_auc_draw": pr_auc_draw,
            "roc_auc_draw": roc_auc_draw,
            "draw_prevalence": draw_true.mean(),
        }

    base_m = full_metrics("v1.0.0 Baseline", y_test, baseline_pred, baseline_proba)
    ts_m = full_metrics(f"Two-Stage (th={OPTIMAL_THRESHOLD})", y_test, ts_pred, ts_proba)

    print(f"\n{'Métrica':<25} {'Baseline':>12} {'Two-Stage':>12} {'Delta':>12}")
    print("-" * 65)
    print(f"{'Accuracy':<25} {base_m['accuracy']:>12.4f} {ts_m['accuracy']:>12.4f} {ts_m['accuracy']-base_m['accuracy']:>+12.4f}")
    print(f"{'LogLoss':<25} {base_m['log_loss']:>12.4f} {ts_m['log_loss']:>12.4f} {ts_m['log_loss']-base_m['log_loss']:>+12.4f}")
    print(f"{'Brier Global':<25} {base_m['brier']:>12.4f} {ts_m['brier']:>12.4f} {ts_m['brier']-base_m['brier']:>+12.4f}")
    print(f"{'Draw Predictions':<25} {base_m['draw_pred_n']:>12} {ts_m['draw_pred_n']:>12} {ts_m['draw_pred_n']-base_m['draw_pred_n']:>+12}")
    print(f"{'Draw Pred %':<25} {base_m['draw_pred_pct']:>12.1%} {ts_m['draw_pred_pct']:>12.1%} {ts_m['draw_pred_pct']-base_m['draw_pred_pct']:>+12.1%}")
    print(f"{'Draw Precision':<25} {base_m['precision_draw']:>12.3f} {ts_m['precision_draw']:>12.3f} {ts_m['precision_draw']-base_m['precision_draw']:>+12.3f}")
    print(f"{'Draw Recall':<25} {base_m['recall_draw']:>12.3f} {ts_m['recall_draw']:>12.3f} {ts_m['recall_draw']-base_m['recall_draw']:>+12.3f}")
    print(f"{'Draw PR-AUC':<25} {base_m['pr_auc_draw']:>12.4f} {ts_m['pr_auc_draw']:>12.4f} {ts_m['pr_auc_draw']-base_m['pr_auc_draw']:>+12.4f}")
    print(f"{'Draw ROC-AUC':<25} {base_m['roc_auc_draw']:>12.4f} {ts_m['roc_auc_draw']:>12.4f} {ts_m['roc_auc_draw']-base_m['roc_auc_draw']:>+12.4f}")

    # Confusion matrices
    print("\n" + "-" * 65)
    print("CONFUSION MATRIX: Baseline")
    cm_base = confusion_matrix(y_test, baseline_pred)
    print(f"{'':>15} {'Pred H':>10} {'Pred D':>10} {'Pred A':>10}")
    print(f"{'True Home':<15} {cm_base[0,0]:>10} {cm_base[0,1]:>10} {cm_base[0,2]:>10}")
    print(f"{'True Draw':<15} {cm_base[1,0]:>10} {cm_base[1,1]:>10} {cm_base[1,2]:>10}")
    print(f"{'True Away':<15} {cm_base[2,0]:>10} {cm_base[2,1]:>10} {cm_base[2,2]:>10}")

    print("\nCONFUSION MATRIX: Two-Stage")
    cm_ts = confusion_matrix(y_test, ts_pred)
    print(f"{'':>15} {'Pred H':>10} {'Pred D':>10} {'Pred A':>10}")
    print(f"{'True Home':<15} {cm_ts[0,0]:>10} {cm_ts[0,1]:>10} {cm_ts[0,2]:>10}")
    print(f"{'True Draw':<15} {cm_ts[1,0]:>10} {cm_ts[1,1]:>10} {cm_ts[1,2]:>10}")
    print(f"{'True Away':<15} {cm_ts[2,0]:>10} {cm_ts[2,1]:>10} {cm_ts[2,2]:>10}")

    # ========================================
    # EVALUATION ON 30-DAY HOLDOUT
    # ========================================
    print("\n" + "=" * 80)
    print("EVALUACIÓN: HOLDOUT 30 DÍAS")
    print("=" * 80)

    y_test_30 = df_test_30["result"].values
    X_test_30_base = prepare_features(df_test_30, BASE_FEATURES)

    baseline_proba_30 = baseline.predict_proba(X_test_30_base)
    baseline_pred_30 = np.argmax(baseline_proba_30, axis=1)
    ts_pred_30, ts_proba_30 = two_stage.predict_with_threshold(df_test_30, OPTIMAL_THRESHOLD)

    base_m_30 = full_metrics("v1.0.0 Baseline", y_test_30, baseline_pred_30, baseline_proba_30)
    ts_m_30 = full_metrics(f"Two-Stage (th={OPTIMAL_THRESHOLD})", y_test_30, ts_pred_30, ts_proba_30)

    print(f"\n{'Métrica':<25} {'Baseline':>12} {'Two-Stage':>12} {'Delta':>12}")
    print("-" * 65)
    print(f"{'Accuracy':<25} {base_m_30['accuracy']:>12.4f} {ts_m_30['accuracy']:>12.4f} {ts_m_30['accuracy']-base_m_30['accuracy']:>+12.4f}")
    print(f"{'LogLoss':<25} {base_m_30['log_loss']:>12.4f} {ts_m_30['log_loss']:>12.4f} {ts_m_30['log_loss']-base_m_30['log_loss']:>+12.4f}")
    print(f"{'Brier Global':<25} {base_m_30['brier']:>12.4f} {ts_m_30['brier']:>12.4f} {ts_m_30['brier']-base_m_30['brier']:>+12.4f}")
    print(f"{'Draw Pred %':<25} {base_m_30['draw_pred_pct']:>12.1%} {ts_m_30['draw_pred_pct']:>12.1%}")
    print(f"{'Draw Precision':<25} {base_m_30['precision_draw']:>12.3f} {ts_m_30['precision_draw']:>12.3f}")
    print(f"{'Draw Recall':<25} {base_m_30['recall_draw']:>12.3f} {ts_m_30['recall_draw']:>12.3f}")

    # ========================================
    # GO/NO-GO DECISION
    # ========================================
    print("\n" + "=" * 80)
    print("GO/NO-GO DECISION")
    print("=" * 80)

    criteria = {
        "brier_not_worse": ts_m["brier"] <= base_m["brier"] + 0.002,
        "logloss_not_worse": ts_m["log_loss"] <= base_m["log_loss"] + 0.015,
        "draw_predictions_exist": ts_m["draw_pred_pct"] > 0.05,
        "draw_precision_above_random": ts_m["precision_draw"] > ts_m["draw_prevalence"],
        "draw_prauc_improved": ts_m["pr_auc_draw"] > base_m["pr_auc_draw"],
    }

    print("\nCriterios de aceptación:")
    all_pass = True
    for name, passed in criteria.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 80)
    print("REPORTE FINAL PARA AUDITOR")
    print("=" * 80)

    print(f"""
┌─────────────────────────────────────────────────────────────┐
│ FASE 2: TWO-STAGE MODEL - REPORTE FINAL                    │
├─────────────────────────────────────────────────────────────┤
│ Arquitectura:                                               │
│   Stage 1: Binary (draw vs non-draw), sample_weight=1.2    │
│   Stage 2: Binary (home vs away)                            │
│   Decision: threshold override at p_draw > {OPTIMAL_THRESHOLD}              │
├─────────────────────────────────────────────────────────────┤
│ MÉTRICAS 90-DAY HOLDOUT                                     │
│                                                             │
│   Métrica          Baseline    Two-Stage    Delta           │
│   ─────────────────────────────────────────────────        │
│   Brier Global     {base_m['brier']:.4f}      {ts_m['brier']:.4f}       {ts_m['brier']-base_m['brier']:+.4f}          │
│   LogLoss          {base_m['log_loss']:.4f}      {ts_m['log_loss']:.4f}       {ts_m['log_loss']-base_m['log_loss']:+.4f}          │
│   Accuracy         {base_m['accuracy']:.4f}      {ts_m['accuracy']:.4f}       {ts_m['accuracy']-base_m['accuracy']:+.4f}          │
│   Draw Pred %      {base_m['draw_pred_pct']:.1%}        {ts_m['draw_pred_pct']:.1%}        {ts_m['draw_pred_pct']-base_m['draw_pred_pct']:+.1%}          │
│   Draw Precision   {base_m['precision_draw']:.3f}       {ts_m['precision_draw']:.3f}        {ts_m['precision_draw']-base_m['precision_draw']:+.3f}          │
│   Draw PR-AUC      {base_m['pr_auc_draw']:.4f}      {ts_m['pr_auc_draw']:.4f}       {ts_m['pr_auc_draw']-base_m['pr_auc_draw']:+.4f}          │
├─────────────────────────────────────────────────────────────┤
│ VEREDICTO: {"GO" if all_pass else "NO-GO":<50}│
│                                                             │
│ {"Todos los criterios cumplidos" if all_pass else "Criterios fallidos: " + ", ".join(k for k,v in criteria.items() if not v):<59}│
└─────────────────────────────────────────────────────────────┘
""")

    if all_pass:
        print("""
RECOMENDACIÓN DE DEPLOY:
1. Implementar como v1.1.0 en shadow mode
2. Validar 7-14 días comparando vs baseline
3. Si métricas se mantienen, activar gradualmente
4. Mantener v1.0.0 como fallback

CAMBIOS NECESARIOS EN engine.py:
- Agregar TwoStageModel como opción de arquitectura
- Agregar draw_threshold como parámetro configurable
- Agregar feature implied_draw derivada de odds
""")
    else:
        print("""
RECOMENDACIÓN:
- NO proceder con deploy
- Explorar alternativas:
  1. Recalibración post-hoc (isotonic regression)
  2. Ensemble con predictor de odds
  3. Análisis de features adicionales (histórico H2H, etc.)
""")


if __name__ == "__main__":
    asyncio.run(main())
