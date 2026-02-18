# Family S Training Results — Mandato D

**Date**: 2026-02-17
**Model Version**: `v2.0-tier3-family_s`
**Snapshot ID**: 6
**Environment**: Railway (Python 3.12, internal DB)

---

## Overview

Family S is a specialized XGBoost model for 5 Tier 3 leagues where MTV (Market-to-Talent Value) features improve prediction quality. It uses 24 features (17 core + 3 odds + 4 MTV) compared to the baseline's 17 core features.

Family S operates **only in the cascade handler** (post-lineup confirmation) for Tier 3 matches. All other leagues continue using the baseline model (`v1.0.1-league-only`).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model version | `v2.0-tier3-family_s` |
| Feature set | `core17+odds3+mtv4` (24 features) |
| Min training date | 2023-01-01 |
| MTV source | `historical_mtv_features_tm_hiconf_padded.parquet` |
| MTV NaN strategy | Fill with 0.0 (padded) |
| Odds filter | Rows without complete triplet dropped |
| XGBoost CV | 3-fold stratified |
| Feature fidelity | `FeatureEngineer.build_training_dataset()` (P0-1) |

### Tier 3 Leagues

| League ID | League | Country |
|-----------|--------|---------|
| 88 | Eredivisie | Netherlands |
| 94 | Primeira Liga | Portugal |
| 144 | Belgian Pro League | Belgium |
| 203 | Süper Lig | Turkey |
| 265 | Primera División | Chile |

### Feature Columns (24)

**Core (17)**: `home_goals_scored_avg`, `home_goals_conceded_avg`, `home_shots_avg`, `home_corners_avg`, `home_rest_days`, `home_matches_played`, `away_goals_scored_avg`, `away_goals_conceded_avg`, `away_shots_avg`, `away_corners_avg`, `away_rest_days`, `away_matches_played`, `goal_diff_avg`, `rest_diff`, `abs_attack_diff`, `abs_defense_diff`, `abs_strength_gap`

**Odds (3)**: `odds_home`, `odds_draw`, `odds_away`

**MTV (4)**: `home_talent_delta`, `away_talent_delta`, `talent_delta_diff`, `shock_magnitude`

---

## Global Results

| Metric | Value |
|--------|-------|
| **Brier Score (CV avg)** | **0.1934** |
| Fold 1 | 0.1902 |
| Fold 2 | 0.1937 |
| Fold 3 | 0.1964 |
| Fold variance | 0.003 (low — no overfitting to a single split) |
| Total samples | 4,755 |
| FeatureEngineer rows | 4,904 |
| Dropped (no odds) | 149 (3.0%) |
| MTV parquet matched | 4,903/4,904 (99.98%) |
| Model blob size | 71,067 bytes |

### Baseline Comparison

| Snapshot | Model Version | Brier | Samples | Status |
|----------|--------------|-------|---------|--------|
| id=4 | `v1.0.1-league-only` | 0.2035 | 20,233 | **ACTIVE** |
| id=5 | `v1.1.0-twostage` | 0.2078 | 21,806 | inactive |
| id=6 | `v2.0-tier3-family_s` | **0.1934** | 4,755 | inactive |

> **Note**: Brier scores are not directly comparable. Family S trains on a subset of 5 leagues (potentially more predictable markets), while baseline trains on all leagues. The definitive evaluation will be PIT (Point-in-Time) with `evaluate_pit_v3.py` after accumulating N>=300 production predictions.

---

## Per-League Results

Evaluated on training data using multi-class Brier (sum of squared errors across 3 outcome classes, range 0-2). Reference: random prediction (1/3, 1/3, 1/3) = 0.6667.

| Rank | League | Brier (3-class) | Samples | vs Random |
|------|--------|----------------|---------|-----------|
| 1 | Primeira Liga (Portugal) | **0.5131** | 991 | -23.0% |
| 2 | Eredivisie (Netherlands) | **0.5288** | 998 | -20.7% |
| 3 | Süper Lig (Turkey) | **0.5514** | 1,066 | -17.3% |
| 4 | Belgian Pro League | **0.5757** | 968 | -13.6% |
| 5 | Chile Primera | **0.6090** | 732 | -8.7% |

### Per-League Analysis

**Primeira Liga (best, 0.5131)**: Best data coverage — xG available since 2020 (FotMob), strong OddsPortal odds backfill, large sample. Most predictable market among Tier 3.

**Eredivisie (0.5288)**: Similar profile to Primeira Liga — xG since 2020, good odds coverage. Second most predictable.

**Süper Lig (0.5514)**: xG available since 2022 (shorter history). Decent odds coverage from OddsPortal (Saudi backfill pipeline). Middle of the pack.

**Belgian Pro League (0.5757)**: xG since 2020 but complex playoff format may introduce noise. Slightly worse despite good data coverage.

**Chile Primera (worst, 0.6090)**: **Zero xG** from any source (FotMob confirmed Opta not deployed). Fewest samples (732). Despite having the strongest MTV delta in Feature Lab (-0.01719), the lack of xG data limits overall prediction quality. MTV alone cannot compensate for missing xG.

### Key Insight

Per-league Brier strongly correlates with **xG availability**: leagues with xG since 2020 (Primeira, Eredivisie) outperform those with limited/no xG (Chile). This suggests xG is a more impactful feature than MTV for overall prediction quality, even though MTV provides incremental improvement on top.

---

## Feature Lab Deltas (pre-training reference)

These were the MTV improvement signals from Feature Lab that selected these 5 leagues for Tier 3:

| League | Feature Lab Δ (MTV vs baseline) |
|--------|--------------------------------|
| Chile Primera | -0.01719 (strongest signal) |
| Primeira Liga | -0.01085 |
| Eredivisie | -0.00987 |
| Süper Lig | -0.00624 |
| Belgian Pro League | strong coverage (no specific Δ) |

---

## P0 Verification Checklist

| Check | Status | Detail |
|-------|--------|--------|
| P0-1 Feature fidelity | PASS | `FeatureEngineer.build_training_dataset()` used (no custom SQL) |
| P0-2 24-feature serving | PASS | `FamilySEngine.FEATURE_COLUMNS` override (24 items) |
| P0-3 Safe persistence | PASS | `persist_family_s_snapshot()`, baseline untouched |
| P0-4 Baseline SSOT | PASS | Baseline id=4 `is_active=true`, Family S id=6 `is_active=false` |
| P1 Init always | PASS | Startup loads Family S regardless of flag |
| P1 Serving scope | PASS | Family S only in cascade handler |

---

## Execution Timeline

| Time (UTC) | Event |
|------------|-------|
| 23:39:54 | Training endpoint triggered |
| 23:39:55 | FeatureEngineer query started |
| 23:40:07 | 4,904 matches found, building features |
| 23:43:51 | Features complete. MTV merge: 4,903 rows. Odds filter: 4,755 rows |
| 23:43:51 | XGBoost training started (3-fold CV) |
| 23:44:39 | Fold 1: Brier = 0.1902 |
| 23:45:28 | Fold 2: Brier = 0.1937 |
| 23:46:13 | Fold 3: Brier = 0.1964 |
| 23:46:57 | Snapshot persisted: id=6, is_active=false |
| **~7 min** | **Total training time** |

---

## Activation Sequence

1. ABE confirms GO
2. Restart Railway service (loads snapshot id=6 at startup via `init_family_s_engine()`)
3. Flip env var: `LEAGUE_ROUTER_MTV_ENABLED=true` (no redeploy needed)
4. Verify cascade logs show `strategy=FAMILY_S` for Tier 3 matches
5. Cleanup: remove parquet from git + temporary endpoints
6. Post-activation PIT evaluation at N>=300 predictions
