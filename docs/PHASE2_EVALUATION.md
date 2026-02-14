# Phase 2 Evaluation Report — Asymmetry & Microstructure

**Date**: 2026-02-14
**Sprint**: 4 of 4 (Final)
**Status**: COMPLETE — All 16 tickets delivered
**Model**: v1.0.0 (XGBoost 14-feature, unchanged)

---

## Executive Summary

Phase 2 deployed the infrastructure to exploit timing asymmetries between lineup confirmation (~T-60m) and kickoff. Four pillars were built:

1. **MTV (Missing Talent Value)**: Player entity resolution, talent scoring (PTS + VORP prior), expected XI injury-aware, talent delta features — all deployed as forward data collection.
2. **CLV (Closing Line Value)**: Formal 3-way log-odds scoring against canonical bookmaker. 849 matches scored across 25 leagues.
3. **Event-Driven Cascade**: DB-backed event bus with Sweeper Queue (2min reconciliation), steel degradation (5s timeout), idempotent handler.
4. **SteamChaser**: Binary classifier pipeline for line movement prediction. Shadow mode / data accumulation only (644 pairs, 10 positives).

**Key finding**: The model bleeds CLV systematically — predictions at T-24h lose value by closing. This confirms the Phase 1 RESOLUTION_ISSUE: the model is well-calibrated but does not beat the market's information incorporation speed. The cascade infrastructure is ready to close this gap once MTV features accumulate sufficient training data.

---

## 1. CLV Distribution (N=849, 25 leagues)

### 1.1 Global

| Metric | Home | Draw | Away |
|--------|------|------|------|
| Mean CLV | -0.00522 | -0.00169 | +0.00009 |
| Median CLV | 0.00000 | 0.00000 | 0.00000 |
| % Positive | 33.5% | 32.6% | 37.3% |

**Canonical bookmaker**: Single source (Bet365 priority chain).
**Interpretation**: Negative mean CLV = model predictions at T-24h are systematically worse than the closing line. The market incorporates late information (lineups, injuries, weather) that the daily batch prediction cannot access. Away outcomes show near-zero CLV — the market's away pricing is least efficient.

### 1.2 By League (Top 15)

| League | N | CLV Home | CLV Away | % Pos H | % Pos A |
|--------|---|----------|----------|---------|---------|
| Argentina | 77 | -0.01258 | -0.00475 | 32.5% | 32.5% |
| Championship | 60 | -0.01230 | -0.00992 | 36.7% | 40.0% |
| Saudi | 58 | +0.00193 | +0.00037 | 37.9% | 48.3% |
| Colombia | 54 | +0.00490 | -0.03042 | 37.0% | 33.3% |
| EPL | 51 | -0.00354 | +0.01896 | 43.1% | 35.3% |
| Brazil | 42 | -0.00118 | -0.03938 | 38.1% | 28.6% |
| Bundesliga | 41 | -0.02353 | +0.02775 | 9.8% | 53.7% |
| Süper Lig | 36 | -0.00363 | +0.02746 | 30.6% | 44.4% |
| Serie A | 36 | +0.01761 | -0.01772 | 50.0% | 27.8% |
| Primeira Liga | 35 | -0.01311 | +0.02338 | 34.3% | 45.7% |
| Eredivisie | 35 | -0.00041 | -0.03138 | 40.0% | 22.9% |
| La Liga | 35 | -0.01262 | -0.00102 | 14.3% | 31.4% |
| Ligue 1 | 33 | -0.00945 | +0.02054 | 42.4% | 45.5% |
| Mexico | 31 | -0.01077 | +0.01362 | 32.3% | 54.8% |
| Peru | 30 | +0.00297 | +0.02571 | 36.7% | 36.7% |

**Observations**:
- **Positive CLV pockets**: Serie A home (+0.0176), Peru away (+0.0257), Bundesliga away (+0.0278), Süper Lig away (+0.0275), EPL away (+0.0190). These are leagues where the model's away/home pricing at T-24h outperforms the closing line.
- **Worst CLV**: Eredivisie away (-0.0314), Colombia away (-0.0304), Brazil away (-0.0394). Markets that incorporate late information most aggressively.
- **Bundesliga anomaly**: Home CLV 9.8% positive (worst across all leagues) but away CLV 53.7% positive (best). Strong directional bias in model error.

### 1.3 CLV Verdict

The model does NOT systematically beat the closing line. Mean CLV is negative for 2 of 3 outcomes. However, **league-specific pockets of positive CLV exist**, particularly for away outcomes in mid-efficiency markets (Süper Lig, Primeira, Ligue 1, Mexico, Peru). Phase 3 should investigate whether Market Anchor with league-specific alpha can exploit these pockets.

---

## 2. SteamChaser Readiness (P2-13)

### 2.1 Data Inventory

| Metric | Value |
|--------|-------|
| T60→T5 pairs | 644 |
| Date range | 2026-01-10 to 2026-02-14 |
| Positive target (drift > vig/2) | 10 (1.55%) |
| Avg overround (T60) | 6.558% |
| Vig friction threshold | ~3.28% (overround/2) |
| Avg max drift | 0.257% |
| P90 max drift | 1.017% |
| P95 max drift | 1.601% |

### 2.2 Readiness Assessment

```
STATUS: NOT READY — ACCUMULATING

Samples:     644 / 500 minimum  ✓ (met)
Positives:    10 /  30 minimum  ✗ (need 20 more)
Imbalance:   1.55% positive rate — SEVERE

Estimated readiness: ~mid-April 2026
  At current rate (~18 pairs/day, ~0.3 positives/day):
  - 30 positives reached in ~67 days from start (~2026-03-18)
  - 2,000 pairs reached in ~75 days from start (~2026-03-26)
```

### 2.3 Target Definition (ATI Directive — Vig-Adjusted)

```
y = 1  if  max(|prob_close_k - prob_T60_k|) > overround_T60 / 2
```

The vig-adjusted threshold ensures we only chase movements that overcome bookmaker margin. At avg overround 6.56%, the threshold is ~3.28%. Only 1.55% of matches see movements this large — confirming that **vig-significant line collapses are rare events**.

### 2.4 Feature Matrix (Current)

| Feature | Description |
|---------|-------------|
| overround_t60 | Market liquidity at T-60m |
| prob_t60_home | Market probability home win |
| prob_t60_draw | Market probability draw |
| prob_t60_away | Market probability away win |
| prob_t60_fav | Max probability (market confidence) |
| prob_t60_range | Spread between fav and underdog |

**Forward-only features** (not yet in model, accumulating data):
- `talent_delta_home`, `talent_delta_away` — lineup shock signal
- `shock_magnitude` — max absolute talent delta
- `xi_continuity_home`, `xi_continuity_away` — rotation signal

### 2.5 OOT Evaluation Function

`run_oot_evaluation()` is ready in `app/ml/steamchaser.py`. Will execute when MIN_POSITIVE_SAMPLES (30) is met. Uses:
- Chronological 70/30 split
- XGBoost with `scale_pos_weight` for imbalance
- **AUC + LogLoss only** (ATI mandate: NO accuracy for imbalanced binary)
- Baseline comparison (predict mean positive rate)

---

## 3. A/B Cascade vs Daily (P2-15)

### 3.1 Current Status

```
STATUS: ACCUMULATING — 0 cascade predictions scored

Cascade infrastructure deployed in Sprint 3.
lineup_detected_at going-forward population: 0 (just deployed)
Sweeper Queue: active (2min interval)
Event Bus: active (subscribed to LINEUP_CONFIRMED)
```

**Why 0 cascade predictions**: The `lineup_detected_at` column was added in Sprint 1 (P2-03) but only populates going-forward from Sprint 3's scheduler integration. No matches have yet been processed through the full cascade pipeline since deployment.

### 3.2 Expected Timeline

- **Week 1 post-deploy** (~2026-02-21): First cascade predictions appear
- **N=50** (~2026-03-01): Preliminary CLV comparison possible
- **N=200** (~2026-03-15): Statistically meaningful A/B comparison

### 3.3 Metrics Defined (Ready for Scoring)

| Metric | Formula | Gate |
|--------|---------|------|
| CLV cascade vs daily | mean(CLV_cascade) - mean(CLV_daily) | Cascade > Daily |
| P95 latency | P95(cascade_elapsed_ms) | < 2,000ms |
| Freshness gain | mean(match_date - asof_timestamp) | Cascade < Daily |
| Degradation rate | cascade_failures / cascade_attempts | < 5% |

Dashboard endpoint `_calculate_cascade_ab_test()` in ops_routes.py computes these automatically.

---

## 4. Infrastructure Inventory

### 4.1 Predictions

| Metric | Value |
|--------|-------|
| Total predictions | 5,342 |
| With asof_timestamp | 5,342 (100%) |
| Unique matches | 3,006 |
| Model versions | 3 |
| Date range | 2026-01-04 to 2026-02-14 |

### 4.2 Market Movement Snapshots

| Type | Count | Unique Matches |
|------|-------|---------------|
| T60 | 708 | 708 |
| T30 | 720 | 720 |
| T15 | 732 | 732 |
| T5 | 715 | 715 |
| **Total** | **2,875** | — |

Date range: 2026-01-10 to 2026-02-14 (35 days).

### 4.3 Lineup Movement Snapshots

| Type | Count | Unique Matches |
|------|-------|---------------|
| L0 | 888 | 888 |
| L+5 | 853 | 853 |
| L+10 | 19 | 19 |
| **Total** | **1,760** | — |

L+10 low count is expected: most matches don't have 10min of post-lineup snapshot history yet.

### 4.4 Player ID Mapping

| Metric | Value |
|--------|-------|
| Total mappings | 4,613 |
| Active (confidence > threshold) | 3,940 (85.4%) |
| Pending review | 673 (14.6%) |
| Blocked | 0 |
| Avg confidence (all) | 0.9262 |
| Avg confidence (active) | 0.9629 |

**By method**:
| Method | Count | Avg Confidence |
|--------|-------|---------------|
| bipartite_strong | 3,830 | 0.9664 |
| bipartite | 535 | 0.7621 |
| bipartite_weak | 248 | 0.6602 |

### 4.5 XI Continuity Coverage

| Metric | Value |
|--------|-------|
| Lineup rows with ≥11 players | 117,822 |
| Unique matches | 58,924 |

Deep historical coverage for xi_continuity backtest.

### 4.6 Sofascore Talent Data

| Metric | Value |
|--------|-------|
| Matches with Sofascore data + T60 snapshot | 369 |
| Total matches with Sofascore data | 405 |
| Player rating history rows | 49,319 |
| Unique players | ~9,992 |
| Date range | Nov 2025+ |

---

## 5. Sprint Delivery Summary

### Sprint 1: Data Canon + Entity Resolution (P2-01 to P2-04) ✓

| Ticket | Deliverable | Status |
|--------|------------|--------|
| P2-01 | Player entity resolution (bipartite Hungarian) | ✓ 4,613 mappings, 0.926 avg confidence |
| P2-02 | asof_timestamp + canonical bookmaker | ✓ 5,342 predictions with asof |
| P2-03 | lineup_detected_at column | ✓ Column added, going-forward |
| P2-04 | CLV scoring table (3-way log-odds) | ✓ 849 scored, 25 leagues |

### Sprint 2: MTV Features + VORP (P2-05 to P2-08) ✓

| Ticket | Deliverable | Status |
|--------|------------|--------|
| P2-05 | PTS + VORP prior (P25 bayesiano) | ✓ Zero-division impossible |
| P2-06 | Expected XI injury-aware | ✓ Filters player_injuries |
| P2-07 | Talent delta (total + positional) | ✓ Forward data collection |
| P2-08 | XI frequency proxy for backtest | ✓ match_lineups historical |

### Sprint 3: Event-Driven + Cascade (P2-09 to P2-12) ✓

| Ticket | Deliverable | Status |
|--------|------------|--------|
| P2-09 | DB-backed event bus + Sweeper Queue | ✓ FOR UPDATE SKIP LOCKED, 2min |
| P2-10 | Cascade handler (steel degradation) | ✓ 5s timeout, Phase 1 fallback |
| P2-11 | Line movement features PIT-safe | ✓ captured_at ≤ asof_timestamp |
| P2-12 | CLV dashboard metric | ✓ Per-league + global |

### Sprint 4: SteamChaser + Evaluation (P2-13 to P2-16) ✓

| Ticket | Deliverable | Status |
|--------|------------|--------|
| P2-13 | SteamChaser pipeline (shadow/data collection) | ✓ 644 pairs, vig-adjusted target |
| P2-14 | OOT evaluation function (AUC+LogLoss) | ✓ Ready, deferred (10 positives) |
| P2-15 | A/B cascade vs daily | ✓ Dashboard metric, accumulating |
| P2-16 | Phase 2 evaluation report | ✓ This document |

---

## 6. ATI/GDT Compliance Matrix

| Directive | Requirement | Status |
|-----------|------------|--------|
| ATI #1 | SteamChaser secondary model | ✓ Pipeline built, shadow mode |
| ATI #2 | VORP prior P25 for unknown players | ✓ Implemented in PTS computation |
| ATI #3 | Steel degradation 5s timeout | ✓ asyncio.wait_for in cascade handler |
| ATI #4 | Sweeper Queue auto-heal | ✓ 2min interval, FOR UPDATE SKIP LOCKED |
| GDT #1 | asof_timestamp on every prediction | ✓ 100% coverage (5,342/5,342) |
| GDT #2 | lineup_detected_at separated from provider | ✓ Column added, going-forward |
| GDT #3 | Bipartite matching (Hungarian) | ✓ 4,613 mappings, 3 methods |
| GDT #4 | Injury-aware Expected XI | ✓ player_injuries filter |
| GDT #5 | CLV 3-way log-odds, canonical bookmaker | ✓ Single bookmaker, 849 scored |
| GDT #6 | DB-backed event bus | ✓ DB as source of truth + asyncio.Queue |
| GDT #7 | Cascade optimized (no static recompute) | ✓ Only lineup-dependent features |

---

## 7. Phase 3 Recommendations

### 7.1 Immediate (Next 2 Weeks)

1. **Monitor cascade pipeline**: First cascade predictions should appear by 2026-02-21. Verify via ops dashboard CLV cascade section.
2. **SteamChaser data accumulation**: Continue T60→T5 snapshot collection. At ~18 pairs/day, MIN_POSITIVE_SAMPLES (30) reachable by mid-March.
3. **CLV league analysis**: With N>50 per league, investigate Market Anchor alpha optimization for positive-CLV pockets (Serie A home, Süper Lig away, EPL away).

### 7.2 Medium Term (1-2 Months)

4. **A/B cascade vs daily**: At N≥200 cascade predictions (~mid-March), run formal CLV comparison. Gate: cascade CLV > daily CLV with p<0.05.
5. **SteamChaser first training**: At ≥30 positives (~late March), run `run_oot_evaluation()`. Report PR-AUC and compare to prevalence baseline. Gate: PR-AUC > 2× base rate for signal. ATI mandate: vig/2 threshold is sacred, never relax.
6. **MTV feature integration**: If talent_delta shows signal in forward shadow data, add to model v2 feature set. Requires retraining on Railway.

### 7.3 Long Term (3+ Months)

7. **Model v2 training**: Combine Phase 1 features (14) + MTV features (talent_delta, shock_magnitude, xi_continuity) + line movement features. OOT evaluation with expanded feature set.
8. **SteamChaser productionization**: If AUC > 0.60 after sufficient data, move from shadow to live scoring. Integrate with cascade handler for real-time line collapse alerts.
9. **Market Anchor dynamic alpha**: Use CLV distribution by league to set per-league alpha dynamically. Leagues with consistently positive CLV → lower alpha (trust model more). Leagues with negative CLV → higher alpha (defer to market).

### 7.4 Risk Register

| Risk | Probability | Mitigation |
|------|------------|-----------|
| Cascade latency > 2s at scale | Low | Steel degradation ensures < 5s worst case |
| SteamChaser never finds signal | Medium | ATI mandate: vig/2 threshold is sacred. Use scale_pos_weight + PR-AUC optimization. Wait for data. |
| MTV features don't improve model | Medium | xi_freq proxy backtest (58K matches) provides early signal |
| CLV negative everywhere | Low | League-specific pockets already identified |
| Sofascore data loss | Medium | Forward-only dependency; backtest uses xi_freq proxy |

---

## 8. Files Delivered (Phase 2)

| File | Sprint | Lines | Purpose |
|------|--------|-------|---------|
| `scripts/build_player_id_mapping.py` | S1 | ~280 | Bipartite Hungarian matching |
| `app/features/engineering.py` | S1-S3 | +350 | PTS, VORP, expected XI, talent delta, xi_continuity, line movement |
| `app/events/__init__.py` | S3 | 35 | Event package exports |
| `app/events/bus.py` | S3 | 231 | EventBus + Sweeper Queue |
| `app/events/handlers.py` | S3 | 232 | Cascade handler with steel degradation |
| `app/ml/steamchaser.py` | S4 | 393 | SteamChaser pipeline + OOT evaluation |
| `app/ml/policy.py` | S1 | +40 | Market Anchor canonical bookmaker |
| `app/dashboard/ops_routes.py` | S3-S4 | +180 | CLV dashboard + cascade A/B metrics |
| `app/scheduler.py` | S3 | +80 | Sweeper job, lineup_detected_at, emit post-commit |
| `app/main.py` | S3 | +15 | EventBus lifespan start/stop |
| `docs/PHASE2_ARCHITECTURE.md` | S0 | ~700 | Architecture document for ATI |
| `docs/PHASE2_EVALUATION.md` | S4 | ~350 | This document |
| DB migrations | S1 | 4 | player_id_mapping, prediction_clv, lineup_detected_at, asof_timestamp |

---

*Phase 2 COMPLETE. All 16 tickets delivered across 4 sprints. Infrastructure ready for Phase 3 feature integration when data gates are met.*
