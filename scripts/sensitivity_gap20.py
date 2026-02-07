#!/usr/bin/env python3
"""
READ-ONLY / REPRODUCIBLE / NO PRODUCCIÓN
=========================================
GAP20 Sensitivity Analysis: Power devig + allow-ev-negative
ATI final operational request — 2026-02-07

Purpose: Validate NO-GO verdict for GAP20 guardrail via 2 sensitivity checks:
  1) CLV(SFAV) with power devig (vs proportional baseline)
  2) ROI/CLV with allow-ev-negative (drop EV>0 constraint)

CTEs: See docs/gap20_ctes.sql for the exact SQL used to build the dataset.

Usage:
  source .env
  python3 scripts/sensitivity_gap20.py [path_to_data.json]

Data format: JSON array of rows with columns:
  match_id, home_goals, away_goals, oh, od, oa, m_h, m_d, m_a,
  mkt_h, mkt_d, mkt_a, result, t5_oh, t5_od, t5_oa

Gates for re-run: v1.0.1-league-only N(FT)>=300 AND N(SFAV)>=30
"""
import json
import sys
import os
import numpy as np

# --- Load data ---
if len(sys.argv) > 1:
    DATA_FILE = sys.argv[1]
else:
    # Default: look for cached MCP result (from initial 2026-02-07 run)
    DATA_FILE = os.path.expanduser(
        "~/.claude/projects/-Users-inseqio-FutbolStats/"
        "fb1efec5-5102-4232-bc3c-8eb0bef7e509/tool-results/"
        "mcp-railway-postgres-query-1770498884313.txt"
    )

if not os.path.exists(DATA_FILE):
    print(f"ERROR: Data file not found: {DATA_FILE}")
    print("Generate data with: docs/gap20_ctes.sql (CTE 1: full dataset)")
    sys.exit(1)

with open(DATA_FILE) as f:
    raw = json.load(f)
    # MCP format: [{type: "text", text: "..."}] where text contains JSON array
    data_text = raw[0]["text"] if isinstance(raw, list) and isinstance(raw[0], dict) and raw[0].get("type") == "text" else raw
    if isinstance(data_text, str):
        rows = json.loads(data_text)
    else:
        rows = data_text

print(f"Total rows loaded: {len(rows)}")

# --- Devig functions ---
def devig_proportional(oh, od, oa):
    ih, id_, ia = 1/oh, 1/od, 1/oa
    total = ih + id_ + ia
    return ih/total, id_/total, ia/total

def devig_power(oh, od, oa):
    ih, id_, ia = 1/oh, 1/od, 1/oa
    overround = ih + id_ + ia
    if abs(overround - 1.0) < 0.001:
        return ih, id_, ia
    # Bisection: find k where sum(p^k) = 1
    k_lo, k_hi = 0.1, 3.0
    for _ in range(50):
        k_mid = (k_lo + k_hi) / 2.0
        f_mid = ih**k_mid + id_**k_mid + ia**k_mid - 1.0
        if f_mid > 0:
            k_lo = k_mid
        else:
            k_hi = k_mid
    k = (k_lo + k_hi) / 2.0
    total = ih**k + id_**k + ia**k
    if total < 0.001:
        return 1/3, 1/3, 1/3
    return ih**k / total, id_**k / total, ia**k / total

# --- Process rows ---
def classify_row(r, devig_fn):
    oh, od, oa = float(r['oh']), float(r['od']), float(r['oa'])
    m_h, m_d, m_a = float(r['m_h']), float(r['m_d']), float(r['m_a'])
    result = int(r['result'])

    mkt_h, mkt_d, mkt_a = devig_fn(oh, od, oa)

    # Model fav
    model_probs = [m_h, m_d, m_a]
    model_fav = model_probs.index(max(model_probs))

    # Market fav
    mkt_probs = [mkt_h, mkt_d, mkt_a]
    market_fav = mkt_probs.index(max(mkt_probs))
    market_fav_prob = max(mkt_probs)

    # Gap on model fav
    gap_on_mf = model_probs[model_fav] - mkt_probs[model_fav]

    # Disagree?
    disagree = model_fav != market_fav

    # SFAV?
    is_sfav = disagree and gap_on_mf >= 0.20 and market_fav_prob >= 0.45

    # Edges
    edges = [m_h - mkt_h, m_d - mkt_d, m_a - mkt_a]
    max_edge_idx = edges.index(max(edges))
    max_edge = edges[max_edge_idx]

    odds = [oh, od, oa]
    ev = model_probs[max_edge_idx] * odds[max_edge_idx]

    return {
        'match_id': r['match_id'],
        'model_fav': model_fav, 'market_fav': market_fav,
        'market_fav_prob': market_fav_prob, 'gap_on_mf': gap_on_mf,
        'disagree': disagree, 'is_sfav': is_sfav,
        'best_idx': max_edge_idx, 'max_edge': max_edge, 'ev': ev,
        'odds': odds, 'model_probs': model_probs, 'mkt_probs': mkt_probs,
        'result': result,
        't5_oh': r.get('t5_oh'), 't5_od': r.get('t5_od'), 't5_oa': r.get('t5_oa'),
    }

def compute_bet_roi(classified, require_ev_positive=True, edge_threshold=0.05):
    """Simulate bets and compute ROI"""
    bets = []
    for c in classified:
        if c['max_edge'] < edge_threshold:
            continue
        if require_ev_positive and c['ev'] <= 1.0:
            continue
        # Place bet on best_idx
        won = c['result'] == c['best_idx']
        pnl = c['odds'][c['best_idx']] - 1.0 if won else -1.0
        bets.append({
            'match_id': c['match_id'],
            'best_idx': c['best_idx'],
            'odds': c['odds'][c['best_idx']],
            'won': won,
            'pnl': pnl,
            'is_sfav': c['is_sfav'],
        })
    return bets

def compute_clv(classified, devig_fn, require_ev_positive=True, edge_threshold=0.05):
    """Compute CLV for bettable matches with T5 data"""
    clvs = []
    for c in classified:
        if c['max_edge'] < edge_threshold:
            continue
        if require_ev_positive and c['ev'] <= 1.0:
            continue
        if c['t5_oh'] is None or c['t5_od'] is None or c['t5_oa'] is None:
            continue
        t5_oh, t5_od, t5_oa = float(c['t5_oh']), float(c['t5_od']), float(c['t5_oa'])
        if t5_oh <= 1.0 or t5_od <= 1.0 or t5_oa <= 1.0:
            continue

        close_probs = list(devig_fn(t5_oh, t5_od, t5_oa))
        open_prob = c['mkt_probs'][c['best_idx']]
        close_prob = close_probs[c['best_idx']]

        if open_prob > 0:
            clv = (close_prob / open_prob) - 1.0
            clvs.append({
                'match_id': c['match_id'],
                'clv': clv,
                'is_sfav': c['is_sfav'],
            })
    return clvs

def bootstrap_ci(values, n_iter=5000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = []
    for _ in range(n_iter):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, 2.5), np.mean(means), np.percentile(means, 97.5)

# ============================================================
# SENSITIVITY RUN 1: CLV(B) con Power Devig
# ============================================================
print("\n" + "="*60)
print("SENSITIVITY 1: CLV(B) con POWER DEVIG")
print("="*60)

classified_power = [classify_row(r, devig_power) for r in rows]
sfav_power = [c for c in classified_power if c['is_sfav']]
print(f"SFAV (power devig): N={len(sfav_power)}")

# Bets for SFAV with power devig
bets_sfav_power = compute_bet_roi(classified_power, require_ev_positive=True)
bets_sfav_only = [b for b in bets_sfav_power if b['is_sfav']]
print(f"SFAV bets (power): N={len(bets_sfav_only)}")

if bets_sfav_only:
    total_pnl = sum(b['pnl'] for b in bets_sfav_only)
    roi = total_pnl / len(bets_sfav_only) * 100
    print(f"SFAV ROI (power): {roi:.2f}%")

# CLV for SFAV with power devig
clvs_sfav_power = compute_clv(classified_power, devig_power, require_ev_positive=True)
clvs_sfav_only = [c for c in clvs_sfav_power if c['is_sfav']]
print(f"SFAV CLV data points (power): N={len(clvs_sfav_only)}")

if clvs_sfav_only:
    clv_values = [c['clv'] for c in clvs_sfav_only]
    mean_clv = np.mean(clv_values) * 100
    pct_positive = sum(1 for v in clv_values if v > 0) / len(clv_values) * 100
    print(f"SFAV mean CLV (power): {mean_clv:.3f}%")
    print(f"SFAV % CLV>0 (power): {pct_positive:.1f}%")

    if len(clv_values) >= 10:
        lo, mid, hi = bootstrap_ci(clv_values, 5000, 42)
        print(f"SFAV CLV CI95% (power): [{lo*100:.3f}%, {hi*100:.3f}%]")

# Compare with proportional
print("\n--- Comparación vs Proporcional ---")
classified_prop = [classify_row(r, devig_proportional) for r in rows]
clvs_sfav_prop = compute_clv(classified_prop, devig_proportional, require_ev_positive=True)
clvs_sfav_prop_only = [c for c in clvs_sfav_prop if c['is_sfav']]

if clvs_sfav_prop_only:
    clv_prop = [c['clv'] for c in clvs_sfav_prop_only]
    print(f"Proporcional: mean CLV = {np.mean(clv_prop)*100:.3f}%, N={len(clv_prop)}")
    clv_pow = [c['clv'] for c in clvs_sfav_only]
    print(f"Power:        mean CLV = {np.mean(clv_pow)*100:.3f}%, N={len(clv_pow)}")
    print(f"Delta:        {(np.mean(clv_pow) - np.mean(clv_prop))*100:.3f}pp")

# ============================================================
# SENSITIVITY RUN 2: ROI/CLV Baseline con allow-ev-negative
# ============================================================
print("\n" + "="*60)
print("SENSITIVITY 2: ROI/CLV con ALLOW-EV-NEGATIVE")
print("="*60)

# Using proportional devig (baseline method)
bets_all_ev = compute_bet_roi(classified_prop, require_ev_positive=True)
bets_all_noev = compute_bet_roi(classified_prop, require_ev_positive=False)

print(f"\n--- Baseline (EV>0 required) ---")
print(f"Total bets: {len(bets_all_ev)}")
sfav_ev = [b for b in bets_all_ev if b['is_sfav']]
non_sfav_ev = [b for b in bets_all_ev if not b['is_sfav']]
print(f"  SFAV: N={len(sfav_ev)}, ROI={sum(b['pnl'] for b in sfav_ev)/max(len(sfav_ev),1)*100:.2f}%")
print(f"  non-SFAV: N={len(non_sfav_ev)}, ROI={sum(b['pnl'] for b in non_sfav_ev)/max(len(non_sfav_ev),1)*100:.2f}%")

print(f"\n--- Allow-EV-Negative ---")
print(f"Total bets: {len(bets_all_noev)}")
sfav_noev = [b for b in bets_all_noev if b['is_sfav']]
non_sfav_noev = [b for b in bets_all_noev if not b['is_sfav']]
print(f"  SFAV: N={len(sfav_noev)}, ROI={sum(b['pnl'] for b in sfav_noev)/max(len(sfav_noev),1)*100:.2f}%")
print(f"  non-SFAV: N={len(non_sfav_noev)}, ROI={sum(b['pnl'] for b in non_sfav_noev)/max(len(non_sfav_noev),1)*100:.2f}%")

# Extra bets from relaxing EV constraint
extra = len(bets_all_noev) - len(bets_all_ev)
print(f"\nExtra bets from allow-ev-negative: {extra}")

# CLV comparison
print(f"\n--- CLV Comparison ---")
clvs_ev = compute_clv(classified_prop, devig_proportional, require_ev_positive=True)
clvs_noev = compute_clv(classified_prop, devig_proportional, require_ev_positive=False)

sfav_clv_ev = [c['clv'] for c in clvs_ev if c['is_sfav']]
sfav_clv_noev = [c['clv'] for c in clvs_noev if c['is_sfav']]
all_clv_ev = [c['clv'] for c in clvs_ev]
all_clv_noev = [c['clv'] for c in clvs_noev]

print(f"Baseline (EV>0):")
print(f"  ALL: mean CLV = {np.mean(all_clv_ev)*100:.3f}%, N={len(all_clv_ev)}")
print(f"  SFAV: mean CLV = {np.mean(sfav_clv_ev)*100:.3f}%, N={len(sfav_clv_ev)}")

print(f"Allow-EV-Negative:")
print(f"  ALL: mean CLV = {np.mean(all_clv_noev)*100:.3f}%, N={len(all_clv_noev)}")
print(f"  SFAV: mean CLV = {np.mean(sfav_clv_noev)*100:.3f}%, N={len(sfav_clv_noev)}")

# Bootstrap CI for SFAV ROI (allow-ev-negative)
if sfav_noev:
    pnls = [b['pnl'] for b in sfav_noev]
    lo, mid, hi = bootstrap_ci(pnls, 5000, 42)
    print(f"\nSFAV ROI CI95% (allow-ev-neg): [{lo*100:.2f}%, {hi*100:.2f}%]")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("RESUMEN COMPARATIVO")
print("="*60)
print(f"{'Metric':<35} {'Proportional':>14} {'Power':>14}")
print("-"*63)

# SFAV counts
sfav_prop_n = len([c for c in classified_prop if c['is_sfav']])
sfav_pow_n = len([c for c in classified_power if c['is_sfav']])
print(f"{'SFAV subset N':<35} {sfav_prop_n:>14} {sfav_pow_n:>14}")

# SFAV bets
bets_prop_sfav = [b for b in compute_bet_roi(classified_prop) if b['is_sfav']]
bets_pow_sfav = [b for b in compute_bet_roi(classified_power) if b['is_sfav']]
roi_prop = sum(b['pnl'] for b in bets_prop_sfav)/max(len(bets_prop_sfav),1)*100
roi_pow = sum(b['pnl'] for b in bets_pow_sfav)/max(len(bets_pow_sfav),1)*100
print(f"{'SFAV ROI (EV>0)':<35} {roi_prop:>13.2f}% {roi_pow:>13.2f}%")

# CLV
if sfav_clv_ev and clvs_sfav_only:
    print(f"{'SFAV mean CLV':<35} {np.mean(sfav_clv_ev)*100:>13.3f}% {np.mean([c['clv'] for c in clvs_sfav_only])*100:>13.3f}%")

# Allow-ev-negative ROI
bets_noev_sfav = [b for b in compute_bet_roi(classified_prop, require_ev_positive=False) if b['is_sfav']]
roi_noev = sum(b['pnl'] for b in bets_noev_sfav)/max(len(bets_noev_sfav),1)*100
print(f"{'SFAV ROI (allow-ev-neg, prop)':<35} {roi_noev:>13.2f}% {'N/A':>14}")

print("\n[OK] Sensitivity analysis complete")
