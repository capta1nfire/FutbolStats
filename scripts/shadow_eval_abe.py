#!/usr/bin/env python3
"""
Shadow (two-stage) vs Model A vs Market — ABE P0 Evaluation
READ-ONLY: only SELECT queries, no mutations.

Usage:
    source .env
    python scripts/shadow_eval_abe.py
"""

import os
import json
import numpy as np
import psycopg2
import psycopg2.extras
from datetime import datetime
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────
CUTOFF = "2026-01-15"
N_BOOTSTRAP = 1000
N_ECE_BINS = 10
MARKET_SOURCES = ("Pinnacle", "consensus")  # ABE: Pinnacle or consensus
RNG_SEED = 42


# ─── DB ──────────────────────────────────────────────────────
def get_conn():
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(db_url)


# ─── Extraction (Paso B) ────────────────────────────────────
def extract_intersection(conn) -> list[dict]:
    """Extract intersection dataset: shadow + baseline + market pre-KO."""
    query = """
    WITH shadow AS (
        SELECT DISTINCT ON (sp.match_id)
            sp.match_id,
            sp.baseline_version,
            sp.baseline_home_prob, sp.baseline_draw_prob, sp.baseline_away_prob,
            sp.shadow_home_prob, sp.shadow_draw_prob, sp.shadow_away_prob,
            sp.created_at AS shadow_created_at
        FROM shadow_predictions sp
        WHERE sp.shadow_architecture = 'two_stage'
          AND sp.shadow_version = 'v1.1.0-twostage'
        ORDER BY sp.match_id, sp.created_at DESC
    ),
    market AS (
        SELECT DISTINCT ON (oh.match_id)
            oh.match_id,
            oh.odds_home, oh.odds_draw, oh.odds_away,
            oh.source AS market_source,
            oh.recorded_at AS market_recorded_at
        FROM odds_history oh
        JOIN matches m ON m.id = oh.match_id
        WHERE oh.source IN ('Pinnacle', 'consensus')
          AND oh.recorded_at < m.date
          AND (oh.quarantined IS NULL OR oh.quarantined = false)
          AND (oh.tainted IS NULL OR oh.tainted = false)
          AND oh.odds_home > 1.0 AND oh.odds_draw > 1.0 AND oh.odds_away > 1.0
        ORDER BY oh.match_id,
              CASE oh.source WHEN 'Pinnacle' THEN 1 WHEN 'consensus' THEN 2 END,
              oh.recorded_at DESC
    )
    SELECT
        m.id AS match_id,
        m.date AS kickoff,
        m.league_id,
        al.name AS league_name,
        m.home_goals, m.away_goals,
        m.status,
        -- Shadow (two-stage)
        s.shadow_home_prob, s.shadow_draw_prob, s.shadow_away_prob,
        -- Model A (baseline)
        s.baseline_version,
        s.baseline_home_prob, s.baseline_draw_prob, s.baseline_away_prob,
        -- Market
        mk.odds_home AS mkt_odds_home, mk.odds_draw AS mkt_odds_draw, mk.odds_away AS mkt_odds_away,
        mk.market_source
    FROM shadow s
    JOIN market mk ON mk.match_id = s.match_id
    JOIN matches m ON m.id = s.match_id
    LEFT JOIN admin_leagues al ON al.league_id = m.league_id
    WHERE m.date >= %s::timestamp
      AND m.status IN ('FT', 'AET', 'PEN')
      AND m.home_goals IS NOT NULL
      AND m.away_goals IS NOT NULL
    ORDER BY m.date;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (CUTOFF,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ─── Devig & Label ───────────────────────────────────────────
def devig_proportional(odds_h, odds_d, odds_a):
    """Proportional de-vig: implied / sum(implied)."""
    imp_h, imp_d, imp_a = 1/odds_h, 1/odds_d, 1/odds_a
    total = imp_h + imp_d + imp_a
    return imp_h/total, imp_d/total, imp_a/total


def get_label(row):
    """Label: 0=home, 1=draw, 2=away. AET/PEN => draw."""
    if row["status"] in ("AET", "PEN"):
        return 1  # draw
    if row["home_goals"] > row["away_goals"]:
        return 0
    elif row["home_goals"] == row["away_goals"]:
        return 1
    else:
        return 2


def normalize(h, d, a):
    """Clamp and renormalize to sum=1."""
    h, d, a = max(0, h), max(0, d), max(0, a)
    t = h + d + a
    if t < 1e-6:
        return 1/3, 1/3, 1/3
    return h/t, d/t, a/t


# ─── Metrics ─────────────────────────────────────────────────
def multiclass_brier(y_true, probs):
    """Brier score: mean(sum_k (p_k - y_k)^2)."""
    n = len(y_true)
    score = 0.0
    for i in range(n):
        for k in range(3):
            indicator = 1.0 if y_true[i] == k else 0.0
            score += (probs[i][k] - indicator) ** 2
    return score / n


def multiclass_logloss(y_true, probs):
    """LogLoss: mean(-log(p_true))."""
    eps = 1e-15
    total = 0.0
    for i in range(len(y_true)):
        p = max(eps, probs[i][y_true[i]])
        total -= np.log(p)
    return total / len(y_true)


def ece_per_class(y_true, probs, n_bins=N_ECE_BINS):
    """ECE per class (home/draw/away). Returns dict with ece and worst_bin per class."""
    results = {}
    class_names = ["home", "draw", "away"]
    for k, name in enumerate(class_names):
        p_k = [probs[i][k] for i in range(len(y_true))]
        y_k = [1.0 if y_true[i] == k else 0.0 for i in range(len(y_true))]

        bins_ece = 0.0
        worst_bin_gap = 0.0
        worst_bin_info = ""

        for b in range(n_bins):
            lo = b / n_bins
            hi = (b + 1) / n_bins
            mask = [i for i in range(len(p_k)) if lo <= p_k[i] < hi]
            if not mask:
                continue
            avg_conf = np.mean([p_k[i] for i in mask])
            avg_acc = np.mean([y_k[i] for i in mask])
            gap = abs(avg_conf - avg_acc)
            bins_ece += gap * len(mask) / len(y_true)
            if gap > worst_bin_gap:
                worst_bin_gap = gap
                worst_bin_info = f"[{lo:.1f}-{hi:.1f}] conf={avg_conf:.3f} acc={avg_acc:.3f} n={len(mask)}"

        results[name] = {
            "ece": round(bins_ece, 5),
            "worst_bin_gap": round(worst_bin_gap, 5),
            "worst_bin": worst_bin_info,
        }
    return results


def bootstrap_delta(y_true, probs_a, probs_b, metric_fn, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap CI for metric(A) - metric(B). Paired resampling."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        y_boot = [y_true[i] for i in idx]
        pa_boot = [probs_a[i] for i in idx]
        pb_boot = [probs_b[i] for i in idx]
        delta = metric_fn(y_boot, pa_boot) - metric_fn(y_boot, pb_boot)
        deltas.append(delta)
    deltas = sorted(deltas)
    ci_lo = deltas[int(0.025 * n_boot)]
    ci_hi = deltas[int(0.975 * n_boot)]
    mean_delta = np.mean(deltas)
    return {"mean": round(mean_delta, 5), "ci95_lo": round(ci_lo, 5), "ci95_hi": round(ci_hi, 5)}


# ─── Main ────────────────────────────────────────────────────
def main():
    print(f"\n{'='*70}")
    print(f"  SHADOW EVALUATION — ABE P0")
    print(f"  Cutoff: >= {CUTOFF} | Market: {MARKET_SOURCES}")
    print(f"  Label: AET/PEN => draw")
    print(f"{'='*70}\n")

    conn = get_conn()

    # ─── Paso B: Extract intersection ───
    print("[B] Extracting intersection dataset...")
    rows = extract_intersection(conn)
    conn.close()

    N = len(rows)
    print(f"    N_intersection = {N}")
    if N == 0:
        print("    [ERROR] No intersection data. Aborting.")
        return

    # Build arrays
    y_true = []
    shadow_probs = []
    modela_probs = []
    market_probs = []
    league_ids = []
    league_names = []
    baseline_versions = []

    for r in rows:
        y_true.append(get_label(r))

        sh = normalize(float(r["shadow_home_prob"]), float(r["shadow_draw_prob"]), float(r["shadow_away_prob"]))
        shadow_probs.append(sh)

        ma = normalize(float(r["baseline_home_prob"]), float(r["baseline_draw_prob"]), float(r["baseline_away_prob"]))
        modela_probs.append(ma)

        mk = devig_proportional(float(r["mkt_odds_home"]), float(r["mkt_odds_draw"]), float(r["mkt_odds_away"]))
        market_probs.append(mk)

        league_ids.append(r["league_id"])
        league_names.append(r["league_name"] or f"ID:{r['league_id']}")
        baseline_versions.append(r["baseline_version"])

    # ─── Paso C: Metrics ───
    print(f"\n[C] Computing metrics (N={N})...\n")

    # 1) Global metrics
    brier_shadow = multiclass_brier(y_true, shadow_probs)
    brier_modela = multiclass_brier(y_true, modela_probs)
    brier_market = multiclass_brier(y_true, market_probs)

    ll_shadow = multiclass_logloss(y_true, shadow_probs)
    ll_modela = multiclass_logloss(y_true, modela_probs)
    ll_market = multiclass_logloss(y_true, market_probs)

    skill_shadow = 1 - (brier_shadow / brier_market) if brier_market > 0 else 0
    skill_modela = 1 - (brier_modela / brier_market) if brier_market > 0 else 0

    print("=" * 60)
    print("  1) GLOBAL METRICS")
    print("=" * 60)
    print(f"  {'':20s} {'Brier':>8s} {'LogLoss':>8s} {'Skill%':>8s}")
    print(f"  {'─'*50}")
    print(f"  {'Shadow (two-stage)':20s} {brier_shadow:8.4f} {ll_shadow:8.4f} {skill_shadow*100:+7.2f}%")
    print(f"  {'Model A (baseline)':20s} {brier_modela:8.4f} {ll_modela:8.4f} {skill_modela*100:+7.2f}%")
    print(f"  {'Market (devigged)':20s} {brier_market:8.4f} {ll_market:8.4f}     ref")
    print(f"\n  N = {N} | baseline versions: {dict(zip(*np.unique(baseline_versions, return_counts=True)))}")

    # Bootstrap: Shadow - ModelA
    print(f"\n  Bootstrap Δ(Shadow − ModelA), {N_BOOTSTRAP} resamples:")
    delta_brier = bootstrap_delta(y_true, shadow_probs, modela_probs, multiclass_brier)
    delta_ll = bootstrap_delta(y_true, shadow_probs, modela_probs, multiclass_logloss)
    print(f"    ΔBrier  = {delta_brier['mean']:+.5f}  CI95 [{delta_brier['ci95_lo']:+.5f}, {delta_brier['ci95_hi']:+.5f}]")
    print(f"    ΔLogLoss= {delta_ll['mean']:+.5f}  CI95 [{delta_ll['ci95_lo']:+.5f}, {delta_ll['ci95_hi']:+.5f}]")

    sig_brier = "YES" if (delta_brier["ci95_lo"] > 0 or delta_brier["ci95_hi"] < 0) else "NO"
    print(f"    Significant (CI excludes 0)? {sig_brier}")

    # 2) Calibration (ECE per class)
    print(f"\n{'='*60}")
    print("  2) CALIBRATION (ECE per class)")
    print("=" * 60)
    ece_shadow = ece_per_class(y_true, shadow_probs)
    ece_modela = ece_per_class(y_true, modela_probs)

    print(f"  {'Class':8s} {'Shadow ECE':>12s} {'ModelA ECE':>12s}")
    print(f"  {'─'*36}")
    for cls in ["home", "draw", "away"]:
        print(f"  {cls:8s} {ece_shadow[cls]['ece']:12.5f} {ece_modela[cls]['ece']:12.5f}")

    print(f"\n  Worst bins (Shadow):")
    for cls in ["home", "draw", "away"]:
        print(f"    {cls}: gap={ece_shadow[cls]['worst_bin_gap']:.4f}  {ece_shadow[cls]['worst_bin']}")
    print(f"  Worst bins (ModelA):")
    for cls in ["home", "draw", "away"]:
        print(f"    {cls}: gap={ece_modela[cls]['worst_bin_gap']:.4f}  {ece_modela[cls]['worst_bin']}")

    # 3) Slices
    print(f"\n{'='*60}")
    print("  3) SLICES")
    print("=" * 60)

    # 3a) By league (top 10 by N)
    from collections import Counter
    league_counts = Counter(league_names)
    top_leagues = [name for name, _ in league_counts.most_common(10)]

    print(f"\n  3a) By league (top 10 by N):")
    print(f"  {'League':25s} {'N':>5s} {'Shadow':>8s} {'ModelA':>8s} {'Market':>8s} {'Skill_S':>8s} {'Skill_A':>8s}")
    print(f"  {'─'*80}")

    for league in top_leagues:
        idx = [i for i in range(N) if league_names[i] == league]
        if len(idx) < 5:
            continue
        yt = [y_true[i] for i in idx]
        sp = [shadow_probs[i] for i in idx]
        mp = [modela_probs[i] for i in idx]
        mkp = [market_probs[i] for i in idx]

        bs = multiclass_brier(yt, sp)
        bm = multiclass_brier(yt, mp)
        bmk = multiclass_brier(yt, mkp)
        sk_s = (1 - bs/bmk) * 100 if bmk > 0 else 0
        sk_m = (1 - bm/bmk) * 100 if bmk > 0 else 0

        print(f"  {league:25s} {len(idx):5d} {bs:8.4f} {bm:8.4f} {bmk:8.4f} {sk_s:+7.1f}% {sk_m:+7.1f}%")

    # 3b) Strong favorite vs parejo
    print(f"\n  3b) By favorite strength (mkt_fav_prob >= 0.55 vs < 0.45):")
    strong_fav = [i for i in range(N) if max(market_probs[i]) >= 0.55]
    parejo = [i for i in range(N) if max(market_probs[i]) < 0.45]

    for label, idx_list in [("Strong fav (>=0.55)", strong_fav), ("Parejo (<0.45)", parejo)]:
        if len(idx_list) < 5:
            print(f"  {label}: N={len(idx_list)} (insufficient)")
            continue
        yt = [y_true[i] for i in idx_list]
        bs = multiclass_brier(yt, [shadow_probs[i] for i in idx_list])
        bm = multiclass_brier(yt, [modela_probs[i] for i in idx_list])
        bmk = multiclass_brier(yt, [market_probs[i] for i in idx_list])
        sk_s = (1 - bs/bmk) * 100 if bmk > 0 else 0
        sk_m = (1 - bm/bmk) * 100 if bmk > 0 else 0
        print(f"  {label:25s} N={len(idx_list):4d}  Shadow={bs:.4f} ModelA={bm:.4f} Market={bmk:.4f}  Skill_S={sk_s:+.1f}% Skill_A={sk_m:+.1f}%")

    # 3c) Draw market alto vs bajo
    print(f"\n  3c) By draw market probability (mkt_draw >= 0.30 vs < 0.30):")
    draw_high = [i for i in range(N) if market_probs[i][1] >= 0.30]
    draw_low = [i for i in range(N) if market_probs[i][1] < 0.30]

    for label, idx_list in [("Draw high (>=0.30)", draw_high), ("Draw low (<0.30)", draw_low)]:
        if len(idx_list) < 5:
            print(f"  {label}: N={len(idx_list)} (insufficient)")
            continue
        yt = [y_true[i] for i in idx_list]
        bs = multiclass_brier(yt, [shadow_probs[i] for i in idx_list])
        bm = multiclass_brier(yt, [modela_probs[i] for i in idx_list])
        bmk = multiclass_brier(yt, [market_probs[i] for i in idx_list])
        sk_s = (1 - bs/bmk) * 100 if bmk > 0 else 0
        sk_m = (1 - bm/bmk) * 100 if bmk > 0 else 0
        print(f"  {label:25s} N={len(idx_list):4d}  Shadow={bs:.4f} ModelA={bm:.4f} Market={bmk:.4f}  Skill_S={sk_s:+.1f}% Skill_A={sk_m:+.1f}%")

    # 4) Conclusion
    print(f"\n{'='*60}")
    print("  4) CONCLUSIÓN OPERATIVA")
    print("=" * 60)

    # Decision logic
    shadow_beats_modela = delta_brier["ci95_hi"] < 0  # Shadow significantly better (lower Brier)
    shadow_worse_modela = delta_brier["ci95_lo"] > 0   # Shadow significantly worse
    shadow_beats_market = skill_shadow > 0
    modela_beats_market = skill_modela > 0

    if shadow_beats_modela and shadow_beats_market:
        verdict = "GO"
        reason = "Shadow supera a Model A (CI95 excluye 0) y al mercado"
    elif shadow_worse_modela:
        verdict = "NO-GO"
        reason = "Shadow es significativamente peor que Model A"
    elif N < 300:
        verdict = "CONTINUE-SHADOW"
        reason = f"N={N} insuficiente para decisión robusta (gate: N≥300)"
    elif not shadow_beats_market and not modela_beats_market:
        verdict = "CONTINUE-SHADOW"
        reason = "Ni Shadow ni Model A superan al mercado; shadow no daña"
    else:
        verdict = "CONTINUE-SHADOW"
        reason = "Diferencia no significativa (CI95 incluye 0); más datos necesarios"

    print(f"\n  VEREDICTO: {verdict}")
    print(f"  Razón: {reason}")
    print(f"\n  Umbrales para GO:")
    print(f"    - N ≥ 300 (actual: {N})")
    print(f"    - ΔBrier(Shadow-ModelA) CI95 < 0 (actual: [{delta_brier['ci95_lo']:+.5f}, {delta_brier['ci95_hi']:+.5f}])")
    print(f"    - Brier Skill vs Market > 0 (actual: Shadow={skill_shadow*100:+.2f}%, ModelA={skill_modela*100:+.2f}%)")

    print(f"\n  5 items a monitorear si CONTINUE-SHADOW:")
    print(f"    1. Coverage: shadow predictions / total matches (ahora {N}/748 = {N/748*100:.0f}% con market)")
    print(f"    2. Staleness: shadow_created_at vs kickoff (pre-KO gap < 2h)")
    print(f"    3. ECE drift: re-evaluar ECE cuando N>300 (draw class más importante)")
    print(f"    4. Calibration drift: shadow draw_prob vs actual draw rate (monitorear cada 100 matches)")
    print(f"    5. Odds source health: Pinnacle coverage ({sum(1 for r in rows if r.get('market_source')=='Pinnacle')}/{N} = {sum(1 for r in rows if r.get('market_source')=='Pinnacle')/N*100:.0f}%)")

    # Save JSON
    output = {
        "meta": {
            "cutoff": CUTOFF,
            "n_intersection": N,
            "market_sources": list(MARKET_SOURCES),
            "n_bootstrap": N_BOOTSTRAP,
            "timestamp": datetime.utcnow().isoformat(),
        },
        "global": {
            "shadow": {"brier": round(brier_shadow, 5), "logloss": round(ll_shadow, 5), "skill_vs_market": round(skill_shadow, 5)},
            "modela": {"brier": round(brier_modela, 5), "logloss": round(ll_modela, 5), "skill_vs_market": round(skill_modela, 5)},
            "market": {"brier": round(brier_market, 5), "logloss": round(ll_market, 5)},
        },
        "delta_shadow_minus_modela": {
            "brier": delta_brier,
            "logloss": delta_ll,
        },
        "calibration": {
            "shadow": ece_shadow,
            "modela": ece_modela,
        },
        "verdict": verdict,
        "reason": reason,
    }

    out_path = Path("scripts/output/shadow_eval_abe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
