"""
Benchmark Matrix endpoint for dashboard.

Returns a league × source matrix with Brier Skill % vs Pinnacle anchor.
Supports our models (Model A v1.0.0, v1.0.1, Shadow) and 13 bookmakers.

ATI P0 guardrails:
- PIT: recorded_at < (kickoff AT TIME ZONE 'UTC'), created_at <= kickoff
- Pairwise intersection: each cell only over matches where source AND Pinnacle exist
- AET/PEN → draw label
- Devig proportional consistent
- Two separate queries to avoid model×bookie inflation
- Bookie allowlist explicit
"""

import logging
import math
import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.ml.devig import devig_proportional
from app.security import verify_dashboard_token
from app.utils.cache import SimpleCache

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(verify_dashboard_token)],
)

# =============================================================================
# Configuration
# =============================================================================

BOOKIE_ALLOWLIST = [
    "Pinnacle", "Bet365", "1xBet", "Betano", "Betfair", "consensus",
    "Marathonbet", "William Hill", "188Bet", "888Sport", "SBO", "Superbet", "10Bet",
]

MODEL_VERSIONS = ["v1.0.0", "v1.0.1-league-only"]

SHADOW_VERSION = "v1.1.0-twostage"

FAMILY_S_VERSION = "v2.0-tier3-family_s"

ALL_CUTOFF = date(2026, 1, 15)  # period=all starts here

VALID_PERIODS = {"30d", "60d", "90d", "all"}

N_BOOTSTRAP = 500  # only for model columns

# =============================================================================
# Pydantic Models
# =============================================================================


class BenchmarkCell(BaseModel):
    skill_pct: Optional[float] = None
    delta_brier: Optional[float] = None
    delta_logloss: Optional[float] = None
    brier_abs: Optional[float] = None
    pinnacle_brier: Optional[float] = None
    logloss_abs: Optional[float] = None
    pinnacle_logloss: Optional[float] = None
    n: int = 0
    confidence_tier: str = "insufficient"
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None


class BenchmarkLeague(BaseModel):
    league_id: int
    name: str
    country: str
    total_resolved: int


class BenchmarkSource(BaseModel):
    key: str
    label: str
    kind: str  # "bookmaker" | "model"
    total_matches: int


class BenchmarkMatrixResponse(BaseModel):
    generated_at: str
    period: str
    anchor: str
    leagues: List[BenchmarkLeague]
    sources: List[BenchmarkSource]
    cells: Dict[str, BenchmarkCell]
    global_row: Dict[str, BenchmarkCell]


# =============================================================================
# Cache
# =============================================================================

_cache = SimpleCache(ttl=300)  # 5 minutes, param-aware

# =============================================================================
# Helpers
# =============================================================================


def _get_cutoff(period: str) -> date:
    """Return cutoff as datetime.date (asyncpg requires native types, not strings)."""
    if period == "all":
        return ALL_CUTOFF
    days = int(period.replace("d", ""))
    return (datetime.utcnow() - timedelta(days=days)).date()


def _get_label(status: str, home_goals: int, away_goals: int) -> int:
    """Label: 0=home, 1=draw, 2=away. AET/PEN => draw."""
    if status in ("AET", "PEN"):
        return 1
    if home_goals > away_goals:
        return 0
    elif home_goals == away_goals:
        return 1
    else:
        return 2


def _multiclass_brier(y_true: list, probs: list) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    score = 0.0
    for i in range(n):
        for k in range(3):
            indicator = 1.0 if y_true[i] == k else 0.0
            score += (probs[i][k] - indicator) ** 2
    return score / n


def _per_match_brier(y_true: list, probs: list) -> list:
    """Return per-match Brier scores."""
    result = []
    for i in range(len(y_true)):
        s = 0.0
        for k in range(3):
            indicator = 1.0 if y_true[i] == k else 0.0
            s += (probs[i][k] - indicator) ** 2
        result.append(s)
    return result


def _multiclass_logloss(y_true: list, probs: list) -> float:
    eps = 1e-15
    n = len(y_true)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        p = max(eps, probs[i][y_true[i]])
        total -= math.log(p)
    return total / n


def _confidence_tier(n: int) -> str:
    if n < 20:
        return "insufficient"
    if n < 50:
        return "low"
    if n < 100:
        return "normal"
    return "confident"


def _variance_ci(per_match_scores: list, n: int) -> tuple:
    """
    Variance-based CI95 for the mean of a per-match series.

    We use this for bookmakers (cheap, scalable) instead of bootstrap.
    Intended semantics: CI on the *delta vs Pinnacle* (per-match paired deltas).
    """
    if n < 5:
        return None, None
    import numpy as np
    arr = np.array(per_match_scores, dtype=float)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    lo = mean - (1.96 * se)
    hi = mean + (1.96 * se)
    return round(lo, 5), round(hi, 5)


def _bootstrap_delta_brier(y_true, probs_source, probs_anchor, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for Brier(source) - Brier(anchor). Paired."""
    rng = random.Random(42)
    n = len(y_true)
    if n < 10:
        return None, None
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        ps = [probs_source[i] for i in idx]
        pa = [probs_anchor[i] for i in idx]
        d = _multiclass_brier(yt, ps) - _multiclass_brier(yt, pa)
        deltas.append(d)
    deltas.sort()
    return round(deltas[int(0.025 * n_boot)], 5), round(deltas[int(0.975 * n_boot)], 5)


def _compute_cell(
    y_true: list,
    probs_source: list,
    probs_pinnacle: list,
    is_model: bool = False,
) -> BenchmarkCell:
    """Compute metrics for a single (league, source) cell on the pairwise intersection."""
    n = len(y_true)
    tier = _confidence_tier(n)

    if n == 0:
        return BenchmarkCell(n=0, confidence_tier="insufficient")

    brier = _multiclass_brier(y_true, probs_source)
    pin_brier = _multiclass_brier(y_true, probs_pinnacle)
    ll = _multiclass_logloss(y_true, probs_source)
    pin_ll = _multiclass_logloss(y_true, probs_pinnacle)

    skill = (1 - brier / pin_brier) * 100 if pin_brier > 0 else None
    delta_b = brier - pin_brier
    delta_ll = ll - pin_ll

    # CI
    ci_lo, ci_hi = None, None
    if is_model and n >= 10:
        ci_lo, ci_hi = _bootstrap_delta_brier(y_true, probs_source, probs_pinnacle)
    elif n >= 5:
        # Paired deltas vs Pinnacle (PIT-safe intersection already enforced by caller)
        pm_s = _per_match_brier(y_true, probs_source)
        pm_p = _per_match_brier(y_true, probs_pinnacle)
        pm_delta = [pm_s[i] - pm_p[i] for i in range(n)]
        ci_lo, ci_hi = _variance_ci(pm_delta, n)

    return BenchmarkCell(
        skill_pct=round(skill, 2) if skill is not None else None,
        delta_brier=round(delta_b, 5),
        delta_logloss=round(delta_ll, 5),
        brier_abs=round(brier, 5),
        pinnacle_brier=round(pin_brier, 5),
        logloss_abs=round(ll, 5),
        pinnacle_logloss=round(pin_ll, 5),
        n=n,
        confidence_tier=tier,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )


# =============================================================================
# SQL Queries
# =============================================================================

RESOLVED_CTE = """
    resolved AS (
        SELECT m.id, m.league_id, m.date, m.status, m.home_goals, m.away_goals
        FROM matches m
        WHERE m.status IN ('FT','AET','PEN')
          AND m.home_goals IS NOT NULL
          AND m.away_goals IS NOT NULL
          AND (m.tainted IS NULL OR m.tainted = false)
          AND m.date >= :cutoff
    )
"""

QUERY_BOOKIES = f"""
    WITH {RESOLVED_CTE},
    last_pre_ko AS (
        SELECT DISTINCT ON (oh.match_id, oh.source)
            oh.match_id, oh.source,
            oh.odds_home, oh.odds_draw, oh.odds_away
        FROM odds_history oh
        JOIN resolved r ON r.id = oh.match_id
        WHERE oh.recorded_at < (r.date AT TIME ZONE 'UTC')
          AND oh.source = ANY(:bookie_allowlist)
          AND (oh.quarantined IS NULL OR oh.quarantined = false)
          AND (oh.tainted IS NULL OR oh.tainted = false)
          AND oh.odds_home > 1.0 AND oh.odds_draw > 1.0 AND oh.odds_away > 1.0
        ORDER BY oh.match_id, oh.source, oh.recorded_at DESC
    )
    SELECT r.id AS match_id, r.league_id, r.date AS kickoff, r.status,
           r.home_goals, r.away_goals,
           lp.source, lp.odds_home, lp.odds_draw, lp.odds_away,
           COALESCE(al.display_name, al.name) AS league_name, COALESCE(al.country, '') AS country
    FROM resolved r
    JOIN last_pre_ko lp ON lp.match_id = r.id
    LEFT JOIN admin_leagues al ON al.league_id = r.league_id
    ORDER BY r.league_id, r.id, lp.source
"""

QUERY_MODELS = f"""
    WITH {RESOLVED_CTE},
    model_preds AS (
        SELECT DISTINCT ON (p.match_id, p.model_version)
            p.match_id, p.model_version,
            p.home_prob, p.draw_prob, p.away_prob
        FROM predictions p
        JOIN resolved r ON r.id = p.match_id
        WHERE p.model_version = ANY(:model_versions)
          AND p.created_at <= r.date
        ORDER BY p.match_id, p.model_version, p.created_at DESC
    ),
    shadow_preds AS (
        SELECT DISTINCT ON (sp.match_id)
            sp.match_id,
            sp.shadow_home_prob, sp.shadow_draw_prob, sp.shadow_away_prob
        FROM shadow_predictions sp
        JOIN resolved r ON r.id = sp.match_id
        WHERE sp.shadow_architecture = 'two_stage'
          AND sp.shadow_version = :shadow_version
          AND sp.created_at <= r.date
        ORDER BY sp.match_id, sp.created_at DESC
    ),
    family_s_preds AS (
        SELECT DISTINCT ON (fp.match_id)
            fp.match_id,
            fp.home_prob AS fs_h, fp.draw_prob AS fs_d, fp.away_prob AS fs_a
        FROM predictions fp
        JOIN resolved r ON r.id = fp.match_id
        WHERE fp.model_version LIKE '%%family_s%%'
          AND fp.created_at <= r.date
        ORDER BY fp.match_id, fp.created_at DESC
    ),
    pinnacle_anchor AS (
        SELECT DISTINCT ON (oh.match_id)
            oh.match_id, oh.odds_home, oh.odds_draw, oh.odds_away
        FROM odds_history oh
        JOIN resolved r ON r.id = oh.match_id
        WHERE oh.source = 'Pinnacle'
          AND oh.recorded_at < (r.date AT TIME ZONE 'UTC')
          AND (oh.quarantined IS NULL OR oh.quarantined = false)
          AND (oh.tainted IS NULL OR oh.tainted = false)
          AND oh.odds_home > 1.0 AND oh.odds_draw > 1.0 AND oh.odds_away > 1.0
        ORDER BY oh.match_id, oh.recorded_at DESC
    )
    SELECT r.id AS match_id, r.league_id, r.date AS kickoff, r.status,
           r.home_goals, r.away_goals,
           mp.model_version, mp.home_prob AS ma_h, mp.draw_prob AS ma_d, mp.away_prob AS ma_a,
           sp.shadow_home_prob AS sh_h, sp.shadow_draw_prob AS sh_d, sp.shadow_away_prob AS sh_a,
           fsp.fs_h, fsp.fs_d, fsp.fs_a,
           pin.odds_home AS pin_h, pin.odds_draw AS pin_d, pin.odds_away AS pin_a,
           COALESCE(al.display_name, al.name) AS league_name, COALESCE(al.country, '') AS country
    FROM resolved r
    LEFT JOIN model_preds mp ON mp.match_id = r.id
    LEFT JOIN shadow_preds sp ON sp.match_id = r.id
    LEFT JOIN family_s_preds fsp ON fsp.match_id = r.id
    LEFT JOIN pinnacle_anchor pin ON pin.match_id = r.id
    LEFT JOIN admin_leagues al ON al.league_id = r.league_id
    ORDER BY r.league_id, r.id
"""

# =============================================================================
# Endpoint
# =============================================================================


@router.get("/benchmark-matrix", response_model=BenchmarkMatrixResponse)
async def get_benchmark_matrix(
    period: str = Query("30d", description="Period: 30d, 60d, 90d, all"),
    db: AsyncSession = Depends(get_async_session),
):
    """Benchmark matrix: league × source with Brier Skill % vs Pinnacle."""

    if period not in VALID_PERIODS:
        period = "30d"

    # Cache check
    cache_hit, cached = _cache.get(params=period)
    if cache_hit:
        return cached

    cutoff = _get_cutoff(period)

    # ── Query A: Bookmakers ──
    result_bk = await db.execute(
        text(QUERY_BOOKIES),
        {"cutoff": cutoff, "bookie_allowlist": BOOKIE_ALLOWLIST},
    )
    bookie_rows = result_bk.fetchall()

    # ── Query B: Models + Pinnacle anchor ──
    result_md = await db.execute(
        text(QUERY_MODELS),
        {
            "cutoff": cutoff,
            "model_versions": MODEL_VERSIONS,
            "shadow_version": SHADOW_VERSION,
        },
    )
    model_rows = result_md.fetchall()

    # ── Build per-match data ──

    # Bookies: group by (match_id) → {source: devigged_probs}
    # Also track Pinnacle probs per match
    match_meta = {}  # match_id → {league_id, league_name, country, label}
    pinnacle_by_match = {}  # match_id → (h,d,a) devigged
    bookie_by_match = defaultdict(dict)  # match_id → {source: (h,d,a)}
    league_resolved_counts = defaultdict(int)  # league_id → total resolved

    for row in bookie_rows:
        mid = row.match_id
        if mid not in match_meta:
            match_meta[mid] = {
                "league_id": row.league_id,
                "league_name": row.league_name or f"ID:{row.league_id}",
                "country": row.country or "",
                "label": _get_label(row.status, row.home_goals, row.away_goals),
            }
        source = row.source
        probs = devig_proportional(float(row.odds_home), float(row.odds_draw), float(row.odds_away))
        if source == "Pinnacle":
            pinnacle_by_match[mid] = probs
        bookie_by_match[mid][source] = probs

    # Models: build per-match model probs + pinnacle anchor (separate from bookies)
    model_data_by_match = {}  # match_id → {model_version: (h,d,a), "Shadow": (h,d,a), "pin": (h,d,a)}

    for row in model_rows:
        mid = row.match_id
        if mid not in match_meta:
            match_meta[mid] = {
                "league_id": row.league_id,
                "league_name": row.league_name or f"ID:{row.league_id}",
                "country": row.country or "",
                "label": _get_label(row.status, row.home_goals, row.away_goals),
            }

        if mid not in model_data_by_match:
            model_data_by_match[mid] = {}

        # Pinnacle anchor
        if row.pin_h is not None and "pin" not in model_data_by_match[mid]:
            pin_probs = devig_proportional(float(row.pin_h), float(row.pin_d), float(row.pin_a))
            model_data_by_match[mid]["pin"] = pin_probs
            # Also populate pinnacle_by_match if not already
            if mid not in pinnacle_by_match:
                pinnacle_by_match[mid] = pin_probs

        # Model A versions
        if row.model_version and row.ma_h is not None:
            mv = row.model_version
            if mv not in model_data_by_match[mid]:
                model_data_by_match[mid][mv] = (float(row.ma_h), float(row.ma_d), float(row.ma_a))

        # Shadow (deduplicated: 1 per match)
        if row.sh_h is not None and "Shadow" not in model_data_by_match[mid]:
            model_data_by_match[mid]["Shadow"] = (float(row.sh_h), float(row.sh_d), float(row.sh_a))

        # Family S (Tier 3 MTV, deduplicated: 1 per match)
        if row.fs_h is not None and "Family_S" not in model_data_by_match[mid]:
            model_data_by_match[mid]["Family_S"] = (float(row.fs_h), float(row.fs_d), float(row.fs_a))

    # Count resolved per league (from match_meta, unique match_ids)
    for mid, meta in match_meta.items():
        league_resolved_counts[meta["league_id"]] += 1

    # ── Compute cells ──
    # Structure: accumulate (y_true, source_probs, pinnacle_probs) per (league_id, source_key)
    accumulator = defaultdict(lambda: {"y": [], "sp": [], "pp": []})
    source_match_counts = defaultdict(set)  # source_key → set of match_ids

    # Bookies
    for mid, sources in bookie_by_match.items():
        if mid not in pinnacle_by_match:
            continue  # no Pinnacle anchor → skip (pairwise intersection)
        meta = match_meta[mid]
        lid = meta["league_id"]
        label = meta["label"]
        pin_probs = pinnacle_by_match[mid]

        for source, probs in sources.items():
            if source == "Pinnacle":
                continue  # Pinnacle is anchor, not a column
            key = f"{lid}:{source}"
            gkey = f"ALL:{source}"
            accumulator[key]["y"].append(label)
            accumulator[key]["sp"].append(probs)
            accumulator[key]["pp"].append(pin_probs)
            accumulator[gkey]["y"].append(label)
            accumulator[gkey]["sp"].append(probs)
            accumulator[gkey]["pp"].append(pin_probs)
            source_match_counts[source].add(mid)

    # Models (from separate query, no inflation)
    for mid, mdata in model_data_by_match.items():
        if "pin" not in mdata:
            continue  # no Pinnacle anchor → skip
        meta = match_meta[mid]
        lid = meta["league_id"]
        label = meta["label"]
        pin_probs = mdata["pin"]

        for source_key in list(MODEL_VERSIONS) + ["Shadow", "Family_S"]:
            if source_key not in mdata:
                continue
            probs = mdata[source_key]
            # Normalize source key for cell dict
            sk = f"Model_A_{source_key}" if source_key in MODEL_VERSIONS else source_key
            key = f"{lid}:{sk}"
            gkey = f"ALL:{sk}"
            accumulator[key]["y"].append(label)
            accumulator[key]["sp"].append(probs)
            accumulator[key]["pp"].append(pin_probs)
            accumulator[gkey]["y"].append(label)
            accumulator[gkey]["sp"].append(probs)
            accumulator[gkey]["pp"].append(pin_probs)
            source_match_counts[sk].add(mid)

    # Compute cell metrics
    cells = {}
    global_row = {}

    for cell_key, data in accumulator.items():
        is_global = cell_key.startswith("ALL:")
        source_key = cell_key.split(":", 1)[1]
        is_model = source_key.startswith("Model_A_") or source_key in ("Shadow", "Family_S")

        cell = _compute_cell(data["y"], data["sp"], data["pp"], is_model=is_model)

        if is_global:
            global_row[source_key] = cell
        else:
            cells[cell_key] = cell

    # ── Build response metadata ──

    # Leagues (sorted by total resolved desc)
    league_info = {}
    for mid, meta in match_meta.items():
        lid = meta["league_id"]
        if lid not in league_info:
            league_info[lid] = {
                "league_id": lid,
                "name": meta["league_name"],
                "country": meta["country"],
            }

    leagues = [
        BenchmarkLeague(
            league_id=lid,
            name=info["name"],
            country=info["country"],
            total_resolved=league_resolved_counts.get(lid, 0),
        )
        for lid, info in sorted(
            league_info.items(),
            key=lambda x: league_resolved_counts.get(x[0], 0),
            reverse=True,
        )
    ]

    # Sources
    all_source_keys = []
    # Models first
    for mv in MODEL_VERSIONS:
        sk = f"Model_A_{mv}"
        if sk in source_match_counts:
            all_source_keys.append(BenchmarkSource(
                key=sk,
                label=f"Model A {mv}",
                kind="model",
                total_matches=len(source_match_counts[sk]),
            ))
    if "Shadow" in source_match_counts:
        all_source_keys.append(BenchmarkSource(
            key="Shadow",
            label=f"Shadow {SHADOW_VERSION}",
            kind="model",
            total_matches=len(source_match_counts["Shadow"]),
        ))
    if "Family_S" in source_match_counts:
        all_source_keys.append(BenchmarkSource(
            key="Family_S",
            label=f"Family S {FAMILY_S_VERSION}",
            kind="model",
            total_matches=len(source_match_counts["Family_S"]),
        ))
    # Then bookmakers (sorted by total matches desc), excluding Pinnacle (it's the anchor)
    for bk in sorted(
        [b for b in BOOKIE_ALLOWLIST if b != "Pinnacle"],
        key=lambda b: len(source_match_counts.get(b, set())),
        reverse=True,
    ):
        if bk in source_match_counts:
            all_source_keys.append(BenchmarkSource(
                key=bk,
                label=bk,
                kind="bookmaker",
                total_matches=len(source_match_counts[bk]),
            ))

    response = BenchmarkMatrixResponse(
        generated_at=datetime.utcnow().isoformat(),
        period=period,
        anchor="Pinnacle",
        leagues=leagues,
        sources=all_source_keys,
        cells=cells,
        global_row=global_row,
    )

    _cache.set(response, params=period)
    return response
