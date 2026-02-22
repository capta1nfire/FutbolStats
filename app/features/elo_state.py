"""Elo state computation for production inference.

Computes current Elo K=10 ratings and K=32 momentum for all teams,
consistent with scripts/feature_lab.py's compute_elo_k_sweep() and
compute_elo_momentum(). Processes per-league for Lab consistency.

Results are cached module-level with 6-hour TTL.
"""

import logging
import time
from collections import defaultdict

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Constants — must match scripts/feature_lab.py exactly
ELO_INITIAL = 1500
ELO_HOME_ADV = 100
ELO_K = 32       # For momentum (compute_elo_momentum)
ELO_K10 = 10     # For elo_k10 (compute_elo_k_sweep, k_val=10)

# Module-level cache (6h TTL)
_cache: dict = {"state": None, "ts": 0.0, "ttl": 21600}


async def get_elo_state(session: AsyncSession) -> dict:
    """Get current Elo K=10 and momentum state for all teams.

    Returns:
        dict: {team_id: {"elo_k10": float, "elo_momentum": float}}
        Cached for 6 hours.
    """
    now = time.time()
    if _cache["state"] is not None and (now - _cache["ts"]) < _cache["ttl"]:
        return _cache["state"]

    state = await _compute(session)
    _cache["state"] = state
    _cache["ts"] = now
    return state


def invalidate_elo_cache():
    """Force recomputation on next call (e.g., after match results update)."""
    _cache["state"] = None
    _cache["ts"] = 0.0


async def _compute(session: AsyncSession) -> dict:
    """Compute Elo K=10 + K=32 momentum per team, per league.

    Processes ALL historical completed matches per league to converge
    ratings, exactly as the Lab does during CSV extraction.
    """
    t0 = time.time()

    result = await session.execute(text("""
        SELECT league_id, home_team_id, away_team_id,
               home_goals, away_goals
        FROM matches
        WHERE status IN ('FT', 'AET', 'PEN')
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY league_id, date
    """))
    rows = result.fetchall()

    # Group by league (consistent with Lab per-league CSVs)
    leagues = defaultdict(list)
    for r in rows:
        leagues[r[0]].append(r)

    state = {}  # team_id -> {elo_k10, elo_momentum}

    for league_id, matches in leagues.items():
        # K=10 system (compute_elo_k_sweep with k_val=10)
        k10 = {}
        # K=32 system + momentum (compute_elo_momentum)
        k32 = {}
        hist = {}

        for m in matches:
            _, h_id, a_id, hg, ag = m

            s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            s_a = 1.0 - s_h

            # ── K=10 update ──
            rh10 = k10.get(h_id, ELO_INITIAL)
            ra10 = k10.get(a_id, ELO_INITIAL)
            exp10 = 1.0 / (1.0 + 10.0 ** ((ra10 - (rh10 + ELO_HOME_ADV)) / 400.0))
            k10[h_id] = rh10 + ELO_K10 * (s_h - exp10)
            k10[a_id] = ra10 + ELO_K10 * (s_a - (1.0 - exp10))

            # ── K=32 + momentum tracking ──
            rh32 = k32.get(h_id, ELO_INITIAL)
            ra32 = k32.get(a_id, ELO_INITIAL)
            exp32 = 1.0 / (1.0 + 10.0 ** ((ra32 - (rh32 + ELO_HOME_ADV)) / 400.0))
            new_rh = rh32 + ELO_K * (s_h - exp32)
            new_ra = ra32 + ELO_K * (s_a - (1.0 - exp32))
            k32[h_id] = new_rh
            k32[a_id] = new_ra
            hist.setdefault(h_id, []).append(new_rh)
            hist.setdefault(a_id, []).append(new_ra)

        # Extract final state per team in this league
        all_teams = set(k10) | set(k32)
        for tid in all_teams:
            current_k32 = k32.get(tid, ELO_INITIAL)
            h = hist.get(tid, [ELO_INITIAL])
            momentum = current_k32 - float(np.mean(h[-5:]))

            state[tid] = {
                "elo_k10": k10.get(tid, ELO_INITIAL),
                "elo_momentum": momentum,
            }

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "[ELO-STATE] Computed state for %d teams from %d matches "
        "across %d leagues in %.0fms",
        len(state), len(rows), len(leagues), elapsed_ms,
    )
    return state
