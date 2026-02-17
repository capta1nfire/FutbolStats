#!/usr/bin/env python3
"""
Build Historical MTV (Match Talent Variance) Features.

Materializes talent_delta for matches 2023+ into parquet using EXACT
production functions from app/features/engineering.py (zero Train-Serve Skew).

GDT D2: No proxies. Uses predict_expected_xi_injury_aware() + full cascade.
ABE P0-1: xi_window=15 (XI prediction), pts_limit=10 (PTS rolling).
ABE P0-2: Canary-first protocol. Run --league 39 first, report metrics.

Usage:
    source .env
    python scripts/build_historical_mtv.py --league 39              # Canary EPL
    python scripts/build_historical_mtv.py --resume --concurrency 10 # Full run

TM Injuries integration (offline, explicit):
    python scripts/build_historical_mtv.py --league 39 \
        --tm-injuries data/tm_injuries_by_match.parquet
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features.engineering import compute_match_talent_delta_features
from app.models import Match

# -- ABE P0-1: Named constants, registered in metadata --
XI_WINDOW = 15    # Matches recientes para predecir expected XI
PTS_LIMIT = 10    # Matches con rating para calcular PTS rolling

# -- TM confidence threshold (ABE: hiConf vs midConf) --
TM_HI_CONF_THRESHOLD = 0.95

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_mtv")


def create_script_engine(concurrency):
    """Create async engine optimized for batch processing.

    Independent from app/database.py to avoid global engine initialization
    and to use batch-appropriate pool/timeout settings.
    """
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Run: source .env")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    pool = max(concurrency + 5, 15)
    return create_async_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        pool_size=pool,
        max_overflow=10,
        pool_recycle=300,
        pool_timeout=60,
        connect_args={
            "server_settings": {"statement_timeout": "120000"}  # 120s
        },
    )


async def load_match_ids(engine, league_id=None):
    """Load eligible match IDs (lightweight query, no ORM objects)."""
    async with sessionmaker(engine, class_=AsyncSession)() as session:
        params = {}
        where = (
            "WHERE m.status = 'FT' "
            "AND m.date >= '2023-01-01' "
            "AND m.tainted = false"
        )
        if league_id is not None:
            where += " AND m.league_id = :league_id"
            params["league_id"] = league_id

        result = await session.execute(
            text(
                f"SELECT m.id, m.league_id, m.date "
                f"FROM matches m {where} "
                f"ORDER BY m.date, m.id"
            ),
            params,
        )
        rows = result.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]


def load_existing_ids(output_path):
    """Load already-processed match_ids for resume capability."""
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_parquet(output_path, columns=["match_id"])
        return set(existing["match_id"].tolist())
    except Exception:
        return set()


def load_tm_exclusions(parquet_path, mapping_path=None):
    """Load TM injury exclusions per match x team (ABE: offline/explicit).

    Args:
        parquet_path: Path to tm_injuries_by_match.parquet
        mapping_path: Optional path to tm_player_mapping.json for confidence

    Returns:
        exclusions: {(match_id, team_id): set(player_external_ids)}
        conf_map: {player_external_id: 'hi'|'mid'} confidence classification
    """
    pq = Path(parquet_path)
    if not pq.exists():
        raise FileNotFoundError(f"TM injuries parquet not found: {pq}")

    df = pd.read_parquet(pq)
    logger.info(
        "TM injuries loaded: %d records from %s", len(df), pq.name
    )

    exclusions = {}
    for _, row in df.iterrows():
        key = (int(row["match_id"]), int(row["team_id"]))
        if key not in exclusions:
            exclusions[key] = set()
        exclusions[key].add(int(row["player_external_id"]))

    # Confidence map from player mapping
    conf_map = {}
    mp = Path(mapping_path) if mapping_path else None
    if mp and mp.exists():
        with open(mp) as f:
            mapping = json.load(f)
        for entry in mapping.get("mappings", []):
            af_id = entry["af_id"]
            score = entry.get("score", 0)
            conf_map[af_id] = "hi" if score >= TM_HI_CONF_THRESHOLD else "mid"
        logger.info(
            "TM confidence map: %d players (%d hi, %d mid)",
            len(conf_map),
            sum(1 for v in conf_map.values() if v == "hi"),
            sum(1 for v in conf_map.values() if v == "mid"),
        )
    else:
        logger.info("TM confidence map: not available (no mapping file)")

    return exclusions, conf_map


def log_tm_coverage(records, tm_exclusions, conf_map, all_matches):
    """Log TM injury coverage per league (ABE: coverage logging).

    Reports: %matches with >=1 exclusion, avg_excluded, hiConf vs midConf.
    """
    if not tm_exclusions:
        return {}

    # Group matches by league
    by_league = defaultdict(list)
    for mid, lid, _ in all_matches:
        by_league[lid].append(mid)

    # Build per-match TM stats from records
    tm_stats_by_match = {}
    for r in records:
        mid = r["match_id"]
        h_exc = r.get("tm_home_excluded", 0)
        a_exc = r.get("tm_away_excluded", 0)
        tm_stats_by_match[mid] = h_exc + a_exc

    # Classify exclusions by confidence
    all_hi = 0
    all_mid = 0
    for key, player_ids in tm_exclusions.items():
        for pid in player_ids:
            c = conf_map.get(pid, "mid")
            if c == "hi":
                all_hi += 1
            else:
                all_mid += 1

    logger.info("=" * 60)
    logger.info("TM INJURY COVERAGE BY LEAGUE:")
    logger.info(
        "%-8s %8s %8s %8s %10s",
        "League", "Matches", "w/Excl", "%Cover", "AvgExcl",
    )
    logger.info("-" * 50)

    coverage_report = {}
    for lid in sorted(by_league.keys()):
        match_ids = by_league[lid]
        n_total = len(match_ids)
        n_with_excl = 0
        total_excl = 0
        for mid in match_ids:
            exc = tm_stats_by_match.get(mid, 0)
            if exc > 0:
                n_with_excl += 1
                total_excl += exc

        pct = 100 * n_with_excl / n_total if n_total > 0 else 0
        avg = total_excl / n_with_excl if n_with_excl > 0 else 0

        logger.info(
            "%-8d %8d %8d %7.1f%% %9.1f",
            lid, n_total, n_with_excl, pct, avg,
        )
        coverage_report[str(lid)] = {
            "matches": n_total,
            "with_exclusions": n_with_excl,
            "coverage_pct": round(pct, 1),
            "avg_excluded": round(avg, 2),
        }

    logger.info("-" * 50)
    logger.info(
        "Confidence: %d hiConf (score>=%.2f), %d midConf",
        all_hi, TM_HI_CONF_THRESHOLD, all_mid,
    )
    logger.info("=" * 60)

    coverage_report["_confidence"] = {"hi": all_hi, "mid": all_mid}
    return coverage_report


async def process_match(session_maker, match_id, semaphore, tm_exclusions=None):
    """Process a single match using EXACT production functions."""
    async with semaphore:
        try:
            async with session_maker() as session:
                # Load Match ORM object (GDT: fidelity 1:1 with production)
                result = await session.execute(
                    text("SELECT * FROM matches WHERE id = :id"),
                    {"id": match_id},
                )
                row = result.mappings().fetchone()
                if not row:
                    return {"match_id": match_id, "error": "not_found"}

                # Build Match model from row data
                match = Match(**{k: row[k] for k in row.keys()})

                # TM exclusions: team-scoped (ABE P0)
                home_extra = None
                away_extra = None
                if tm_exclusions:
                    home_extra = tm_exclusions.get(
                        (match_id, row["home_team_id"])
                    )
                    away_extra = tm_exclusions.get(
                        (match_id, row["away_team_id"])
                    )

                # EXACT production function call
                # ABE P0-1: window=XI_WINDOW (15), limit_matches=PTS_LIMIT (10)
                features = await compute_match_talent_delta_features(
                    session,
                    match=match,
                    window=XI_WINDOW,
                    limit_matches=PTS_LIMIT,
                    include_doubtful=False,
                    home_extra_excluded=home_extra,
                    away_extra_excluded=away_extra,
                )

                out = {
                    "match_id": match_id,
                    "league_id": row["league_id"],
                    "date": row["date"],
                    **features,
                }
                # Track TM exclusion counts for coverage logging
                if tm_exclusions is not None:
                    out["tm_home_excluded"] = len(home_extra) if home_extra else 0
                    out["tm_away_excluded"] = len(away_extra) if away_extra else 0

                return out
        except Exception as e:
            logger.warning("Match %d: %s", match_id, str(e)[:200])
            return {"match_id": match_id, "error": str(e)[:200]}


async def run(args):
    engine = create_script_engine(args.concurrency)
    session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load TM exclusions if provided (ABE: offline/explicit CLI flag)
    tm_exclusions = None
    tm_conf_map = {}
    if args.tm_injuries:
        mapping_path = Path(args.tm_injuries).parent / "tm_player_mapping.json"
        tm_exclusions, tm_conf_map = load_tm_exclusions(
            args.tm_injuries, str(mapping_path)
        )

    # Load match IDs
    logger.info("Loading eligible matches...")
    all_matches = await load_match_ids(engine, league_id=args.league)
    logger.info("Total eligible matches: %d", len(all_matches))

    # Resume: skip already processed
    if args.resume:
        existing = load_existing_ids(output_path)
        all_matches = [
            (mid, lid, d) for mid, lid, d in all_matches if mid not in existing
        ]
        logger.info(
            "After resume filter: %d remaining (%d already done)",
            len(all_matches),
            len(existing),
        )

    if not all_matches:
        logger.info("Nothing to process.")
        await engine.dispose()
        return

    # Process in batches (ABE P0-2: capture per-batch metrics)
    BATCH_SIZE = 500
    all_results = []
    batch_durations = []
    batch_errors_list = []
    total_errors = 0
    total_timeouts = 0
    start_time = time.time()

    for batch_start in range(0, len(all_matches), BATCH_SIZE):
        batch = all_matches[batch_start : batch_start + BATCH_SIZE]
        batch_t0 = time.time()

        tasks = [
            process_match(session_maker, mid, semaphore, tm_exclusions)
            for mid, _, _ in batch
        ]
        batch_results = await asyncio.gather(*tasks)

        batch_dur = time.time() - batch_t0
        batch_durations.append(batch_dur)

        # Count errors/timeouts in batch
        batch_err = 0
        for r in batch_results:
            if "error" in r:
                batch_err += 1
                if "timeout" in str(r.get("error", "")).lower():
                    total_timeouts += 1
        batch_errors_list.append(batch_err)
        total_errors += batch_err

        all_results.extend(batch_results)

        done = batch_start + len(batch)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(all_matches) - done) / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d (%.1f%%) batch=%.1fs errors=%d rate=%.1f/s ETA=%.1fmin",
            done,
            len(all_matches),
            100 * done / len(all_matches),
            batch_dur,
            batch_err,
            rate,
            eta / 60,
        )

    # Build DataFrame
    records = []
    for r in all_results:
        if "error" in r:
            continue
        rec = {
            "match_id": r["match_id"],
            "league_id": r["league_id"],
            "date": r["date"],
            "home_talent_delta": r.get("home_talent_delta"),
            "away_talent_delta": r.get("away_talent_delta"),
            "talent_delta_diff": r.get("talent_delta_diff"),
            "shock_magnitude": r.get("shock_magnitude"),
            "talent_delta_missing": r.get("talent_delta_missing", 1),
        }
        if tm_exclusions is not None:
            rec["tm_home_excluded"] = r.get("tm_home_excluded", 0)
            rec["tm_away_excluded"] = r.get("tm_away_excluded", 0)
        records.append(rec)

    df = pd.DataFrame(records)

    # Resume: merge with existing
    if args.resume and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=["match_id"], keep="last")

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    df.to_parquet(output_path, index=False)

    elapsed_total = time.time() - start_time

    # -- ABE P0-2: Canary metrics --
    p50_batch = statistics.median(batch_durations) if batch_durations else 0
    p95_batch = (
        sorted(batch_durations)[int(len(batch_durations) * 0.95)]
        if len(batch_durations) > 1
        else (batch_durations[0] if batch_durations else 0)
    )

    canary_metrics = {
        "duration_s": round(elapsed_total, 1),
        "n_batches": len(batch_durations),
        "p50_batch_s": round(p50_batch, 2),
        "p95_batch_s": round(p95_batch, 2),
        "errors": total_errors,
        "timeouts": total_timeouts,
    }

    logger.info("=" * 60)
    logger.info("CANARY METRICS:")
    logger.info("  Total duration: %.1f min", elapsed_total / 60)
    logger.info(
        "  Batches: %d, p50=%.1fs, p95=%.1fs",
        len(batch_durations),
        p50_batch,
        p95_batch,
    )
    logger.info(
        "  Errors: %d (%.1f%%)",
        total_errors,
        100 * total_errors / len(all_matches) if all_matches else 0,
    )
    logger.info("  Timeouts: %d", total_timeouts)
    logger.info("=" * 60)
    logger.info(
        "Done: %d success, %d errors, %d total rows in parquet.",
        len(records),
        total_errors,
        len(df),
    )

    # -- TM coverage report (ABE: hiConf vs midConf) --
    tm_coverage = {}
    if tm_exclusions is not None:
        tm_coverage = log_tm_coverage(
            records, tm_exclusions, tm_conf_map, all_matches
        )

    # -- ABE P0-1: Save metadata --
    n_with_delta = int(df["home_talent_delta"].notna().sum())
    n_missing = int(df["talent_delta_missing"].sum())
    metadata = {
        "xi_window": XI_WINDOW,
        "pts_limit": PTS_LIMIT,
        "include_doubtful": False,
        "min_date": "2023-01-01",
        "league_filter": args.league,
        "tm_injuries": args.tm_injuries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_matches": len(all_matches),
        "success": len(records),
        "errors": total_errors,
        "rows_in_parquet": len(df),
        "with_talent_delta": n_with_delta,
        "talent_delta_missing": n_missing,
        "coverage_pct": round(100 * n_with_delta / len(df), 1) if len(df) > 0 else 0,
        "canary_metrics": canary_metrics,
    }
    if tm_coverage:
        metadata["tm_coverage"] = tm_coverage

    meta_path = output_path.with_suffix(".json").parent / "historical_mtv_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved to %s", meta_path)

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(
        description="Build Historical MTV Features (Parquet)"
    )
    parser.add_argument(
        "--output",
        default="data/historical_mtv_features.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--league",
        type=int,
        default=None,
        help="Restrict to single league_id (canary mode)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent DB sessions (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip matches already in parquet",
    )
    parser.add_argument(
        "--tm-injuries",
        type=str,
        default=None,
        help="Path to TM injuries parquet (offline injection, zero prod impact)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
