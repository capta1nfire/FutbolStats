#!/usr/bin/env python3
"""
Build PIT Dataset (Point-In-Time) - Anti-Leakage Export

Generates a reproducible dataset joining:
- odds_snapshots (lineup_confirmed)
- predictions (created_at <= snapshot_at) -- ANTI-LEAKAGE

Output: DuckDB file for fast local analysis

Usage:
    DATABASE_URL="postgresql://..." python3 scripts/build_pit_dataset.py
    DATABASE_URL="postgresql://..." python3 scripts/build_pit_dataset.py --output data/pit_dataset.duckdb
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

import asyncpg
import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PIT Dataset Query (Anti-Leakage)
PIT_DATASET_QUERY = """
WITH pit_snapshots AS (
    SELECT
        os.id as snapshot_id,
        os.match_id,
        os.snapshot_at,
        os.odds_home as pit_odds_home,
        os.odds_draw as pit_odds_draw,
        os.odds_away as pit_odds_away,
        os.delta_to_kickoff_seconds,
        os.odds_freshness,
        os.bookmaker,
        m.league_id,
        m.external_id as match_external_id,
        m.date as match_date,
        m.home_goals,
        m.away_goals,
        m.status,
        m.home_team_id,
        m.away_team_id,
        ht.name as home_team,
        at.name as away_team,
        -- Canonical odds (cascade-resolved from match_canonical_odds)
        co.odds_home AS canonical_odds_home,
        co.odds_draw AS canonical_odds_draw,
        co.odds_away AS canonical_odds_away,
        co.source AS canonical_odds_source
    FROM odds_snapshots os
    JOIN matches m ON m.id = os.match_id
    LEFT JOIN match_canonical_odds co ON co.match_id = m.id
    LEFT JOIN teams ht ON ht.id = m.home_team_id
    LEFT JOIN teams at ON at.id = m.away_team_id
    WHERE os.snapshot_type = 'lineup_confirmed'
      AND m.status = 'FT'
),
baseline_market_same AS (
    -- Baseline odds from market_movement_snapshots for the SAME bookmaker as PIT (best-effort).
    -- We pick the earliest available pre-kickoff snapshot (max minutes_to_kickoff).
    SELECT DISTINCT ON (mms.match_id, mms.bookmaker)
        mms.match_id,
        mms.bookmaker,
        mms.snapshot_type as baseline_snapshot_type,
        mms.captured_at as baseline_captured_at,
        mms.minutes_to_kickoff as baseline_minutes_to_kickoff,
        mms.odds_home as baseline_odds_home,
        mms.odds_draw as baseline_odds_draw,
        mms.odds_away as baseline_odds_away
    FROM market_movement_snapshots mms
    WHERE mms.captured_at < mms.kickoff_time
    ORDER BY mms.match_id, mms.bookmaker, mms.minutes_to_kickoff DESC, mms.captured_at ASC
),
baseline_market_any AS (
    -- Fallback baseline odds from market_movement_snapshots for ANY bookmaker.
    SELECT DISTINCT ON (mms.match_id)
        mms.match_id,
        mms.bookmaker,
        mms.snapshot_type as baseline_snapshot_type,
        mms.captured_at as baseline_captured_at,
        mms.minutes_to_kickoff as baseline_minutes_to_kickoff,
        mms.odds_home as baseline_odds_home,
        mms.odds_draw as baseline_odds_draw,
        mms.odds_away as baseline_odds_away
    FROM market_movement_snapshots mms
    WHERE mms.captured_at < mms.kickoff_time
    ORDER BY mms.match_id, mms.minutes_to_kickoff DESC, mms.captured_at ASC
),
predictions_asof AS (
    -- Latest prediction BEFORE snapshot (ANTI-LEAKAGE constraint)
    SELECT DISTINCT ON (p.match_id, ps.snapshot_id)
        ps.snapshot_id,
        p.id as prediction_id,
        p.match_id,
        p.home_prob,
        p.draw_prob,
        p.away_prob,
        p.model_version,
        p.created_at as prediction_at
    FROM predictions p
    JOIN pit_snapshots ps ON p.match_id = ps.match_id
    WHERE p.created_at < ps.snapshot_at  -- ANTI-LEAKAGE: prediction must exist STRICTLY before snapshot
    ORDER BY p.match_id, ps.snapshot_id, p.created_at DESC
)
SELECT
    ps.snapshot_id,
    ps.match_id,
    ps.match_external_id,
    ps.league_id,
    ps.home_team,
    ps.away_team,
    ps.match_date,
    ps.snapshot_at,
    ps.delta_to_kickoff_seconds,
    ROUND(ps.delta_to_kickoff_seconds / 60.0, 1) as delta_ko_minutes,
    ps.odds_freshness,
    ps.bookmaker,
    -- PIT odds (at lineup confirmation)
    ps.pit_odds_home,
    ps.pit_odds_draw,
    ps.pit_odds_away,
    -- Canonical odds (cascade-resolved, replaces legacy opening_odds)
    ps.canonical_odds_home,
    ps.canonical_odds_draw,
    ps.canonical_odds_away,
    ps.canonical_odds_source,
    -- Baseline odds (CLV proxy) from market_movement_snapshots
    COALESCE(bms.baseline_odds_home, bma.baseline_odds_home) as baseline_odds_home,
    COALESCE(bms.baseline_odds_draw, bma.baseline_odds_draw) as baseline_odds_draw,
    COALESCE(bms.baseline_odds_away, bma.baseline_odds_away) as baseline_odds_away,
    COALESCE(bms.baseline_snapshot_type, bma.baseline_snapshot_type) as baseline_snapshot_type,
    COALESCE(bms.bookmaker, bma.bookmaker) as baseline_bookmaker,
    COALESCE(bms.baseline_captured_at, bma.baseline_captured_at) as baseline_captured_at,
    COALESCE(bms.baseline_minutes_to_kickoff, bma.baseline_minutes_to_kickoff) as baseline_minutes_to_kickoff,
    CASE
        WHEN bms.baseline_odds_home IS NOT NULL THEN 'market_movement_same_bookmaker'
        WHEN bma.baseline_odds_home IS NOT NULL THEN 'market_movement_any_bookmaker'
        ELSE NULL
    END as baseline_source,
    -- Model predictions (as-of, anti-leakage)
    pa.prediction_id,
    pa.home_prob,
    pa.draw_prob,
    pa.away_prob,
    pa.model_version,
    pa.prediction_at,
    -- Result
    ps.home_goals,
    ps.away_goals,
    ps.status,
    CASE
        WHEN ps.home_goals > ps.away_goals THEN 'H'
        WHEN ps.home_goals < ps.away_goals THEN 'A'
        ELSE 'D'
    END as actual_result
FROM pit_snapshots ps
LEFT JOIN baseline_market_same bms
    ON bms.match_id = ps.match_id AND bms.bookmaker = ps.bookmaker
LEFT JOIN baseline_market_any bma
    ON bma.match_id = ps.match_id
LEFT JOIN predictions_asof pa ON pa.snapshot_id = ps.snapshot_id
ORDER BY ps.snapshot_at DESC
"""


async def fetch_pit_data(database_url: str) -> list[dict]:
    """Fetch PIT dataset from PostgreSQL."""
    logger.info("Connecting to PostgreSQL...")
    conn = await asyncpg.connect(database_url, timeout=60)

    try:
        logger.info("Executing PIT dataset query...")
        rows = await conn.fetch(PIT_DATASET_QUERY)
        logger.info(f"Fetched {len(rows)} PIT records")
        return [dict(row) for row in rows]
    finally:
        await conn.close()


def export_to_duckdb(data: list[dict], output_path: str) -> None:
    """Export PIT dataset to DuckDB."""
    if not data:
        logger.warning("No data to export")
        return

    logger.info(f"Exporting {len(data)} records to DuckDB: {output_path}")

    # Create DuckDB connection
    con = duckdb.connect(output_path)

    # Drop existing table to avoid schema mismatch with old datasets
    con.execute("DROP TABLE IF EXISTS pit_dataset")

    # Create table schema
    con.execute("""
        CREATE TABLE pit_dataset (
            snapshot_id INTEGER,
            match_id INTEGER,
            match_external_id INTEGER,
            league_id INTEGER,
            home_team VARCHAR,
            away_team VARCHAR,
            match_date TIMESTAMP,
            snapshot_at TIMESTAMP,
            delta_to_kickoff_seconds DOUBLE,
            delta_ko_minutes DOUBLE,
            odds_freshness VARCHAR,
            bookmaker VARCHAR,
            -- PIT odds
            pit_odds_home DOUBLE,
            pit_odds_draw DOUBLE,
            pit_odds_away DOUBLE,
            -- Canonical odds (cascade-resolved)
            canonical_odds_home DOUBLE,
            canonical_odds_draw DOUBLE,
            canonical_odds_away DOUBLE,
            canonical_odds_source VARCHAR,
            -- Baseline odds (CLV proxy)
            baseline_odds_home DOUBLE,
            baseline_odds_draw DOUBLE,
            baseline_odds_away DOUBLE,
            baseline_snapshot_type VARCHAR,
            baseline_bookmaker VARCHAR,
            baseline_captured_at TIMESTAMP,
            baseline_minutes_to_kickoff DOUBLE,
            baseline_source VARCHAR,
            -- Model predictions
            prediction_id INTEGER,
            home_prob DOUBLE,
            draw_prob DOUBLE,
            away_prob DOUBLE,
            model_version VARCHAR,
            prediction_at TIMESTAMP,
            -- Result
            home_goals INTEGER,
            away_goals INTEGER,
            status VARCHAR,
            actual_result VARCHAR,
            -- Calculated fields (added after insert)
            ev_home DOUBLE,
            ev_draw DOUBLE,
            ev_away DOUBLE,
            PRIMARY KEY (snapshot_id)
        )
    """)

    # Insert data
    for row in data:
        con.execute("""
            INSERT INTO pit_dataset (
                snapshot_id, match_id, match_external_id, league_id,
                home_team, away_team, match_date, snapshot_at,
                delta_to_kickoff_seconds, delta_ko_minutes, odds_freshness, bookmaker,
                pit_odds_home, pit_odds_draw, pit_odds_away,
                canonical_odds_home, canonical_odds_draw, canonical_odds_away, canonical_odds_source,
                baseline_odds_home, baseline_odds_draw, baseline_odds_away,
                baseline_snapshot_type, baseline_bookmaker, baseline_captured_at,
                baseline_minutes_to_kickoff, baseline_source,
                prediction_id, home_prob, draw_prob, away_prob,
                model_version, prediction_at,
                home_goals, away_goals, status, actual_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row.get("snapshot_id"),
            row.get("match_id"),
            row.get("match_external_id"),
            row.get("league_id"),
            row.get("home_team"),
            row.get("away_team"),
            row.get("match_date"),
            row.get("snapshot_at"),
            row.get("delta_to_kickoff_seconds"),
            row.get("delta_ko_minutes"),
            row.get("odds_freshness"),
            row.get("bookmaker"),
            float(row["pit_odds_home"]) if row.get("pit_odds_home") else None,
            float(row["pit_odds_draw"]) if row.get("pit_odds_draw") else None,
            float(row["pit_odds_away"]) if row.get("pit_odds_away") else None,
            float(row["canonical_odds_home"]) if row.get("canonical_odds_home") else None,
            float(row["canonical_odds_draw"]) if row.get("canonical_odds_draw") else None,
            float(row["canonical_odds_away"]) if row.get("canonical_odds_away") else None,
            row.get("canonical_odds_source"),
            float(row["baseline_odds_home"]) if row.get("baseline_odds_home") else None,
            float(row["baseline_odds_draw"]) if row.get("baseline_odds_draw") else None,
            float(row["baseline_odds_away"]) if row.get("baseline_odds_away") else None,
            row.get("baseline_snapshot_type"),
            row.get("baseline_bookmaker"),
            row.get("baseline_captured_at"),
            float(row["baseline_minutes_to_kickoff"]) if row.get("baseline_minutes_to_kickoff") else None,
            row.get("baseline_source"),
            row.get("prediction_id"),
            float(row["home_prob"]) if row.get("home_prob") else None,
            float(row["draw_prob"]) if row.get("draw_prob") else None,
            float(row["away_prob"]) if row.get("away_prob") else None,
            row.get("model_version"),
            row.get("prediction_at"),
            row.get("home_goals"),
            row.get("away_goals"),
            row.get("status"),
            row.get("actual_result"),
        ])

    # Calculate EV fields
    con.execute("""
        UPDATE pit_dataset
        SET
            ev_home = (home_prob * pit_odds_home) - 1,
            ev_draw = (draw_prob * pit_odds_draw) - 1,
            ev_away = (away_prob * pit_odds_away) - 1
        WHERE home_prob IS NOT NULL
          AND pit_odds_home IS NOT NULL
    """)

    con.close()
    logger.info(f"Export complete: {output_path}")


def generate_summary(output_path: str) -> dict:
    """Generate summary statistics from the DuckDB file."""
    con = duckdb.connect(output_path, read_only=True)

    summary = {}

    # Total records
    result = con.execute("SELECT COUNT(*) FROM pit_dataset").fetchone()
    summary["total_records"] = result[0]

    # Records with predictions (anti-leakage valid)
    result = con.execute("SELECT COUNT(*) FROM pit_dataset WHERE prediction_id IS NOT NULL").fetchone()
    summary["with_predictions"] = result[0]

    # Records with canonical odds
    result = con.execute("SELECT COUNT(*) FROM pit_dataset WHERE canonical_odds_home IS NOT NULL").fetchone()
    summary["with_canonical_odds"] = result[0]

    # Records with baseline odds (market movement fallback)
    result = con.execute("SELECT COUNT(*) FROM pit_dataset WHERE baseline_odds_home IS NOT NULL").fetchone()
    summary["with_baseline_odds"] = result[0]

    # Baseline source distribution
    result = con.execute("""
        SELECT baseline_source, COUNT(*) as c
        FROM pit_dataset
        WHERE baseline_source IS NOT NULL
        GROUP BY 1
        ORDER BY c DESC
    """).fetchall()
    summary["baseline_source_distribution"] = {row[0]: row[1] for row in result}

    # By league
    result = con.execute("""
        SELECT league_id, COUNT(*) as count
        FROM pit_dataset
        GROUP BY league_id
        ORDER BY count DESC
    """).fetchall()
    summary["by_league"] = {row[0]: row[1] for row in result}

    # Delta to kickoff distribution
    result = con.execute("""
        SELECT
            CASE
                WHEN delta_ko_minutes BETWEEN 45 AND 90 THEN 'ideal (45-90)'
                WHEN delta_ko_minutes BETWEEN 10 AND 45 THEN 'aceptable (10-45)'
                WHEN delta_ko_minutes BETWEEN 0 AND 10 THEN 'muy tarde (0-10)'
                WHEN delta_ko_minutes < 0 THEN 'post-kickoff (<0)'
                ELSE 'sin dato'
            END as bucket,
            COUNT(*) as count
        FROM pit_dataset
        GROUP BY 1
        ORDER BY count DESC
    """).fetchall()
    summary["delta_ko_distribution"] = {row[0]: row[1] for row in result}

    # Date range
    result = con.execute("""
        SELECT MIN(match_date), MAX(match_date)
        FROM pit_dataset
    """).fetchone()
    summary["date_range"] = {
        "min": str(result[0]) if result[0] else None,
        "max": str(result[1]) if result[1] else None,
    }

    # Anti-leakage validation (strict)
    result = con.execute("""
        SELECT COUNT(*)
        FROM pit_dataset
        WHERE prediction_at >= snapshot_at
    """).fetchone()
    summary["leakage_violations"] = result[0]

    con.close()
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Build PIT Dataset (Anti-Leakage)")
    parser.add_argument(
        "--output", "-o",
        default="data/pit_dataset.duckdb",
        help="Output DuckDB file path (default: data/pit_dataset.duckdb)"
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Fetch and export
    data = await fetch_pit_data(database_url)
    export_to_duckdb(data, args.output)

    # Generate and print summary
    summary = generate_summary(args.output)

    print("\n" + "=" * 60)
    print("PIT DATASET SUMMARY")
    print("=" * 60)
    print(f"Output file: {args.output}")
    print(f"Total records: {summary['total_records']}")
    print(f"With predictions (as-of): {summary['with_predictions']}")
    print(f"With canonical odds: {summary['with_canonical_odds']}")
    print(f"With baseline odds: {summary.get('with_baseline_odds', 0)}")
    print(f"Leakage violations: {summary['leakage_violations']} (must be 0)")
    print(f"\nDate range: {summary['date_range']['min']} to {summary['date_range']['max']}")
    print(f"\nBy league: {summary['by_league']}")
    print(f"\nDelta KO distribution: {summary['delta_ko_distribution']}")
    print("=" * 60)

    # Save summary to JSON
    import json
    summary_path = args.output.replace(".duckdb", "_summary.json")
    summary["generated_at"] = datetime.utcnow().isoformat()
    summary["output_file"] = args.output
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
