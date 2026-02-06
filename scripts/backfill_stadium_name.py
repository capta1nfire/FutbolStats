#!/usr/bin/env python3
"""
Backfill stadium_name using Wikidata QIDs.

Problem: SPARQL query doesn't resolve stadiumLabel for referenced entities.
Solution: Batch query Wikidata for labels of stadium QIDs we already have.

ABE Approval: 2026-02-05
Guardrails:
- Only fill where stadium_wikidata_id IS NOT NULL and stadium_name IS NULL
- Uses SPARQL VALUES clause for batch efficiency
- Dry-run mode by default
- Logging with metrics

Usage:
    # Dry-run (default) - show what would be updated
    python3 scripts/backfill_stadium_name.py

    # Apply changes
    python3 scripts/backfill_stadium_name.py --apply

    # Limit batch size for testing
    python3 scripts/backfill_stadium_name.py --apply --limit 50
"""

import argparse
import logging
import sys
from typing import Optional
import requests

from _db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Wikidata SPARQL endpoint
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Batch size for SPARQL VALUES clause (Wikidata handles ~100 well)
BATCH_SIZE = 100


def get_candidates(conn, limit: Optional[int] = None) -> list:
    """
    Get teams with stadium QID but no stadium name.

    Returns list of dicts with team_id, team_name, stadium_wikidata_id.
    """
    query = """
        SELECT
            twe.team_id,
            t.name as team_name,
            t.country,
            twe.stadium_wikidata_id
        FROM team_wikidata_enrichment twe
        JOIN teams t ON twe.team_id = t.id
        WHERE twe.stadium_wikidata_id IS NOT NULL
          AND twe.stadium_name IS NULL
        ORDER BY t.country, t.name
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(query, (limit or 10000,))
        rows = cur.fetchall()

    return [
        {
            "team_id": row[0],
            "team_name": row[1],
            "country": row[2],
            "stadium_qid": row[3],
        }
        for row in rows
    ]


def fetch_stadium_labels_batch(qids: list) -> dict:
    """
    Fetch stadium labels from Wikidata using VALUES clause.

    Args:
        qids: List of Wikidata QIDs (e.g., ["Q499855", "Q739269"])

    Returns:
        Dict mapping QID -> label (e.g., {"Q499855": "La Bombonera"})
    """
    if not qids:
        return {}

    # Build VALUES clause
    values_clause = " ".join(f"wd:{qid}" for qid in qids)

    query = f"""
    SELECT ?stadium ?stadiumLabel WHERE {{
      VALUES ?stadium {{ {values_clause} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en". }}
    }}
    """

    try:
        response = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "FutbolStats/1.0 (contact@futbolstats.app)"},
            timeout=60,
        )

        if response.status_code == 429:
            logger.warning("Rate limited by Wikidata, waiting 60s")
            import time
            time.sleep(60)
            return fetch_stadium_labels_batch(qids)  # Retry

        if response.status_code != 200:
            logger.warning(f"Wikidata SPARQL error: {response.status_code}")
            return {}

        data = response.json()
        bindings = data.get("results", {}).get("bindings", [])

        result = {}
        for binding in bindings:
            stadium_uri = binding.get("stadium", {}).get("value", "")
            label = binding.get("stadiumLabel", {}).get("value", "")

            # Extract QID from URI
            if "/entity/" in stadium_uri:
                qid = stadium_uri.split("/")[-1]
                if label and label != qid:  # Skip if label is just the QID
                    result[qid] = label

        return result

    except requests.RequestException as e:
        logger.error(f"SPARQL request failed: {e}")
        return {}


def update_stadium_name(conn, team_id: int, stadium_name: str) -> bool:
    """Update stadium_name for a team."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE team_wikidata_enrichment
                SET stadium_name = %s
                WHERE team_id = %s
                  AND stadium_name IS NULL
                """,
                (stadium_name, team_id),
            )
        return True
    except Exception as e:
        logger.error(f"Failed to update team {team_id}: {e}")
        return False


def run_backfill(apply: bool = False, limit: Optional[int] = None):
    """
    Main backfill logic.

    Args:
        apply: If True, apply changes. Otherwise dry-run.
        limit: Max number of teams to process.
    """
    logger.info("=" * 60)
    logger.info("Stadium Name Backfill via Wikidata SPARQL")
    logger.info(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    logger.info(f"Limit: {limit or 'None'}")
    logger.info("=" * 60)

    conn = get_db_connection()

    metrics = {
        "candidates": 0,
        "fetched_ok": 0,
        "fetched_fail": 0,
        "updated_ok": 0,
        "updated_fail": 0,
    }

    try:
        # Get candidates
        candidates = get_candidates(conn, limit)
        metrics["candidates"] = len(candidates)

        logger.info(f"Found {len(candidates)} candidates (have QID, missing name)")

        if not candidates:
            logger.info("No candidates to process")
            return metrics

        # Group by stadium QID for batch processing
        qid_to_teams = {}
        for team in candidates:
            qid = team["stadium_qid"]
            if qid not in qid_to_teams:
                qid_to_teams[qid] = []
            qid_to_teams[qid].append(team)

        unique_qids = list(qid_to_teams.keys())
        logger.info(f"Unique stadium QIDs: {len(unique_qids)}")

        # Process in batches
        all_labels = {}
        for batch_start in range(0, len(unique_qids), BATCH_SIZE):
            batch_qids = unique_qids[batch_start:batch_start + BATCH_SIZE]
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (len(unique_qids) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(f"Fetching labels batch {batch_num}/{total_batches} ({len(batch_qids)} QIDs)")

            labels = fetch_stadium_labels_batch(batch_qids)
            all_labels.update(labels)

            # Rate limit between batches
            if batch_start + BATCH_SIZE < len(unique_qids):
                import time
                time.sleep(2)  # 2s between batches

        logger.info(f"Fetched {len(all_labels)} stadium labels")
        metrics["fetched_ok"] = len(all_labels)
        metrics["fetched_fail"] = len(unique_qids) - len(all_labels)

        # Update teams
        for team in candidates:
            qid = team["stadium_qid"]
            stadium_name = all_labels.get(qid)

            if not stadium_name:
                continue

            team_name = team["team_name"]
            country = team["country"]

            if apply:
                success = update_stadium_name(conn, team["team_id"], stadium_name)
                if success:
                    metrics["updated_ok"] += 1
                    logger.info(f"  OK: {team_name} ({country}) -> {stadium_name}")
                else:
                    metrics["updated_fail"] += 1
                    logger.error(f"  FAIL: {team_name} ({country})")
            else:
                logger.info(f"  [DRY] {team_name} ({country}) -> {stadium_name}")

        # Commit all updates
        if apply:
            conn.commit()
            logger.info("Changes committed")

    finally:
        conn.close()

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Candidates:      {metrics['candidates']}")
    logger.info(f"Unique QIDs:     {metrics['fetched_ok'] + metrics['fetched_fail']}")
    logger.info(f"Labels Found:    {metrics['fetched_ok']}")
    logger.info(f"Labels Missing:  {metrics['fetched_fail']}")
    if apply:
        logger.info(f"Updated OK:      {metrics['updated_ok']}")
        logger.info(f"Updated FAIL:    {metrics['updated_fail']}")
    else:
        logger.info("(DRY-RUN - no updates applied)")
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Backfill stadium_name using Wikidata QIDs"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of teams to process",
    )

    args = parser.parse_args()

    metrics = run_backfill(apply=args.apply, limit=args.limit)

    # Exit with error if there were failures
    if metrics.get("updated_fail", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
