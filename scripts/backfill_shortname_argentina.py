#!/usr/bin/env python3
"""
Backfill short_name for Argentina teams using Promiedos naming convention.

Source: https://www.promiedos.com.ar/league/liga-profesional/hc

ABE Approval: 2026-02-05
"""

import logging
import sys

from _db import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Mapping: team_id -> short_name (from Promiedos)
ARGENTINA_SHORT_NAMES = {
    4142: "Aldosivi",
    4146: "Argentinos",
    4155: "Atl. Tucumán",
    4161: "Banfield",
    4164: "Barracas",
    4166: "Belgrano",
    4149: "Boca",
    4156: "Central Córdoba",
    4150: "Defensa",
    4168: "Riestra",
    6157: "Estudiantes RC",
    4143: "Estudiantes",
    4138: "Gimnasia LP",
    6155: "Gimnasia M",
    4159: "Godoy Cruz",
    4145: "Huracán",
    4157: "Independiente",
    4169: "Ind. Rivadavia",
    4167: "Instituto",
    4148: "Lanús",
    4141: "Newell's",
    4162: "Platense",
    4154: "Racing",
    4160: "River",
    4158: "Central",
    4147: "San Lorenzo",
    6008: "San Martín SJ",
    4163: "Sarmiento",
    4140: "Talleres",
    4165: "Tigre",
    4152: "Unión",
    4144: "Vélez",
}


def run_backfill(apply: bool = False):
    """Update short_name for Argentina teams."""
    logger.info("=" * 60)
    logger.info("Argentina Short Name Backfill (Promiedos)")
    logger.info(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    logger.info("=" * 60)

    conn = get_db_connection()

    updated = 0
    failed = 0

    try:
        with conn.cursor() as cur:
            for team_id, short_name in ARGENTINA_SHORT_NAMES.items():
                if apply:
                    cur.execute(
                        """
                        UPDATE team_wikidata_enrichment
                        SET short_name = %s
                        WHERE team_id = %s
                        """,
                        (short_name, team_id),
                    )
                    if cur.rowcount > 0:
                        updated += 1
                        logger.info(f"  OK: {team_id} -> {short_name}")
                    else:
                        failed += 1
                        logger.warning(f"  SKIP: {team_id} (no enrichment row)")
                else:
                    logger.info(f"  [DRY] {team_id} -> {short_name}")

        if apply:
            conn.commit()
            logger.info("Changes committed")

    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total mappings: {len(ARGENTINA_SHORT_NAMES)}")
    if apply:
        logger.info(f"Updated: {updated}")
        logger.info(f"Skipped: {failed}")
    else:
        logger.info("(DRY-RUN - no updates applied)")
    logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill short_name for Argentina teams")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()
    run_backfill(apply=args.apply)


if __name__ == "__main__":
    main()
