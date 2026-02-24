#!/usr/bin/env python3
"""
Fix Wikidata QIDs using Wikipedia REST API.

Uses Wikipedia's wikibase_item field to get correct QIDs.
Much more reliable than Wikidata reconciliation API for cases
where teams share names with cities.

Usage:
    python scripts/fix_wikidata_via_wikipedia.py --dry-run
    python scripts/fix_wikidata_via_wikipedia.py --apply
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from urllib.parse import quote

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    sys.exit(1)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Teams to fix: ID -> Wikipedia article title
# Format: team_id: "Wikipedia_Article_Title"
TEAMS_TO_FIX = {
    # Spain - conf 0.70 (matched cities)
    1119: "Girona_FC",
    1126: "Valencia_CF",
    1133: "RCD_Espanyol",
    1178: "Real_Oviedo",
    1124: "CA_Osasuna",
    1130: "RCD_Mallorca",
    1180: "Elche_CF",

    # Italy
    1187: "Pisa_S.C.",
    1148: "S.S._Lazio",

    # France - conf 0.70 (matched cities)
    1194: "RC_Lens",
    1198: "OGC_Nice",
    1205: "FC_Lorient",
    1206: "FC_Metz",
    1208: "FC_Nantes",
    1192: "Stade_Rennais_F.C.",

    # Portugal
    2082: "Moreirense_F.C.",
    2084: "C.D._Tondela",
    2088: "Gil_Vicente_F.C.",
    2089: "Sporting_CP",
    2090: "F.C._Arouca",
    2092: "G.D._Estoril_Praia",
    2079: "S.C._Braga",

    # Netherlands - conf 0.70 (matched cities)
    2097: "SC_Heerenveen",
    2099: "PEC_Zwolle",
    2109: "FC_Groningen",
    2110: "PSV_Eindhoven",
    2113: "FC_Utrecht",
    2120: "NEC_Nijmegen",

    # Turkey
    2127: "Fenerbahçe_S.K._(football)",
    2132: "Galatasaray_S.K._(football)",
    2134: "Göztepe_S.K.",
    2141: "Beşiktaş_J.K.",
    2143: "Gençlerbirliği_S.K.",

    # England
    2871: "Preston_North_End_F.C.",

    # USA
    3036: "LA_Galaxy",
    3060: "Charlotte_FC",

    # Brazil - conf 0.70 (matched cities/other)
    3066: "Fortaleza_Esporte_Clube",
    3068: "Clube_de_Regatas_do_Flamengo",
    3070: "Santos_FC",
    3073: "Fluminense_FC",
    3075: "Botafogo_de_Futebol_e_Regatas",
    3077: "Esporte_Clube_Bahia",
    3079: "Sociedade_Esportiva_Palmeiras",
    3081: "CR_Vasco_da_Gama",
    3089: "Esporte_Clube_Vitória",

    # Mexico - conf 0.70 (matched cities)
    4134: "Mazatlán_F.C.",
    4135: "Club_Puebla",
    4136: "C.F._Monterrey",
    4137: "Deportivo_Toluca_F.C.",

    # Chile
    4172: "O'Higgins_F.C.",
    4174: "C.D._Antofagasta",
    4180: "Club_Deportivo_Universidad_Católica",
    4182: "Unión_La_Calera",
    4191: "Deportes_Copiapó",
    6148: "Deportes_Concepción",
    6149: "Rangers_de_Talca",

    # Uruguay
    4224: "Danubio_F.C.",
    4239: "Club_Atlético_Peñarol",
    4245: "Club_Atlético_Progreso",
    4313: "Racing_Club_de_Montevideo",
    6190: "Central_Español_F.C.",

    # Peru
    4235: "Club_Alianza_Lima",
    4282: "Alianza_Universidad",
    4288: "Cusco_FC",

    # Venezuela
    4228: "Caracas_FC",
    4246: "Deportivo_Táchira_F.C.",
    4261: "Monagas_S.C.",
    4271: "Inter_de_Puerto_Cabello",  # Puerto Cabello

    # Ecuador
    4298: "C.D._Técnico_Universitario",
    6144: "C.D._Cuniburo",

    # Paraguay
    4329: "Sportivo_Luqueño",
    6156: "Club_Rubio_Ñu",

    # Argentina
    4156: "Central_Córdoba_de_Santiago_del_Estero",

    # Europe misc
    4378: "KF_Shkëndija",
    4399: "Riga_FC",
    4409: "PAOK_FC",
    4414: "SK_Slavia_Prague",
    4415: "BSC_Young_Boys",
    4418: "K.R.C._Genk",
    4431: "Kuopion_Palloseura",
    4441: "KÍ_Klaksvík",
    4455: "Malmö_FF",
    4461: "AC_Sparta_Prague",
    4464: "Brøndby_IF",
    4469: "Lech_Poznań",
    4482: "Breiðablik_UBK",
    4488: "Ħamrun_Spartans_F.C.",
    4494: "Panathinaikos_F.C.",
    4496: "Aris_Limassol_FC",
    4498: "AEK_Athens_F.C.",
    4499: "Royal_Antwerp_F.C.",
    4556: "FC_Milsami_Orhei",
    4591: "Standard_Liège",
    4636: "R._Charleroi_S.C.",
    4647: "Pafos_FC",
    4682: "Víkingur_Gøta",

    # Other
    6005: "F.C._Alverca",
    6006: "Kocaelispor",
    6015: "FK_Dinamo_Tirana",
    6019: "Araz-Naxçıvan_PFK",
    6024: "FC_Lausanne-Sport",
    6182: "F.C.V._Dender_E.H.",
}


async def fetch_wikipedia_qid(
    title: str,
    client: httpx.AsyncClient,
) -> Optional[dict]:
    """
    Fetch QID from Wikipedia REST API.

    Returns dict with title, description, and wikibase_item (QID).
    """
    try:
        encoded_title = quote(title, safe="")
        response = await client.get(
            f"{WIKIPEDIA_API_BASE}/{encoded_title}",
            headers={
                "User-Agent": "FutbolStats/1.0 (contact@futbolstats.app)",
                "Accept": "application/json",
            },
            timeout=15.0,
            follow_redirects=True,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        qid = data.get("wikibase_item")
        if not qid:
            return None

        return {
            "title": data.get("title"),
            "description": data.get("description"),
            "qid": qid,
        }

    except Exception as e:
        logger.warning(f"  Error fetching {title}: {e}")
        return None


async def main(dry_run: bool = True):
    logger.info("=" * 60)
    logger.info("Wikidata QID Fix via Wikipedia API")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}")
    logger.info(f"Teams to process: {len(TEAMS_TO_FIX)}")
    logger.info("")

    # Connect to DB
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Get current QIDs from DB
    async with async_session() as session:
        result = await session.execute(
            text("""
                SELECT id, name, wikidata_id
                FROM teams
                WHERE id = ANY(:team_ids)
            """),
            {"team_ids": list(TEAMS_TO_FIX.keys())},
        )
        teams_db = {row.id: {"name": row.name, "qid": row.wikidata_id} for row in result.fetchall()}

    # Fetch correct QIDs from Wikipedia
    corrections = []
    errors = []

    async with httpx.AsyncClient() as client:
        for team_id, wiki_title in TEAMS_TO_FIX.items():
            await asyncio.sleep(0.3)  # Rate limit

            team_info = teams_db.get(team_id, {})
            team_name = team_info.get("name", f"ID:{team_id}")
            old_qid = team_info.get("qid")

            wiki_data = await fetch_wikipedia_qid(wiki_title, client)

            if not wiki_data:
                logger.warning(f"  ✗ {team_name}: Wikipedia article not found ({wiki_title})")
                errors.append({"team_id": team_id, "name": team_name, "error": "not_found"})
                continue

            new_qid = wiki_data["qid"]

            if old_qid == new_qid:
                logger.info(f"  = {team_name}: Already correct ({new_qid})")
                continue

            logger.info(f"  → {team_name}: {old_qid} → {new_qid} ({wiki_data['description']})")
            corrections.append({
                "team_id": team_id,
                "team_name": team_name,
                "old_qid": old_qid,
                "new_qid": new_qid,
                "description": wiki_data["description"],
                "wiki_title": wiki_title,
            })

    logger.info("")
    logger.info(f"Corrections found: {len(corrections)}")
    logger.info(f"Errors: {len(errors)}")

    if dry_run:
        logger.info("")
        logger.info("DRY RUN - No changes applied")
        logger.info("Run with --apply to apply corrections")

        # Save corrections to file
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": True,
            "corrections": corrections,
            "errors": errors,
        }
        output_file = f"logs/wikidata_wikipedia_fix_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved to: {output_file}")

    else:
        # Apply corrections
        logger.info("")
        logger.info("Applying corrections...")

        applied = 0
        apply_errors = 0

        for c in corrections:
            async with async_session() as session:
                try:
                    result = await session.execute(
                        text("UPDATE teams SET wikidata_id = :new_qid WHERE id = :team_id"),
                        {"new_qid": c["new_qid"], "team_id": c["team_id"]},
                    )

                    # Clear old enrichment
                    await session.execute(
                        text("DELETE FROM team_wikidata_enrichment WHERE team_id = :team_id"),
                        {"team_id": c["team_id"]},
                    )

                    await session.commit()
                    applied += 1
                    logger.info(f"  ✓ {c['team_name']}: {c['old_qid']} → {c['new_qid']}")

                except Exception as e:
                    await session.rollback()
                    apply_errors += 1
                    logger.error(f"  ✗ {c['team_name']}: {e}")

        logger.info("")
        logger.info(f"Applied: {applied}")
        logger.info(f"Errors: {apply_errors}")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dry_run = not args.apply
    asyncio.run(main(dry_run=dry_run))
