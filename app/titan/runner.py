"""TITAN Runner - End-to-end extraction and materialization.

This is the missing piece that connects:
1. TitanAPIFootballExtractor (extract odds)
2. TitanJobManager (persist to raw_extractions, handle DLQ)
3. FeatureMatrixMaterializer (build feature_matrix rows)

Usage:
    # From CLI
    python -m app.titan.runner --date 2026-01-25 --league 140

    # From code
    from app.titan.runner import TitanRunner
    runner = TitanRunner(session)
    await runner.run_for_date(date(2026, 1, 25), league_id=140)
"""

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional


def _utc_now() -> datetime:
    """Get current UTC timestamp (timezone-aware) for TIMESTAMPTZ compatibility."""
    return datetime.now(timezone.utc)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.titan.config import get_titan_settings
from app.titan.extractors.api_football import TitanAPIFootballExtractor
from app.titan.jobs.job_manager import TitanJobManager
from app.titan.materializers.feature_matrix import (
    FeatureMatrixMaterializer,
    PITViolationError,
)

logger = logging.getLogger(__name__)
titan_settings = get_titan_settings()


class TitanRunner:
    """End-to-end runner for TITAN extraction and materialization.

    Orchestrates the full pipeline:
    1. Fetch upcoming matches from public.matches
    2. Extract odds from API-Football
    3. Persist extractions to titan.raw_extractions
    4. Compute form/H2H from public.matches
    5. Materialize to titan.feature_matrix
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.extractor = TitanAPIFootballExtractor()
        self.job_manager = TitanJobManager(session)
        self.materializer = FeatureMatrixMaterializer(session)
        self.schema = titan_settings.TITAN_SCHEMA

    async def close(self):
        """Close resources."""
        await self.extractor.close()

    async def run_for_date(
        self,
        target_date: date,
        league_id: Optional[int] = None,
        limit: int = 50,
        dry_run: bool = False,
    ) -> dict:
        """Run extraction and materialization for a specific date.

        Args:
            target_date: Date to process matches for
            league_id: Optional filter by league (e.g., 140 for La Liga)
            limit: Max matches to process
            dry_run: If True, don't persist anything

        Returns:
            Dict with stats: matches_found, extracted, materialized, errors
        """
        stats = {
            "target_date": target_date.isoformat(),
            "league_id": league_id,
            "dry_run": dry_run,
            "matches_found": 0,
            "already_extracted": 0,
            "extracted_success": 0,
            "extracted_failed": 0,
            "materialized": 0,
            "with_xg": 0,  # Tier 1b xG data found
            "with_lineup": 0,  # Tier 1c SofaScore lineup found
            "with_xi_depth": 0,  # Tier 1d XI depth found
            "pit_violations": 0,
            "skipped_no_odds": 0,
            "errors": [],
        }

        logger.info(f"TITAN Runner: Processing {target_date} (league={league_id}, limit={limit})")

        # 1. Get matches for the date from public.matches
        matches = await self._get_matches_for_date(target_date, league_id, limit)
        stats["matches_found"] = len(matches)

        if not matches:
            logger.info(f"No matches found for {target_date}")
            return stats

        logger.info(f"Found {len(matches)} matches to process")

        # 2. Process each match
        for match in matches:
            try:
                result = await self._process_match(match, target_date, dry_run)

                if result["status"] == "already_extracted":
                    stats["already_extracted"] += 1
                elif result["status"] == "extracted":
                    stats["extracted_success"] += 1
                    if result.get("materialized"):
                        stats["materialized"] += 1
                    if result.get("with_xg"):
                        stats["with_xg"] += 1
                    if result.get("with_lineup"):
                        stats["with_lineup"] += 1
                    if result.get("with_xi_depth"):
                        stats["with_xi_depth"] += 1
                    if result.get("skipped_no_odds"):
                        stats["skipped_no_odds"] += 1
                elif result["status"] == "failed":
                    stats["extracted_failed"] += 1
                    if result.get("error"):
                        stats["errors"].append(result["error"])
                elif result["status"] == "pit_violation":
                    stats["pit_violations"] += 1

            except Exception as e:
                logger.error(f"Error processing match {match['external_id']}: {e}")
                stats["errors"].append(f"match_{match['external_id']}: {str(e)}")

        logger.info(
            f"TITAN Runner complete: {stats['extracted_success']} extracted, "
            f"{stats['materialized']} materialized, {stats['extracted_failed']} failed"
        )

        return stats

    async def _get_matches_for_date(
        self,
        target_date: date,
        league_id: Optional[int],
        limit: int,
    ) -> list[dict]:
        """Get matches from public.matches for a date."""
        # public.matches.date is TIMESTAMP (naive), so use naive datetimes
        params = {
            "start": datetime.combine(target_date, datetime.min.time()),
            "end": datetime.combine(target_date, datetime.max.time()),
            "limit": limit,
        }

        league_filter = ""
        if league_id:
            league_filter = "AND m.league_id = :league_id"
            params["league_id"] = league_id

        query = text(f"""
            SELECT
                m.id,
                m.external_id,
                m.date as kickoff_utc,
                m.league_id as competition_id,
                m.season,
                m.home_team_id,
                m.away_team_id,
                m.status
            FROM public.matches m
            WHERE m.date >= :start
              AND m.date <= :end
              AND m.status = 'NS'
              {league_filter}
            ORDER BY m.date
            LIMIT :limit
        """)

        result = await self.session.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "id": row[0],
                "external_id": row[1],
                "kickoff_utc": row[2],
                "competition_id": row[3],
                "season": row[4],
                "home_team_id": row[5],
                "away_team_id": row[6],
                "status": row[7],
            }
            for row in rows
        ]

    async def _process_match(
        self,
        match: dict,
        date_bucket: date,
        dry_run: bool,
    ) -> dict:
        """Process a single match: extract odds and materialize.

        Returns:
            Dict with status and details
        """
        external_id = match["external_id"]
        kickoff_utc = match["kickoff_utc"]

        # Ensure kickoff is in the future (PIT requirement)
        # Make kickoff timezone-aware if naive (from DB)
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)

        now = _utc_now()
        if kickoff_utc <= now:
            return {"status": "pit_violation", "reason": "kickoff_in_past"}

        # 1. Check idempotency BEFORE making API request (saves API quota)
        from app.titan.extractors.base import compute_idempotency_key
        odds_params = {"fixture": external_id, "bookmaker": 8}
        idempotency_key = compute_idempotency_key(
            source_id=self.extractor.SOURCE_ID,
            endpoint="odds",
            params=odds_params,
            date_bucket=date_bucket,
        )

        if await self.job_manager.check_idempotency(idempotency_key):
            logger.debug(f"Match {external_id}: already extracted (pre-check)")
            return {"status": "already_extracted"}

        # 2. Extract odds (only if not already done)
        extraction = await self.extractor.extract_odds(
            fixture_id=external_id,
            date_bucket=date_bucket,
        )

        # 3. Handle extraction result
        if not extraction.is_success:
            if not dry_run:
                await self.job_manager.send_to_dlq(extraction)
            return {
                "status": "failed",
                "error": f"{extraction.error_type}: {extraction.error_message}",
            }

        # 4. Persist extraction
        if not dry_run:
            await self.job_manager.save_extraction(extraction)

        # 5. Parse odds from response
        odds_data = self._parse_odds_from_extraction(extraction)
        if not odds_data:
            logger.debug(f"Match {external_id}: no odds in response")
            return {"status": "extracted", "skipped_no_odds": True}

        # 6. Build odds features
        odds_features = self.materializer.build_odds_features(
            odds_home=odds_data["home"],
            odds_draw=odds_data["draw"],
            odds_away=odds_data["away"],
            captured_at=extraction.captured_at,
        )

        # 7. Compute form (Tier 2) - optional, fail-open
        form_home = None
        form_away = None
        try:
            form_home = await self.materializer.compute_form_features(
                team_id=match["home_team_id"],
                kickoff_utc=kickoff_utc,
            )
            form_away = await self.materializer.compute_form_features(
                team_id=match["away_team_id"],
                kickoff_utc=kickoff_utc,
            )
        except Exception as e:
            logger.warning(f"Form computation failed for match {external_id}: {e}")

        # 8. Compute H2H (Tier 3) - optional, fail-open
        h2h = None
        try:
            h2h = await self.materializer.compute_h2h_features(
                home_team_id=match["home_team_id"],
                away_team_id=match["away_team_id"],
                kickoff_utc=kickoff_utc,
            )
        except Exception as e:
            logger.warning(f"H2H computation failed for match {external_id}: {e}")

        # 8b. Compute xG (Tier 1b) - optional enrichment, fail-open
        xg = None
        try:
            xg = await self.materializer.compute_xg_last5_features(
                home_team_id=match["home_team_id"],
                away_team_id=match["away_team_id"],
                kickoff_utc=kickoff_utc,
            )
        except Exception as e:
            logger.warning(f"xG computation failed for match {external_id}: {e}")

        # 8c. Compute SofaScore Lineup (Tier 1c) - optional enrichment, fail-open
        lineup = None
        try:
            lineup = await self.materializer.compute_lineup_features(
                match_id=match["id"],  # internal ID (public.matches.id), not external_id
                kickoff_utc=kickoff_utc,
            )
        except Exception as e:
            logger.warning(f"Lineup computation failed for match {external_id}: {e}")

        # 8d. Compute XI Depth (Tier 1d) - optional enrichment, fail-open
        xi_depth = None
        try:
            xi_depth = await self.materializer.compute_xi_depth_features(
                match_id=match["id"],  # internal ID (public.matches.id)
                kickoff_utc=kickoff_utc,
                home_formation=lineup.sofascore_home_formation if lineup else None,
                away_formation=lineup.sofascore_away_formation if lineup else None,
            )
        except Exception as e:
            logger.warning(f"XI Depth computation failed for match {external_id}: {e}")

        # 9. Materialize to feature_matrix
        if not dry_run:
            try:
                inserted = await self.materializer.insert_row(
                    match_id=external_id,
                    kickoff_utc=kickoff_utc,
                    competition_id=match["competition_id"],
                    season=match["season"],
                    home_team_id=match["home_team_id"],
                    away_team_id=match["away_team_id"],
                    odds=odds_features,
                    form_home=form_home,
                    form_away=form_away,
                    h2h=h2h,
                    xg=xg,
                    lineup=lineup,
                    xi_depth=xi_depth,
                )
                return {
                    "status": "extracted",
                    "materialized": inserted,
                    "with_xg": xg is not None,
                    "with_lineup": lineup is not None,
                    "with_xi_depth": xi_depth is not None,
                }

            except PITViolationError as e:
                logger.error(f"PIT violation for match {external_id}: {e}")
                return {"status": "pit_violation", "reason": str(e)}

        return {"status": "extracted", "materialized": False, "dry_run": True}

    def _parse_odds_from_extraction(self, extraction) -> Optional[dict]:
        """Parse odds from API-Football extraction response.

        API-Football odds endpoint returns:
        {
            "response": [{
                "bookmakers": [{
                    "bets": [{
                        "name": "Match Winner",
                        "values": [
                            {"value": "Home", "odd": "2.10"},
                            {"value": "Draw", "odd": "3.40"},
                            {"value": "Away", "odd": "3.20"}
                        ]
                    }]
                }]
            }]
        }
        """
        if not extraction.response_body:
            return None

        response = extraction.response_body.get("response", [])
        if not response:
            return None

        # Get first fixture's odds
        fixture_odds = response[0] if response else {}
        bookmakers = fixture_odds.get("bookmakers", [])
        if not bookmakers:
            return None

        # Get first bookmaker (usually Bet365 since we filter by bookmaker=8)
        bookmaker = bookmakers[0]
        bets = bookmaker.get("bets", [])

        # Find "Match Winner" bet
        for bet in bets:
            if bet.get("name") == "Match Winner":
                values = bet.get("values", [])
                odds = {}
                for v in values:
                    if v.get("value") == "Home":
                        odds["home"] = float(v.get("odd", 0))
                    elif v.get("value") == "Draw":
                        odds["draw"] = float(v.get("odd", 0))
                    elif v.get("value") == "Away":
                        odds["away"] = float(v.get("odd", 0))

                if odds.get("home") and odds.get("draw") and odds.get("away"):
                    return odds

        return None


async def run_titan_pipeline(
    target_date: date,
    league_id: Optional[int] = None,
    limit: int = 50,
    dry_run: bool = False,
) -> dict:
    """Standalone function to run TITAN pipeline.

    Can be called from scheduler or CLI.
    """
    async with AsyncSessionLocal() as session:
        runner = TitanRunner(session)
        try:
            return await runner.run_for_date(
                target_date=target_date,
                league_id=league_id,
                limit=limit,
                dry_run=dry_run,
            )
        finally:
            await runner.close()


def main():
    """CLI entrypoint for TITAN runner."""
    parser = argparse.ArgumentParser(description="TITAN OMNISCIENCE Runner")
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Target date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--league",
        type=int,
        default=None,
        help="League ID filter (e.g., 140 for La Liga)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max matches to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't persist anything, just show what would happen",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse date
    try:
        target_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
        return

    # Run
    print(f"\nTITAN Runner - Processing {target_date}")
    print(f"  League: {args.league or 'ALL'}")
    print(f"  Limit: {args.limit}")
    print(f"  Dry run: {args.dry_run}")
    print()

    stats = asyncio.run(run_titan_pipeline(
        target_date=target_date,
        league_id=args.league,
        limit=args.limit,
        dry_run=args.dry_run,
    ))

    # Print results
    print("\n" + "=" * 50)
    print("TITAN Runner Results")
    print("=" * 50)
    print(f"  Matches found:      {stats['matches_found']}")
    print(f"  Already extracted:  {stats['already_extracted']}")
    print(f"  Extracted (new):    {stats['extracted_success']}")
    print(f"  Extraction failed:  {stats['extracted_failed']}")
    print(f"  Materialized:       {stats['materialized']}")
    print(f"  With xG (Tier 1b):  {stats['with_xg']}")
    print(f"  With lineup (1c):   {stats['with_lineup']}")
    print(f"  With XI depth (1d): {stats['with_xi_depth']}")
    print(f"  Skipped (no odds):  {stats['skipped_no_odds']}")
    print(f"  PIT violations:     {stats['pit_violations']}")

    if stats["errors"]:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            print(f"    - {err}")

    print()


if __name__ == "__main__":
    main()
