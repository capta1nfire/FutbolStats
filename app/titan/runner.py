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
from sqlalchemy.exc import IntegrityError
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
            "rematerialized": 0,  # Re-materialized from internal sources (no API call)
            "odds_history_newer": 0,  # Fresher odds found in odds_history
            "skipped_pit_refresh": 0,  # Skipped refresh because snapshot >= kickoff
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
                    # Re-materialization counters (when already_extracted but refreshed)
                    if result.get("rematerialized"):
                        stats["rematerialized"] += 1
                        stats["materialized"] += 1
                    if result.get("odds_history_newer"):
                        stats["odds_history_newer"] += 1
                    if result.get("skipped_pit_refresh"):
                        stats["skipped_pit_refresh"] += 1
                    if result.get("with_xg"):
                        stats["with_xg"] += 1
                    if result.get("with_lineup"):
                        stats["with_lineup"] += 1
                    if result.get("with_xi_depth"):
                        stats["with_xi_depth"] += 1
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
                # CRITICAL: Rollback to prevent cascade failures (InFailedSQLTransactionError)
                # When a DB operation fails, the connection stays in error state until rollback
                try:
                    await self.session.rollback()
                except Exception:
                    pass  # Best effort - connection may already be broken
                logger.error(
                    f"Error processing match {match['external_id']}: "
                    f"[{type(e).__name__}] {e}"
                )
                stats["errors"].append(f"match_{match['external_id']}: [{type(e).__name__}] {str(e)}")

        logger.info(
            f"TITAN Runner complete: {stats['extracted_success']} extracted, "
            f"{stats['materialized']} materialized, {stats['extracted_failed']} failed, "
            f"{stats['rematerialized']} rematerialized, {stats['odds_history_newer']} odds_history_newer"
        )

        return stats

    async def _get_matches_for_date(
        self,
        target_date: date,
        league_id: Optional[int],
        limit: int,
    ) -> list[dict]:
        """Get matches from public.matches for a date."""
        params = {
            "start": datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc),
            "end": datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc),
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
            logger.debug(f"Match {external_id}: already extracted (pre-check), attempting re-materialization")
            # Already extracted → NO API call, but re-materialize from internal sources
            if not dry_run:
                try:
                    remat_result = await self._rematerialize_from_internal(match, kickoff_utc)
                    return {
                        "status": "already_extracted",
                        **remat_result,
                    }
                except Exception as e:
                    logger.warning(f"Match {external_id}: re-materialization failed: {e}")
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
            try:
                await self.session.rollback()
            except Exception:
                pass

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
            try:
                await self.session.rollback()
            except Exception:
                pass

        # 8b. Compute xG (Tier 1b) - optional enrichment, fail-open
        # competition_id = m.league_id (API-Football league_id; see _get_matches_for_date() SELECT line 202)
        xg = None
        try:
            xg = await self.materializer.compute_xg_last5_features(
                home_team_id=match["home_team_id"],
                away_team_id=match["away_team_id"],
                kickoff_utc=kickoff_utc,
                league_id=match.get("competition_id"),  # m.league_id aliased as competition_id
            )
        except Exception as e:
            logger.warning(f"xG computation failed for match {external_id}: {e}")
            try:
                await self.session.rollback()
            except Exception:
                pass

        # 8c. Compute SofaScore Lineup (Tier 1c) - optional enrichment, fail-open
        lineup = None
        try:
            lineup = await self.materializer.compute_lineup_features(
                match_id=match["id"],  # internal ID (public.matches.id), not external_id
                kickoff_utc=kickoff_utc,
            )
        except Exception as e:
            logger.warning(f"Lineup computation failed for match {external_id}: {e}")
            try:
                await self.session.rollback()
            except Exception:
                pass

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
            try:
                await self.session.rollback()
            except Exception:
                pass

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
            except IntegrityError as e:
                err_msg = str(e.orig) if hasattr(e, "orig") else str(e)
                if "pit_valid" in err_msg:
                    logger.error(
                        f"DB pit_valid check failed for match {external_id} "
                        f"(likely kickoff reschedule): {err_msg}"
                    )
                    try:
                        await self.session.rollback()
                    except Exception:
                        pass
                    return {"status": "pit_violation", "reason": err_msg}
                raise

        return {"status": "extracted", "materialized": False, "dry_run": True}

    async def _fetch_latest_odds_from_history(
        self,
        internal_match_id: int,
        kickoff_utc: datetime,
    ) -> Optional["OddsFeatures"]:
        """Fetch the latest odds snapshot from public.odds_history.

        PIT-safe: Only returns snapshots where recorded_at < kickoff_utc.
        Skips quarantined/tainted rows.

        Args:
            internal_match_id: public.matches.id
            kickoff_utc: Match kickoff time (aware UTC)

        Returns:
            OddsFeatures built from the latest clean snapshot, or None.
        """
        query = text("""
            SELECT odds_home, odds_draw, odds_away, recorded_at
            FROM public.odds_history
            WHERE match_id = :match_id
              AND recorded_at < :kickoff
              AND (quarantined IS NOT TRUE)
              AND (tainted IS NOT TRUE)
              AND odds_home IS NOT NULL
              AND odds_draw IS NOT NULL
              AND odds_away IS NOT NULL
            ORDER BY recorded_at DESC
            LIMIT 1
        """)
        result = await self.session.execute(query, {
            "match_id": internal_match_id,
            "kickoff": kickoff_utc,
        })
        row = result.fetchone()
        if not row:
            return None

        odds_home, odds_draw, odds_away, recorded_at = row
        return self.materializer.build_odds_features(
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            captured_at=recorded_at,
        )

    async def _get_current_fm_odds_captured_at(
        self,
        external_id: int,
    ) -> Optional[datetime]:
        """Get current odds_captured_at from feature_matrix for staleness comparison."""
        query = text(f"""
            SELECT odds_captured_at
            FROM {self.schema}.feature_matrix
            WHERE match_id = :match_id
        """)
        result = await self.session.execute(query, {"match_id": external_id})
        row = result.fetchone()
        return row[0] if row else None

    async def _rematerialize_from_internal(
        self,
        match: dict,
        kickoff_utc: datetime,
    ) -> dict:
        """Re-materialize feature_matrix row from internal sources (no API call).

        Called when odds extraction was already done (idempotency hit) but:
        - odds_history may have a fresher snapshot than feature_matrix.odds_captured_at
        - xG/form/H2H/lineup/xi_depth can always be recalculated (internal reads)

        This closes the "freshness gap" where feature_matrix gets stale because
        the idempotency pre-check prevented any update.

        Returns:
            Dict with rematerialization details for stats tracking.
        """
        external_id = match["external_id"]
        result = {
            "rematerialized": False,
            "odds_history_newer": False,
            "skipped_pit_refresh": False,
            "with_xg": False,
            "with_lineup": False,
            "with_xi_depth": False,
        }

        # 1. Check if odds_history has a fresher snapshot
        current_fm_captured = await self._get_current_fm_odds_captured_at(external_id)
        history_odds = await self._fetch_latest_odds_from_history(
            internal_match_id=match["id"],
            kickoff_utc=kickoff_utc,
        )

        odds_features = None
        if history_odds:
            if current_fm_captured is None:
                # No FM row yet (shouldn't happen if already_extracted, but defensive)
                odds_features = history_odds
                result["odds_history_newer"] = True
            else:
                # Ensure both are tz-aware for comparison
                fm_ts = current_fm_captured
                if fm_ts.tzinfo is None:
                    fm_ts = fm_ts.replace(tzinfo=timezone.utc)
                hist_ts = history_odds.captured_at
                if hist_ts.tzinfo is None:
                    hist_ts = hist_ts.replace(tzinfo=timezone.utc)

                # Only use history odds if they're meaningfully newer (> 30 min)
                delta_minutes = (hist_ts - fm_ts).total_seconds() / 60
                if delta_minutes > 30:
                    # PIT check: snapshot must be before kickoff
                    if hist_ts < kickoff_utc:
                        odds_features = history_odds
                        result["odds_history_newer"] = True
                        logger.info(
                            f"Match {external_id}: odds_history is {delta_minutes:.0f}min newer "
                            f"({fm_ts.isoformat()} → {hist_ts.isoformat()})"
                        )
                    else:
                        result["skipped_pit_refresh"] = True
                        logger.debug(
                            f"Match {external_id}: odds_history snapshot >= kickoff, skipping PIT"
                        )

        # If no fresher odds from history, use existing FM odds (read back from DB)
        # We still want to re-materialize xG/form/H2H/lineup/xi to refresh those timestamps
        if odds_features is None:
            # Read current odds from feature_matrix to pass through
            current_odds = await self._read_current_fm_odds(external_id)
            if current_odds is None:
                logger.debug(f"Match {external_id}: no odds in FM or history, skip rematerialization")
                return result
            odds_features = current_odds

        # 2. Compute all internal tiers (same as fresh extraction path)
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
            logger.warning(f"Remat form failed for match {external_id}: {e}")
            try:
                await self.session.rollback()
            except Exception:
                pass

        h2h = None
        try:
            h2h = await self.materializer.compute_h2h_features(
                home_team_id=match["home_team_id"],
                away_team_id=match["away_team_id"],
                kickoff_utc=kickoff_utc,
            )
        except Exception as e:
            logger.warning(f"Remat H2H failed for match {external_id}: {e}")
            try:
                await self.session.rollback()
            except Exception:
                pass

        xg = None
        try:
            xg = await self.materializer.compute_xg_last5_features(
                home_team_id=match["home_team_id"],
                away_team_id=match["away_team_id"],
                kickoff_utc=kickoff_utc,
                league_id=match.get("competition_id"),  # m.league_id aliased as competition_id
            )
        except Exception as e:
            logger.warning(f"Remat xG failed for match {external_id}: {e}")
            try:
                await self.session.rollback()
            except Exception:
                pass

        lineup = None
        try:
            lineup = await self.materializer.compute_lineup_features(
                match_id=match["id"],
                kickoff_utc=kickoff_utc,
            )
        except Exception as e:
            logger.warning(f"Remat lineup failed for match {external_id}: {e}")
            try:
                await self.session.rollback()
            except Exception:
                pass

        xi_depth = None
        try:
            xi_depth = await self.materializer.compute_xi_depth_features(
                match_id=match["id"],
                kickoff_utc=kickoff_utc,
                home_formation=lineup.sofascore_home_formation if lineup else None,
                away_formation=lineup.sofascore_away_formation if lineup else None,
            )
        except Exception as e:
            logger.warning(f"Remat XI depth failed for match {external_id}: {e}")
            try:
                await self.session.rollback()
            except Exception:
                pass

        # 3. Materialize (upsert)
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
            result["rematerialized"] = inserted
            result["with_xg"] = xg is not None
            result["with_lineup"] = lineup is not None
            result["with_xi_depth"] = xi_depth is not None
            logger.info(
                f"Match {external_id}: re-materialized "
                f"(odds_newer={result['odds_history_newer']}, xg={xg is not None}, "
                f"lineup={lineup is not None}, xi={xi_depth is not None})"
            )
        except PITViolationError as e:
            logger.error(f"Remat PIT violation for match {external_id}: {e}")
        except IntegrityError as e:
            err_msg = str(e.orig) if hasattr(e, "orig") else str(e)
            if "pit_valid" in err_msg:
                logger.error(
                    f"Remat DB pit_valid check failed for match {external_id} "
                    f"(likely kickoff reschedule): {err_msg}"
                )
                try:
                    await self.session.rollback()
                except Exception:
                    pass
            else:
                raise

        return result

    async def _read_current_fm_odds(
        self,
        external_id: int,
    ) -> Optional["OddsFeatures"]:
        """Read current odds from feature_matrix to use as passthrough during re-materialization."""
        from app.titan.materializers.feature_matrix import OddsFeatures
        query = text(f"""
            SELECT odds_home_close, odds_draw_close, odds_away_close, odds_captured_at
            FROM {self.schema}.feature_matrix
            WHERE match_id = :match_id
              AND odds_captured_at IS NOT NULL
        """)
        result = await self.session.execute(query, {"match_id": external_id})
        row = result.fetchone()
        if not row or row[0] is None:
            return None

        captured_at = row[3]
        if captured_at and captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)

        return self.materializer.build_odds_features(
            odds_home=float(row[0]),
            odds_draw=float(row[1]),
            odds_away=float(row[2]),
            captured_at=captured_at,
        )

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
    print(f"  Re-materialized:    {stats['rematerialized']}")
    print(f"  Odds history newer: {stats['odds_history_newer']}")
    print(f"  Skipped PIT refresh:{stats['skipped_pit_refresh']}")
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
