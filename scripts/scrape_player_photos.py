"""
Scrape Player Photos Pipeline v2.

Fetches HQ player photos from club websites, ALWAYS processes through PhotoRoom
(bg removal + cleanup), then stores original clean + face crop.

Sources: Club Site (P1, HQ official), API-Football (P5 fallback).
Pipeline: fetch -> validate -> identity match -> PhotoRoom (always) -> face crop -> upload -> DB insert.

Policy: PhotoRoom ALWAYS runs — even if image already has transparent bg, it improves quality.

Usage:
  source .env && python3 scripts/scrape_player_photos.py --mode pilot --league 239 --dry-run
  source .env && python3 scripts/scrape_player_photos.py --mode full --league 239
  source .env && python3 scripts/scrape_player_photos.py --mode full --league 239 --source club_site
  source .env && python3 scripts/scrape_player_photos.py --mode pilot --team 1137 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scrape_player_photos")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Imports (after sys.path)
# ---------------------------------------------------------------------------
from app.photos.config import get_photos_settings, build_player_photo_key, compute_content_hash
from app.photos.validator import validate_player_photo
from app.photos.matcher import CandidateSignals, PlayerDB, score_identity
from app.photos.processor import crop_face

photos_settings = get_photos_settings()

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
PLAYERS_SQL = """
    SELECT p.external_id, p.name, p.firstname, p.lastname,
           p.position, p.jersey_number, p.team_id, p.team_external_id,
           t.name as team_name, p.photo_url,
           pim.sofascore_id
    FROM players p
    JOIN teams t ON t.id = p.team_id
    LEFT JOIN player_id_mapping pim
        ON pim.api_football_id = p.external_id AND pim.status = 'active'
    WHERE t.id IN (
        SELECT DISTINCT home_team_id FROM matches WHERE league_id = $1
        UNION
        SELECT DISTINCT away_team_id FROM matches WHERE league_id = $1
    )
    ORDER BY t.name, p.name
"""

# Check if asset already exists (dedup by content_hash)
EXISTING_HASH_SQL = """
    SELECT id FROM player_photo_assets
    WHERE player_external_id = $1 AND content_hash = $2
    LIMIT 1
"""

# Deactivate previous active assets for this slot
DEACTIVATE_SQL = """
    UPDATE player_photo_assets
    SET is_active = false, deactivated_at = NOW(), changed_by = 'pipeline'
    WHERE player_external_id = $1
      AND COALESCE(context_team_id, 0) = $2
      AND asset_type = $3
      AND style = $4
      AND is_active = true
"""

INSERT_SQL = """
    INSERT INTO player_photo_assets (
        player_external_id, context_team_id, season, role, kit_variant,
        asset_type, style, r2_key, cdn_url, content_hash, revision,
        source, processor, quality_score, photo_meta,
        review_status, is_active, activated_at, changed_by, run_id,
        created_at, updated_at
    ) VALUES (
        $1, $2, $3, $4, $5,
        $6, $7, $8, $9, $10, $11,
        $12, $13, $14, $15::jsonb,
        $16, $17, $18, $19, $20,
        NOW(), NOW()
    )
"""

# Insert candidate for review (no R2 upload, no PhotoRoom)
INSERT_CANDIDATE_SQL = """
    INSERT INTO player_photo_assets (
        player_external_id, context_team_id, season, role, kit_variant,
        asset_type, style, r2_key, cdn_url, content_hash, revision,
        source, processor, quality_score, photo_meta,
        review_status, is_active, activated_at, changed_by, run_id,
        created_at, updated_at
    ) VALUES (
        $1, NULL, NULL, NULL, NULL,
        'candidate', 'raw', '', '', $2, 0,
        $3, 'none', $4, $5::jsonb,
        'pending_review', false, NULL, 'pipeline', $6,
        NOW(), NOW()
    )
"""

# Delete old candidates from previous runs for same player
DELETE_OLD_CANDIDATES_SQL = """
    DELETE FROM player_photo_assets
    WHERE player_external_id = $1
      AND asset_type = 'candidate'
      AND review_status = 'pending_review'
"""


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------
@dataclass
class PipelineStats:
    total_players: int = 0
    candidates_fetched: int = 0
    validation_passed: int = 0
    validation_failed: int = 0
    identity_passed: int = 0
    identity_failed: int = 0
    vision_passed: int = 0
    vision_skipped: int = 0
    vision_failed: int = 0
    bg_removed: int = 0
    bg_skipped: int = 0
    uploaded: int = 0
    skipped_existing: int = 0
    errors: int = 0
    sources: dict = field(default_factory=lambda: defaultdict(int))

    def summary(self) -> str:
        lines = [
            f"Total players:       {self.total_players}",
            f"Candidates fetched:  {self.candidates_fetched}",
            f"Validation passed:   {self.validation_passed} / failed: {self.validation_failed}",
            f"Identity passed:     {self.identity_passed} / failed: {self.identity_failed}",
            f"Vision passed:       {self.vision_passed} / skipped: {self.vision_skipped} / failed: {self.vision_failed}",
            f"BG removed:          {self.bg_removed} / skipped: {self.bg_skipped}",
            f"Uploaded:            {self.uploaded}",
            f"Skipped (existing):  {self.skipped_existing}",
            f"Errors:              {self.errors}",
            f"Sources:             {dict(self.sources)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
async def fetch_candidate(player: dict, source: str, club_site_map: dict = None) -> dict:
    """Fetch candidate photo from specified source.

    Returns dict with image_bytes, source, quality_cap, error, candidate_name, candidate_jersey, candidate_position.
    """
    ext_id = player["external_id"]
    team_ext_id = player.get("team_external_id")

    if source == "club_site":
        if not club_site_map:
            return {"image_bytes": None, "source": source, "quality_cap": 0, "error": "No club_site_map"}

        team_players = club_site_map.get(team_ext_id, [])
        if not team_players:
            return {"image_bytes": None, "source": source, "quality_cap": 0, "error": "No club config for team"}

        # Find best name match from club site players
        from app.photos.matcher import normalize_name, fuzzy_name_score
        best_match = None
        best_ratio = 0.0
        for cp in team_players:
            # Use fuzzy_name_score which handles lastname-only, first+last combos
            ratio = fuzzy_name_score(
                cp.name,
                player["name"],
                player.get("firstname"),
                player.get("lastname"),
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cp
            if ratio >= 0.95:
                break  # Exact match

        if not best_match or best_ratio < 0.55:
            return {"image_bytes": None, "source": source, "quality_cap": 0,
                    "error": f"No name match (best={best_ratio:.2f})"}

        # Fetch the photo
        from app.photos.scrapers.club_site import fetch_club_photo
        result = await fetch_club_photo(best_match.image_url, quality_cap=90)
        return {
            "image_bytes": result.image_bytes,
            "source": "club_site",
            "quality_cap": result.quality_cap,
            "error": result.error,
            "candidate_name": best_match.name,
            "candidate_jersey": best_match.jersey_number,
            "candidate_position": best_match.position,
            "candidate_url": best_match.image_url,
            "match_ratio": best_ratio,
        }

    elif source == "api_football":
        from app.photos.scrapers.api_football import fetch_apifb_photo
        result = await fetch_apifb_photo(ext_id)
        return {
            "image_bytes": result.image_bytes,
            "source": result.source,
            "quality_cap": result.quality_cap,
            "error": result.error,
            "candidate_url": f"https://media.api-sports.io/football/players/{ext_id}.png",
        }
    else:
        return {"image_bytes": None, "source": source, "quality_cap": 0, "error": f"No {source} ID available"}


async def process_player(
    player: dict,
    sources: list,
    run_id: str,
    conn,
    r2_client,
    cdn_base: str,
    stats: PipelineStats,
    dry_run: bool = False,
    use_vision: bool = True,
    skip_photoroom: bool = False,
    club_site_map: dict = None,
    save_candidates: bool = False,
):
    """Full pipeline for one player: fetch -> validate -> upload."""
    ext_id = player["external_id"]
    player_name = player["name"]

    # Build PlayerDB for identity matching
    player_db = PlayerDB(
        external_id=ext_id,
        name=player_name,
        firstname=player.get("firstname"),
        lastname=player.get("lastname"),
        jersey_number=player.get("jersey_number"),
        position=player.get("position"),
        team_external_id=player.get("team_external_id"),
    )

    # Try each source in priority order
    for source in sources:
        candidate = await fetch_candidate(player, source, club_site_map=club_site_map)

        if candidate["error"] or not candidate["image_bytes"]:
            logger.debug(f"  [{source}] {player_name}: {candidate.get('error', 'no image')}")
            continue

        stats.candidates_fetched += 1
        stats.sources[source] += 1
        image_bytes = candidate["image_bytes"]

        # Step 1: Validate photo
        vr = validate_player_photo(image_bytes)
        if not vr.valid:
            stats.validation_failed += 1
            logger.debug(f"  [{source}] {player_name}: QA failed — {vr.errors}")
            continue
        stats.validation_passed += 1

        # Step 2: Identity match
        # For club_site: use scraped name/jersey/position as signals
        # For API-Football: trust the ID mapping (name+team context)
        candidate_signals = CandidateSignals(
            name=candidate.get("candidate_name") or player_name,
            jersey_number=candidate.get("candidate_jersey"),
            position=candidate.get("candidate_position"),
            team_external_id=player.get("team_external_id"),
        )
        identity = score_identity(candidate_signals, player_db)
        if not identity.passes_threshold:
            stats.identity_failed += 1
            logger.debug(f"  [{source}] {player_name}: identity score {identity.score} < threshold")
            continue
        stats.identity_passed += 1

        # Save candidate mode: store URL + metadata in DB, skip PhotoRoom/R2
        if save_candidates:
            quality = min(identity.score, candidate["quality_cap"])
            current_api_url = f"https://media.api-sports.io/football/players/{ext_id}.png"
            candidate_url = candidate.get("candidate_url") or current_api_url
            meta = json.dumps({
                "candidate_url": candidate_url,
                "current_url": current_api_url,
                "width": vr.width,
                "height": vr.height,
                "format": vr.format,
                "size_bytes": len(image_bytes),
                "identity_score": identity.score,
                "identity_details": identity.details,
                "candidate_name": candidate.get("candidate_name"),
                "player_name": player_name,
                "team_name": player.get("team_name"),
                "team_external_id": player.get("team_external_id"),
            })
            content_hash = compute_content_hash(image_bytes)
            if conn:
                await conn.execute(DELETE_OLD_CANDIDATES_SQL, ext_id)
                await conn.execute(
                    INSERT_CANDIDATE_SQL,
                    ext_id, content_hash, source, quality, meta, run_id,
                )
            stats.uploaded += 1
            logger.info(
                f"  [{source}] {player_name}: CANDIDATE saved — "
                f"quality={quality}, size={vr.width}x{vr.height}"
            )
            return

        # Step 3: Vision validation (optional)
        if use_vision and photos_settings.PHOTOS_GEMINI_VISION_ENABLED:
            from app.photos.vision import validate_with_vision
            vision = await validate_with_vision(image_bytes, player_name)
            if vision.error:
                stats.vision_skipped += 1
                logger.debug(f"  [{source}] {player_name}: vision error — {vision.error}")
                # Don't block on vision errors, continue
            elif not vision.passes:
                stats.vision_failed += 1
                logger.info(f"  [{source}] {player_name}: vision rejected — {vision.reasoning}")
                continue
            else:
                stats.vision_passed += 1
        else:
            stats.vision_skipped += 1

        # Step 4: PhotoRoom (ALWAYS — improves even already-transparent images)
        processed_bytes = image_bytes
        processor = "none"
        if photos_settings.PHOTOROOM_API_KEY and not skip_photoroom:
            from app.photos.photoroom import remove_background
            bg_result = await remove_background(image_bytes)
            if bg_result:
                processed_bytes = bg_result
                processor = "photoroom"
                stats.bg_removed += 1
            else:
                logger.warning(f"  [{source}] {player_name}: PhotoRoom failed, using original")
                stats.bg_skipped += 1
        else:
            stats.bg_skipped += 1

        # Step 5: Re-save as optimized PNG RGBA with attribution metadata
        import io
        from PIL import Image as PILImage
        from PIL.PngImagePlugin import PngInfo
        img = PILImage.open(io.BytesIO(processed_bytes))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        png_meta = PngInfo()
        png_meta.add_text("Author", "BON JOGO")
        png_meta.add_text("Copyright", "BON JOGO - bonjogo.com")
        png_meta.add_text("Source", "https://bonjogo.com")
        png_meta.add_text("Comment", f"{player_name} | id:{ext_id} | bonjogo.com")
        original_buf = io.BytesIO()
        img.save(original_buf, format="PNG", optimize=True, pnginfo=png_meta)
        original_bytes = original_buf.getvalue()

        # Step 6: Compute content hash
        content_hash = compute_content_hash(original_bytes)

        # Step 7: Check for existing asset with same hash (dedup)
        if conn:
            existing = await conn.fetchval(EXISTING_HASH_SQL, ext_id, content_hash)
            if existing:
                stats.skipped_existing += 1
                logger.debug(f"  [{source}] {player_name}: already exists (hash={content_hash[:12]})")
                return

        # Step 8: Face crop (512px square)
        face_bytes = crop_face(original_bytes, output_size=512, player_name=player_name, player_ext_id=ext_id)

        # Determine quality score and review status
        quality = min(identity.score, candidate["quality_cap"])
        review_status = "approved" if quality >= 80 else "pending_review"
        is_active = review_status == "approved"

        # Photo metadata
        photo_meta = {
            "width": vr.width,
            "height": vr.height,
            "format": vr.format,
            "original_size_bytes": len(image_bytes),
            "processed_size_bytes": len(original_bytes),
            "face_crop_size_bytes": len(face_bytes) if face_bytes else 0,
            "identity_score": identity.score,
            "identity_details": identity.details,
            "processor": processor,
        }

        if dry_run:
            stats.uploaded += 1
            logger.info(
                f"  [{source}] {player_name}: DRY-RUN OK — quality={quality}, "
                f"review={review_status}, hash={content_hash[:12]}, "
                f"size={vr.width}x{vr.height}, face={'yes' if face_bytes else 'no'}"
            )
            return

        # Step 9: Upload to R2
        style = "segmented" if processor == "photoroom" else "raw"

        # Upload original (full body clean)
        original_key = build_player_photo_key(ext_id, content_hash, "original", style, "png")
        original_cdn = f"{cdn_base}/{original_key}" if cdn_base else original_key

        # Upload face crop (if available)
        face_key = None
        face_cdn = None
        if face_bytes:
            face_key = build_player_photo_key(ext_id, content_hash, "face", style, "png")
            face_cdn = f"{cdn_base}/{face_key}" if cdn_base else face_key

        if r2_client:
            original_ok = await r2_client.put_object(original_key, original_bytes, "image/png")
            if not original_ok:
                stats.errors += 1
                logger.error(f"  [{source}] {player_name}: R2 upload failed (original)")
                continue

            if face_bytes and face_key:
                face_ok = await r2_client.put_object(face_key, face_bytes, "image/png")
                if not face_ok:
                    logger.warning(f"  [{source}] {player_name}: R2 upload failed (face), original OK")

        # Step 10: DB insert (deactivate old -> insert new)
        if conn:
            context_team_coalesce = 0  # NULL -> 0 for COALESCE in deactivate
            meta_json = json.dumps(photo_meta)

            # Deactivate old original
            await conn.execute(DEACTIVATE_SQL, ext_id, context_team_coalesce, "original", style)
            # Insert original
            await conn.execute(
                INSERT_SQL,
                ext_id, None, None, None, None,
                "original", style, original_key, original_cdn, content_hash, 1,
                source, processor, quality, meta_json,
                review_status, is_active, datetime.utcnow() if is_active else None,
                "pipeline", run_id,
            )

            # Face crop (if available)
            if face_bytes and face_key:
                await conn.execute(DEACTIVATE_SQL, ext_id, context_team_coalesce, "face", style)
                await conn.execute(
                    INSERT_SQL,
                    ext_id, None, None, None, None,
                    "face", style, face_key, face_cdn, content_hash, 1,
                    source, processor, quality, meta_json,
                    review_status, is_active, datetime.utcnow() if is_active else None,
                    "pipeline", run_id,
                )

        stats.uploaded += 1
        logger.info(
            f"  [{source}] {player_name}: OK — quality={quality}, "
            f"review={review_status}, hash={content_hash[:12]}, face={'yes' if face_bytes else 'no'}"
        )
        return  # Success, no need to try next source

    # All sources exhausted
    logger.debug(f"  {player_name}: no photo found from any source")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Scrape player photos pipeline")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot = 5 players/team, full = all players")
    parser.add_argument("--league", type=int, default=239,
                        help="League ID (default: 239 Colombia)")
    parser.add_argument("--team", type=int, default=None,
                        help="Single team external_id (overrides --league)")
    parser.add_argument("--source", choices=["club_site", "api_football", "all"], default="all",
                        help="Source to scrape (default: all = club_site -> api_football)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, no upload/DB write")
    parser.add_argument("--no-vision", action="store_true",
                        help="Skip Gemini Vision validation")
    parser.add_argument("--skip-photoroom", action="store_true",
                        help="Skip PhotoRoom (default: ALWAYS run PhotoRoom)")
    parser.add_argument("--save-candidates", action="store_true",
                        help="Save candidates for review (no PhotoRoom, no R2 upload)")
    parser.add_argument("--approved-only", action="store_true",
                        help="Process only previously approved candidates through PhotoRoom + R2")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between PhotoRoom requests (seconds, respects rate limit)")
    args = parser.parse_args()

    run_id = str(uuid.uuid4())[:8]
    logger.info(f"=== Player Photos Pipeline v2 (run_id={run_id}) ===")
    logger.info(f"Mode: {args.mode}, League: {args.league}, Source: {args.source}")
    if args.save_candidates:
        logger.info(f"CANDIDATE MODE: saving for review (no PhotoRoom, no R2)")
    elif args.approved_only:
        logger.info(f"APPROVED-ONLY MODE: processing approved candidates")
    else:
        logger.info(f"Dry-run: {args.dry_run}, Vision: {not args.no_vision}, PhotoRoom: {'SKIP' if args.skip_photoroom else 'ALWAYS'}")

    # Source priority
    if args.source == "all":
        sources = ["club_site", "api_football"]
    else:
        sources = [args.source]

    # DB connection
    import asyncpg
    db_url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    conn = await asyncpg.connect(db_url)
    logger.info("DB connected")

    # R2 client (reuse logos)
    r2_client = None
    cdn_base = ""
    if not args.dry_run and not args.save_candidates:
        from app.logos.r2_client import get_logos_r2_client
        from app.logos.config import get_logos_settings
        r2_client = get_logos_r2_client()
        logos_s = get_logos_settings()
        cdn_base = logos_s.LOGOS_CDN_BASE_URL.rstrip("/") if logos_s.LOGOS_CDN_BASE_URL else ""
        if r2_client:
            logger.info(f"R2 client: OK (cdn={cdn_base or 'not set'})")
        else:
            logger.warning("R2 client: DISABLED (LOGOS_R2_ENABLED=false). DB writes only, no file upload.")

    # ===========================================================================
    # APPROVED-ONLY MODE: process previously approved candidates
    # ===========================================================================
    if args.approved_only:
        APPROVED_SQL = """
            SELECT id, player_external_id, source, quality_score, photo_meta, content_hash
            FROM player_photo_assets
            WHERE asset_type = 'candidate' AND review_status = 'approved'
            ORDER BY id
        """
        approved_rows = await conn.fetch(APPROVED_SQL)
        logger.info(f"Found {len(approved_rows)} approved candidates to process")

        from app.logos.r2_client import get_logos_r2_client
        from app.logos.config import get_logos_settings
        r2_client = get_logos_r2_client()
        logos_s = get_logos_settings()
        cdn_base = logos_s.LOGOS_CDN_BASE_URL.rstrip("/") if logos_s.LOGOS_CDN_BASE_URL else ""
        logger.info(f"R2 client: OK (cdn={cdn_base})")

        stats = PipelineStats(total_players=len(approved_rows))
        t0 = time.time()

        for i, row in enumerate(approved_rows, 1):
            if i % 20 == 0:
                logger.info(f"Progress: {i}/{len(approved_rows)} ({stats.uploaded} uploaded)")

            ext_id = row["player_external_id"]
            meta = row["photo_meta"] or {}
            candidate_url = meta.get("candidate_url", "")
            player_name = meta.get("player_name", f"player_{ext_id}")
            source = row["source"]
            quality = row["quality_score"] or 40

            if not candidate_url:
                logger.warning(f"  {player_name}: no candidate_url in meta, skipping")
                stats.errors += 1
                continue

            try:
                # Download candidate image
                import httpx
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(candidate_url)
                    if resp.status_code != 200:
                        logger.warning(f"  {player_name}: download failed ({resp.status_code})")
                        stats.errors += 1
                        continue
                    image_bytes = resp.content

                # PhotoRoom
                processed_bytes = image_bytes
                processor = "none"
                if photos_settings.PHOTOROOM_API_KEY:
                    from app.photos.photoroom import remove_background
                    bg_result = await remove_background(image_bytes)
                    if bg_result:
                        processed_bytes = bg_result
                        processor = "photoroom"
                        stats.bg_removed += 1
                    else:
                        stats.bg_skipped += 1

                # Re-save as PNG RGBA with metadata
                import io
                from PIL import Image as PILImage
                from PIL.PngImagePlugin import PngInfo
                img = PILImage.open(io.BytesIO(processed_bytes))
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                png_meta = PngInfo()
                png_meta.add_text("Author", "BON JOGO")
                png_meta.add_text("Copyright", "BON JOGO - bonjogo.com")
                png_meta.add_text("Source", "https://bonjogo.com")
                png_meta.add_text("Comment", f"{player_name} | id:{ext_id} | bonjogo.com")
                original_buf = io.BytesIO()
                img.save(original_buf, format="PNG", optimize=True, pnginfo=png_meta)
                original_bytes = original_buf.getvalue()

                content_hash = compute_content_hash(original_bytes)

                # Face crop
                face_bytes = crop_face(original_bytes, output_size=512, player_name=player_name, player_ext_id=ext_id)

                style = "segmented" if processor == "photoroom" else "raw"
                review_status = "approved"
                is_active = True

                photo_meta = json.dumps({
                    "width": img.size[0], "height": img.size[1],
                    "processor": processor, "source_url": candidate_url,
                    "player_name": player_name,
                    "team_name": meta.get("team_name"),
                })

                # Upload to R2
                original_key = build_player_photo_key(ext_id, content_hash, "original", style, "png")
                original_cdn = f"{cdn_base}/{original_key}"
                await r2_client.put_object(original_key, original_bytes, "image/png")

                face_key = None
                face_cdn = None
                if face_bytes:
                    face_key = build_player_photo_key(ext_id, content_hash, "face", style, "png")
                    face_cdn = f"{cdn_base}/{face_key}"
                    await r2_client.put_object(face_key, face_bytes, "image/png")

                # DB: deactivate old, insert new original + face
                await conn.execute(DEACTIVATE_SQL, ext_id, 0, "original", style)
                await conn.execute(
                    INSERT_SQL,
                    ext_id, None, None, None, None,
                    "original", style, original_key, original_cdn, content_hash, 1,
                    source, processor, quality, photo_meta,
                    review_status, is_active, datetime.utcnow(),
                    "pipeline", run_id,
                )

                if face_bytes and face_key:
                    await conn.execute(DEACTIVATE_SQL, ext_id, 0, "face", style)
                    await conn.execute(
                        INSERT_SQL,
                        ext_id, None, None, None, None,
                        "face", style, face_key, face_cdn, content_hash, 1,
                        source, processor, quality, photo_meta,
                        review_status, is_active, datetime.utcnow(),
                        "pipeline", run_id,
                    )

                # Mark candidate as processed (superseded)
                await conn.execute(
                    "UPDATE player_photo_assets SET review_status = 'superseded', updated_at = NOW() WHERE id = $1",
                    row["id"],
                )

                stats.uploaded += 1
                stats.sources[source] += 1
                logger.info(f"  [{source}] {player_name}: OK (photoroom={processor}, face={'yes' if face_bytes else 'no'})")

            except Exception as e:
                stats.errors += 1
                logger.error(f"  {player_name}: error — {e}")

            await asyncio.sleep(args.delay)

        elapsed = time.time() - t0
        await conn.close()
        if r2_client:
            await r2_client.close()

        logger.info(f"\n{'='*60}")
        logger.info(f"Approved-only complete in {elapsed:.1f}s")
        logger.info(f"{'='*60}")
        logger.info(f"\n{stats.summary()}")
        return

    # ===========================================================================
    # STANDARD MODE: scrape + process
    # ===========================================================================

    # Fetch players
    if args.team:
        # Single team mode
        TEAM_PLAYERS_SQL = """
            SELECT p.external_id, p.name, p.firstname, p.lastname,
                   p.position, p.jersey_number, p.team_id, p.team_external_id,
                   p.photo_url,
                   pim.sofascore_id
            FROM players p
            JOIN teams t ON t.id = p.team_id
            LEFT JOIN player_id_mapping pim
                ON pim.api_football_id = p.external_id AND pim.status = 'active'
            WHERE t.external_id = $1
            ORDER BY p.name
        """
        rows = await conn.fetch(TEAM_PLAYERS_SQL, args.team)
        players = [dict(r) for r in rows]
        logger.info(f"Found {len(players)} players for team {args.team}")
    else:
        rows = await conn.fetch(PLAYERS_SQL, args.league)
        players = [dict(r) for r in rows]
        logger.info(f"Found {len(players)} players in league {args.league}")

    # Pilot mode: 5 per team
    if args.mode == "pilot":
        by_team = defaultdict(list)
        for p in players:
            by_team[p["team_external_id"]].append(p)
        players = []
        for team_ext, team_players in by_team.items():
            players.extend(team_players[:5])
        logger.info(f"Pilot mode: {len(players)} players ({len(by_team)} teams x 5)")

    stats = PipelineStats(total_players=len(players))

    # Pre-scrape club sites (once per team, reuse for all players)
    club_site_map = {}  # team_external_id -> list of ClubSitePlayer
    if "club_site" in sources:
        from app.photos.scrapers.club_site import scrape_club_squad, load_club_config
        club_config = load_club_config()
        # Get unique team_external_ids that have club config
        team_ext_ids = set(p["team_external_id"] for p in players if p.get("team_external_id"))
        teams_with_config = [t for t in team_ext_ids if str(t) in club_config]
        logger.info(f"Club site pre-scrape: {len(teams_with_config)}/{len(team_ext_ids)} teams have config")

        for team_ext_id in sorted(teams_with_config):
            team_cfg = club_config[str(team_ext_id)]
            result = await scrape_club_squad(team_ext_id, team_cfg.get("team_name"), config=club_config)
            if result.error:
                logger.warning(f"  Club scrape {team_cfg.get('team_name', team_ext_id)}: {result.error}")
            elif result.players:
                club_site_map[team_ext_id] = result.players
                logger.info(f"  Club scrape {result.team_name}: {len(result.players)} players found")
            else:
                logger.warning(f"  Club scrape {result.team_name}: 0 players extracted")
            await asyncio.sleep(1.0)  # Be respectful to club sites

        logger.info(f"Club site map: {sum(len(v) for v in club_site_map.values())} total scraped players across {len(club_site_map)} teams")

    # Process
    t0 = time.time()
    for i, player in enumerate(players, 1):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(players)} ({stats.uploaded} uploaded)")

        try:
            await process_player(
                player=player,
                sources=sources,
                run_id=run_id,
                conn=conn if (not args.dry_run or args.save_candidates) else None,
                r2_client=r2_client,
                cdn_base=cdn_base,
                stats=stats,
                dry_run=args.dry_run,
                use_vision=not args.no_vision,
                skip_photoroom=args.skip_photoroom,
                club_site_map=club_site_map,
                save_candidates=args.save_candidates,
            )
        except Exception as e:
            stats.errors += 1
            logger.error(f"  {player['name']}: pipeline error — {e}")

        # Rate limit
        await asyncio.sleep(args.delay)

    elapsed = time.time() - t0

    # Close
    await conn.close()
    if r2_client:
        await r2_client.close()

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"{'='*60}")
    logger.info(f"\n{stats.summary()}")

    # JSON report for auditor
    report = {
        "run_id": run_id,
        "mode": args.mode,
        "league": args.league,
        "source": args.source,
        "dry_run": args.dry_run,
        "elapsed_seconds": round(elapsed, 1),
        "stats": {
            "total_players": stats.total_players,
            "candidates_fetched": stats.candidates_fetched,
            "validation_passed": stats.validation_passed,
            "validation_failed": stats.validation_failed,
            "identity_passed": stats.identity_passed,
            "identity_failed": stats.identity_failed,
            "vision_passed": stats.vision_passed,
            "vision_skipped": stats.vision_skipped,
            "vision_failed": stats.vision_failed,
            "bg_removed": stats.bg_removed,
            "uploaded": stats.uploaded,
            "skipped_existing": stats.skipped_existing,
            "errors": stats.errors,
            "sources": dict(stats.sources),
        },
    }
    print(f"\n{json.dumps(report, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
