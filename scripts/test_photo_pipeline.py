#!/usr/bin/env python3
"""Test photo pipeline for a single player.

Demonstrates the full flow:
1. Scrape photo from club website
2. Validate quality (QA gate)
3. Detect transparency (skip PhotoRoom if already clean)
4. PhotoRoom bg removal (if needed, uses trial credits)
5. Save original clean + face crop locally

Usage:
    source .env
    python scripts/test_photo_pipeline.py --player "match" --team 1138
    python scripts/test_photo_pipeline.py --player "mena" --team 1128
    python scripts/test_photo_pipeline.py --player "match" --team 1138 --skip-photoroom
"""

import argparse
import asyncio
import io
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.photos.scrapers.club_site import scrape_club_squad, fetch_club_photo
from app.photos.validator import validate_player_photo
from app.photos.processor import has_transparent_background, crop_face
from app.photos.photoroom import remove_background

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "photo_test")


async def main():
    parser = argparse.ArgumentParser(description="Test photo pipeline for a single player")
    parser.add_argument("--player", required=True, help="Player name search term (e.g. 'match', 'mena')")
    parser.add_argument("--team", required=True, type=int, help="Team external ID (e.g. 1138)")
    parser.add_argument("--skip-photoroom", action="store_true", help="Skip PhotoRoom API call")
    parser.add_argument("--force-photoroom", action="store_true", help="Force PhotoRoom even if image already has transparent bg")
    args = parser.parse_args()

    player_search = args.player.lower()
    team_ext_id = args.team
    skip_photoroom = args.skip_photoroom
    force_photoroom = args.force_photoroom

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Target: '{player_search}' in team {team_ext_id}")

    # Step 1: Scrape club squad page
    logger.info(f"--- Step 1: Scraping squad page for team {team_ext_id} ---")
    result = await scrape_club_squad(team_ext_id)

    if result.error:
        logger.error(f"Scrape failed: {result.error}")
        return

    logger.info(f"Found {len(result.players)} players on squad page")

    # Find player by search term
    match = None
    for p in result.players:
        if player_search in p.name.lower():
            match = p
            break

    if not match:
        logger.error(f"Could not find '{player_search}' in squad page. Players found:")
        for p in result.players:
            logger.info(f"  - {p.name} ({p.image_url[:80]}...)")
        return

    logger.info(f"Found: {match.name} | image: {match.image_url}")

    # Step 2: Download the photo
    logger.info("--- Step 2: Downloading photo ---")
    photo = await fetch_club_photo(match.image_url, quality_cap=result.quality_cap)

    if photo.error or not photo.image_bytes:
        logger.error(f"Download failed: {photo.error}")
        return

    logger.info(f"Downloaded: {len(photo.image_bytes)} bytes")

    # Save raw source
    raw_path = os.path.join(OUTPUT_DIR, "01_raw_source.png")
    with open(raw_path, "wb") as f:
        f.write(photo.image_bytes)
    logger.info(f"Saved raw source -> {raw_path}")

    # Step 3: QA validation
    logger.info("--- Step 3: QA Validation ---")
    val = validate_player_photo(photo.image_bytes)
    logger.info(f"Validation: {val}")

    if not val.valid:
        logger.error(f"QA FAILED: {val.errors}")
        return

    # Step 4: Detect transparency
    logger.info("--- Step 4: Transparency Detection ---")
    is_transparent = has_transparent_background(photo.image_bytes)
    logger.info(f"Has transparent background: {is_transparent}")

    # Step 5: PhotoRoom bg removal (ALWAYS — improves even transparent images)
    clean_bytes = photo.image_bytes

    if skip_photoroom:
        logger.info("PhotoRoom skipped (--skip-photoroom flag)")
    else:
        logger.info(f"--- Step 5: PhotoRoom Background Removal (ALWAYS policy) ---")
        api_key = os.environ.get("PHOTOROOM_API_KEY", "")
        if not api_key:
            logger.warning("PHOTOROOM_API_KEY not set, skipping bg removal")
            logger.info("Set PHOTOROOM_API_KEY in .env or run with --skip-photoroom")
        else:
            result_bytes = await remove_background(photo.image_bytes)
            if result_bytes:
                clean_bytes = result_bytes
                logger.info(f"PhotoRoom result: {len(result_bytes)} bytes")

                # Save PhotoRoom version for comparison
                pr_path = os.path.join(OUTPUT_DIR, "02b_photoroom.png")
                with open(pr_path, "wb") as f:
                    f.write(result_bytes)
                logger.info(f"Saved PhotoRoom result -> {pr_path}")
            else:
                logger.warning("PhotoRoom failed, using original image")

    # Save original clean (after bg removal if applied)
    original_path = os.path.join(OUTPUT_DIR, "02_original_clean.png")
    # Re-save as PNG with RGBA
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(clean_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.save(original_path, format="PNG", optimize=True)
    logger.info(f"Saved original clean ({img.size[0]}x{img.size[1]}) -> {original_path}")

    # Step 6: Face crop
    logger.info("--- Step 6: Face Crop ---")
    face_bytes = crop_face(clean_bytes, output_size=512, player_name=match.name, player_ext_id=0)

    if face_bytes:
        face_path = os.path.join(OUTPUT_DIR, "03_face_crop_512.png")
        with open(face_path, "wb") as f:
            f.write(face_bytes)
        face_img = PILImage.open(io.BytesIO(face_bytes))
        logger.info(f"Saved face crop ({face_img.size[0]}x{face_img.size[1]}) -> {face_path}")
    else:
        logger.warning("Face crop failed")

    # Summary
    logger.info("--- Summary ---")
    logger.info(f"Player: {match.name}")
    logger.info(f"Source: {match.image_url}")
    logger.info(f"QA: {'PASS' if val.valid else 'FAIL'} ({val.width}x{val.height}, {val.format})")
    logger.info(f"Transparency: {'already clean' if is_transparent else 'needs PhotoRoom'}")
    logger.info(f"PhotoRoom: {'skipped' if skip_photoroom else 'applied'}")
    logger.info(f"Files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
