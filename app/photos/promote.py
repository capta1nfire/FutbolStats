"""Promote approved photo candidate to active player card.

Pipeline: crop → PhotoRoom bg removal → R2 upload → activate in DB.
Called from the review endpoint when action == 'approve'.
"""

import io
import logging

import httpx
from PIL import Image as PILImage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.logos.r2_client import get_logos_r2_client
from app.photos.config import build_player_photo_key, compute_content_hash
from app.photos.photoroom import remove_background
from app.photos.processor import crop_face

logger = logging.getLogger(__name__)


async def promote_candidate(candidate_id: int, session: AsyncSession) -> dict:
    """Promote an approved candidate to an active player card.

    Steps:
        1. Read candidate metadata from DB
        2. Download source image
        3. Apply manual crop (or auto crop_face)
        4. Run PhotoRoom background removal
        5. Upload to R2
        6. Deactivate previous active cards for this player
        7. Insert new card row with is_active=true

    Returns:
        {"cdn_url": str, "r2_key": str, "card_id": int} on success
        {"error": str} on failure
    """
    # 1. Read candidate
    sql = text("""
        SELECT id, player_external_id, context_team_id, season, role, kit_variant,
               source, photo_meta
        FROM player_photo_assets
        WHERE id = :id AND asset_type = 'candidate' AND review_status = 'approved'
    """)
    result = await session.execute(sql, {"id": candidate_id})
    row = result.mappings().first()
    if not row:
        return {"error": "Candidate not found or not approved"}

    meta = row["photo_meta"] or {}
    candidate_url = meta.get("candidate_url", "")
    if not candidate_url:
        return {"error": "No candidate_url in metadata"}

    ext_id = row["player_external_id"]
    player_name = meta.get("player_name", "")

    # 2. Download source image
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(candidate_url)
            if resp.status_code != 200:
                return {"error": f"Image download failed ({resp.status_code})"}
            image_bytes = resp.content
    except Exception as e:
        return {"error": f"Image download error: {e}"}

    # 3. Apply crop
    manual_crop = meta.get("manual_crop")
    if manual_crop:
        img = PILImage.open(io.BytesIO(image_bytes))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        aw, ah = img.size

        sw = manual_crop.get("source_width", aw)
        sh = manual_crop.get("source_height", ah)
        scale_x = aw / sw if sw > 0 else 1.0
        scale_y = ah / sh if sh > 0 else 1.0

        cx = max(0, int(round(manual_crop["x"] * scale_x)))
        cy = max(0, int(round(manual_crop["y"] * scale_y)))
        cs = max(1, int(round(manual_crop["size"] * min(scale_x, scale_y))))
        cs = min(cs, aw, ah)
        cx = min(cx, max(0, aw - cs))
        cy = min(cy, max(0, ah - cs))

        cropped = img.crop((cx, cy, cx + cs, cy + cs))
        cropped = cropped.resize((512, 512), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        cropped.save(buf, format="PNG", optimize=True)
        crop_bytes = buf.getvalue()
    else:
        crop_bytes = crop_face(
            image_bytes, output_size=512,
            player_name=player_name, player_ext_id=ext_id,
        )
        if not crop_bytes:
            return {"error": "Auto crop failed"}

    # 4. PhotoRoom background removal
    clean_bytes = await remove_background(crop_bytes)
    if not clean_bytes:
        logger.warning(f"promote #{candidate_id}: PhotoRoom failed, using cropped image")
        clean_bytes = crop_bytes

    # 5. Upload to R2
    r2_client = get_logos_r2_client()
    if not r2_client:
        return {"error": "R2 client not available"}

    content_hash = compute_content_hash(clean_bytes)
    r2_key = build_player_photo_key(ext_id, content_hash, "card", "segmented")

    success = await r2_client.put_object(r2_key, clean_bytes, "image/png")
    if not success:
        return {"error": "R2 upload failed"}

    from app.logos.config import get_logos_settings
    cdn_base = get_logos_settings().LOGOS_CDN_BASE_URL.rstrip("/")
    cdn_url = f"{cdn_base}/{r2_key}"

    # 6. Deactivate previous active cards
    await session.execute(text("""
        UPDATE player_photo_assets
        SET is_active = false, deactivated_at = NOW(), changed_by = 'manual'
        WHERE player_external_id = :ext_id
          AND asset_type = 'card'
          AND is_active = true
    """), {"ext_id": ext_id})

    # 7. Insert new card asset
    insert_result = await session.execute(text("""
        INSERT INTO player_photo_assets (
            player_external_id, context_team_id, season, role, kit_variant,
            asset_type, style, r2_key, cdn_url, content_hash, revision,
            source, processor, quality_score, photo_meta,
            review_status, is_active, activated_at, changed_by,
            created_at, updated_at
        ) VALUES (
            :ext_id, :team_id, :season, :role, :kit,
            'card', 'segmented', :r2_key, :cdn_url, :hash, 1,
            :source, 'photoroom', :quality, CAST(:meta AS jsonb),
            'approved', true, NOW(), 'manual',
            NOW(), NOW()
        )
        RETURNING id
    """), {
        "ext_id": ext_id,
        "team_id": row["context_team_id"],
        "season": row["season"],
        "role": row["role"],
        "kit": row["kit_variant"],
        "r2_key": r2_key,
        "cdn_url": cdn_url,
        "hash": content_hash,
        "source": row["source"] or "club_site",
        "quality": meta.get("quality_score"),
        "meta": '{"promoted_from": ' + str(candidate_id) + '}',
    })
    card_row = insert_result.mappings().first()
    await session.commit()

    card_id = card_row["id"] if card_row else None
    logger.info(
        f"promote #{candidate_id}: player {ext_id} → card #{card_id} "
        f"({len(clean_bytes)} bytes, {cdn_url})"
    )

    return {"cdn_url": cdn_url, "r2_key": r2_key, "card_id": card_id}
