"""Image Processing for Player Photos.

New flow (2026-02-16):
1. Detect if image already has transparent background
2. If not transparent → send to PhotoRoom for bg removal
3. Keep original clean (full body, no crop) as "original"
4. From original, crop face close-up as "face"
5. Next.js <Image> handles on-demand sizing (no pre-generated thumbnails)

No heavy dependencies (opencv-python-headless). Face crop is heuristic.
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

logger = logging.getLogger(__name__)


@dataclass
class ProcessedPhoto:
    """Result of photo processing."""

    original_bytes: Optional[bytes] = None  # Full clean image (bg removed), PNG
    face_bytes: Optional[bytes] = None      # Face close-up crop, PNG
    has_transparency: bool = False           # Source already had transparent bg
    error: Optional[str] = None


def has_transparent_background(image_bytes: bytes, threshold: float = 0.15) -> bool:
    """Detect if image already has a transparent background.

    Checks if >15% of pixels have alpha < 128 (semi-transparent or fully transparent).
    This avoids wasting PhotoRoom credits on already-clean images.

    Args:
        image_bytes: Raw image bytes
        threshold: Fraction of transparent pixels needed (default 15%)

    Returns:
        True if the image has significant transparency
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGBA":
            return False

        alpha = np.array(img.split()[3])
        transparent_pixels = np.sum(alpha < 128)
        total_pixels = alpha.size

        ratio = transparent_pixels / total_pixels
        logger.debug(f"Transparency check: {ratio:.1%} transparent pixels")
        return ratio >= threshold

    except Exception as e:
        logger.warning(f"Transparency detection failed: {e}")
        return False


def _find_content_bbox(img):
    """Find bounding box of non-transparent content in an RGBA image.

    Returns (top, bottom, left, right) of the content area,
    or the full image bounds if no alpha channel.
    """
    w, h = img.size
    if img.mode != "RGBA":
        return 0, h, 0, w

    alpha = np.array(img.split()[3])
    rows_with_content = np.any(alpha > 128, axis=1)
    cols_with_content = np.any(alpha > 128, axis=0)

    if not np.any(rows_with_content):
        return 0, h, 0, w

    top = int(np.argmax(rows_with_content))
    bottom = int(h - np.argmax(rows_with_content[::-1]))
    left = int(np.argmax(cols_with_content))
    right = int(w - np.argmax(cols_with_content[::-1]))

    return top, bottom, left, right


def crop_face(image_bytes: bytes, output_size: int = 512, player_name: str = "", player_ext_id: int = 0) -> Optional[bytes]:
    """Crop face close-up from a player portrait.

    Detects the actual content area (skipping transparent padding),
    then takes the top 35% of the content height for the face region.

    Args:
        image_bytes: Source image bytes (should have transparent bg)
        output_size: Output square size in pixels
        player_name: Player name for embedded metadata attribution
        player_ext_id: API-Football player ID for cross-referencing

    Returns:
        PNG bytes of face crop, or None on failure
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        w, h = img.size

        # Find where the actual content starts (skip transparent padding)
        content_top, content_bottom, content_left, content_right = _find_content_bbox(img)
        content_h = content_bottom - content_top
        content_w = content_right - content_left

        logger.debug(f"Content bbox: top={content_top} bottom={content_bottom} ({content_h}px h), left={content_left} right={content_right} ({content_w}px w)")

        if content_h < 50 or content_w < 50:
            logger.warning(f"Content area too small ({content_w}x{content_h})")
            return None

        # Take top 30% of CONTENT height (tighter face crop)
        face_h = int(content_h * 0.30)

        # Center horizontally on the content bounding box.
        # For transparent-bg images, this centers on the person silhouette.
        # For opaque images (pre-PhotoRoom), this centers on the full image.
        content_center_x = content_left + content_w // 2

        crop_size = min(content_w, face_h)
        left = max(0, content_center_x - crop_size // 2)
        left = min(left, w - crop_size)  # clamp to image bounds
        top = content_top

        logger.debug(f"Face center X: {face_center_x}, crop left: {left}, crop_size: {crop_size}")

        face_region = img.crop((left, top, left + crop_size, top + crop_size))
        face_img = face_region.resize((output_size, output_size), Image.Resampling.LANCZOS)

        png_meta = PngInfo()
        png_meta.add_text("Author", "BON JOGO")
        png_meta.add_text("Copyright", "BON JOGO - bonjogo.com")
        png_meta.add_text("Source", "https://bonjogo.com")
        id_part = f" | id:{player_ext_id}" if player_ext_id else ""
        if player_name:
            png_meta.add_text("Comment", f"{player_name}{id_part} | bonjogo.com")
        buf = io.BytesIO()
        face_img.save(buf, format="PNG", optimize=True, pnginfo=png_meta)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Face crop failed: {e}")
        return None


def process_photo(image_bytes: bytes) -> ProcessedPhoto:
    """Process a player photo: detect transparency, prepare original + face crop.

    This does NOT call PhotoRoom — that's handled by the caller based on
    the has_transparency flag.

    Args:
        image_bytes: Raw source image bytes

    Returns:
        ProcessedPhoto with original (re-saved as PNG) and face crop
    """
    try:
        transparency = has_transparent_background(image_bytes)

        # Re-save original as optimized PNG (normalize format)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        original_buf = io.BytesIO()
        img.save(original_buf, format="PNG", optimize=True)
        original_bytes = original_buf.getvalue()

        # Face crop
        face_bytes = crop_face(original_bytes)

        return ProcessedPhoto(
            original_bytes=original_bytes,
            face_bytes=face_bytes,
            has_transparency=transparency,
        )

    except Exception as e:
        logger.error(f"Photo processing failed: {e}")
        return ProcessedPhoto(error=str(e))
