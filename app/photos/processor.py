"""Image Processing for Player Photos.

Handles cropping and thumbnail generation for player headshots.
Pattern follows app/logos/processor.py.

Piloto: heuristic crop (no OpenCV) + Pillow thumbnails.
Fix #6: NO heavy dependencies (opencv-python-headless) in pilot.
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from app.photos.config import get_photos_settings

logger = logging.getLogger(__name__)
photos_settings = get_photos_settings()


@dataclass
class CropResult:
    """Result of smart cropping."""

    card_bytes: Optional[bytes] = None  # ~400x600 portrait crop
    thumb_bytes: Optional[bytes] = None  # ~256x256 square crop
    error: Optional[str] = None


def smart_crop(
    image_bytes: bytes,
    card_width: int = 400,
    card_height: int = 600,
    thumb_size: int = 256,
) -> CropResult:
    """Heuristic smart crop for player headshot.

    Assumes Gemini Vision already validated this is a single-person headshot.
    Uses center-weighted crop (upper 2/3 for card, center for thumb).

    Args:
        image_bytes: Source image bytes (ideally with transparent background)
        card_width: Target card width
        card_height: Target card height
        thumb_size: Target thumbnail size (square)

    Returns:
        CropResult with card and thumb bytes
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        w, h = img.size

        # Card crop: upper 2/3 of image, center-aligned, resize to card dimensions
        card_crop_h = int(h * 0.85)  # top 85% (head + shoulders)
        card_region = img.crop((0, 0, w, card_crop_h))
        card_img = card_region.resize((card_width, card_height), Image.Resampling.LANCZOS)

        card_buf = io.BytesIO()
        card_img.save(card_buf, format="PNG", optimize=True)

        # Thumb crop: center square, resize to thumb_size
        if w > h:
            left = (w - h) // 2
            thumb_region = img.crop((left, 0, left + h, h))
        else:
            top = int(h * 0.05)  # slight offset down to center on face
            size = min(w, h - top)
            thumb_region = img.crop((0, top, size, top + size))

        thumb_img = thumb_region.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)

        thumb_buf = io.BytesIO()
        thumb_img.save(thumb_buf, format="WEBP", quality=photos_settings.PHOTOS_THUMBNAIL_QUALITY)

        return CropResult(
            card_bytes=card_buf.getvalue(),
            thumb_bytes=thumb_buf.getvalue(),
        )

    except Exception as e:
        logger.error(f"Smart crop failed: {e}")
        return CropResult(error=str(e))


def generate_thumbnails(
    image_bytes: bytes,
    sizes: Optional[list] = None,
) -> dict:
    """Generate WebP thumbnails at multiple sizes.

    Pattern follows app/logos/processor.py.

    Args:
        image_bytes: Source image bytes
        sizes: Target sizes (default from settings)

    Returns:
        Dict mapping size -> WebP bytes
    """
    sizes = sizes or photos_settings.PHOTOS_THUMBNAIL_SIZES
    quality = photos_settings.PHOTOS_THUMBNAIL_QUALITY
    thumbnails = {}

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGBA", "LA"):
            img = img.convert("RGBA")
    except Exception as e:
        logger.error(f"Failed to load source image for thumbnails: {e}")
        return thumbnails

    for size in sizes:
        try:
            thumb = img.copy()
            thumb.thumbnail((size, size), Image.Resampling.LANCZOS)

            # Pad to square if needed
            if thumb.width != thumb.height:
                square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                x_off = (size - thumb.width) // 2
                y_off = (size - thumb.height) // 2
                square.paste(thumb, (x_off, y_off))
                thumb = square

            buf = io.BytesIO()
            thumb.save(buf, format="WEBP", quality=quality, lossless=False)
            thumbnails[size] = buf.getvalue()
            logger.debug(f"Generated {size}px thumbnail ({len(thumbnails[size])} bytes)")

        except Exception as e:
            logger.error(f"Failed to generate {size}px thumbnail: {e}")

    return thumbnails
