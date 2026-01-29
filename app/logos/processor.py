"""Image Processing for Logo Thumbnails.

Handles resizing original/IA-generated logos to multiple thumbnail sizes.
Uses Pillow for image manipulation.

Output format: WebP (smaller files, good quality, transparency support)
Sizes: 64, 128, 256, 512 pixels (configurable)
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from app.logos.config import get_logos_settings

logger = logging.getLogger(__name__)
logos_settings = get_logos_settings()


@dataclass
class ProcessedThumbnail:
    """Result of thumbnail generation."""

    size: int
    image_bytes: bytes
    width: int
    height: int
    format: str = "WEBP"


@dataclass
class ProcessingResult:
    """Result of processing a logo into thumbnails."""

    success: bool
    thumbnails: dict[int, ProcessedThumbnail]  # size -> thumbnail
    errors: list[str]
    original_width: Optional[int] = None
    original_height: Optional[int] = None


def resize_to_thumbnail(
    image_bytes: bytes,
    target_size: int,
    output_format: str = "WEBP",
    quality: int = 85,
) -> Optional[ProcessedThumbnail]:
    """Resize image to square thumbnail.

    Uses high-quality LANCZOS resampling for downscaling.
    Maintains aspect ratio and pads/centers if not square.

    Args:
        image_bytes: Source image bytes
        target_size: Target width and height in pixels
        output_format: Output format (WEBP recommended)
        quality: Compression quality (1-100)

    Returns:
        ProcessedThumbnail or None if processing fails
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGBA to ensure alpha channel
        if img.mode not in ("RGBA", "LA"):
            img = img.convert("RGBA")

        # Use thumbnail method (maintains aspect ratio, efficient)
        # Create a copy to avoid modifying original
        thumb = img.copy()
        thumb.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        # If not square, create square canvas and center
        if thumb.width != thumb.height:
            square = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
            x_offset = (target_size - thumb.width) // 2
            y_offset = (target_size - thumb.height) // 2
            square.paste(thumb, (x_offset, y_offset))
            thumb = square

        # Save to bytes
        output = io.BytesIO()
        if output_format == "WEBP":
            thumb.save(output, format="WEBP", quality=quality, lossless=False)
        elif output_format == "PNG":
            thumb.save(output, format="PNG", optimize=True)
        else:
            thumb.save(output, format=output_format, quality=quality)

        return ProcessedThumbnail(
            size=target_size,
            image_bytes=output.getvalue(),
            width=thumb.width,
            height=thumb.height,
            format=output_format,
        )

    except Exception as e:
        logger.error(f"Failed to resize to {target_size}px: {e}")
        return None


def process_logo_thumbnails(
    image_bytes: bytes,
    sizes: Optional[list[int]] = None,
    output_format: Optional[str] = None,
    quality: Optional[int] = None,
) -> ProcessingResult:
    """Generate all thumbnail sizes for a logo.

    Args:
        image_bytes: Source image bytes (PNG or similar)
        sizes: List of target sizes (default from settings)
        output_format: Output format (default from settings)
        quality: Compression quality (default from settings)

    Returns:
        ProcessingResult with all thumbnails
    """
    sizes = sizes or logos_settings.LOGOS_THUMBNAIL_SIZES
    output_format = output_format or logos_settings.LOGOS_THUMBNAIL_FORMAT.upper()
    quality = quality or logos_settings.LOGOS_THUMBNAIL_QUALITY

    thumbnails: dict[int, ProcessedThumbnail] = {}
    errors: list[str] = []
    original_width = None
    original_height = None

    # Get original dimensions
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img.size
    except Exception as e:
        errors.append(f"Failed to read source image: {e}")
        return ProcessingResult(
            success=False,
            thumbnails={},
            errors=errors,
        )

    # Generate each size
    for size in sizes:
        thumb = resize_to_thumbnail(image_bytes, size, output_format, quality)
        if thumb:
            thumbnails[size] = thumb
            logger.debug(f"Generated {size}px thumbnail ({len(thumb.image_bytes)} bytes)")
        else:
            errors.append(f"Failed to generate {size}px thumbnail")

    success = len(errors) == 0 and len(thumbnails) == len(sizes)

    if success:
        logger.info(
            f"Processed {len(thumbnails)} thumbnails from {original_width}x{original_height}"
        )
    else:
        logger.warning(f"Thumbnail processing completed with errors: {errors}")

    return ProcessingResult(
        success=success,
        thumbnails=thumbnails,
        errors=errors,
        original_width=original_width,
        original_height=original_height,
    )


def convert_to_png(image_bytes: bytes) -> Optional[bytes]:
    """Convert image to PNG format.

    Useful for standardizing IA output before storage.

    Args:
        image_bytes: Source image in any supported format

    Returns:
        PNG bytes or None if conversion fails
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Ensure RGBA for transparency
        if img.mode not in ("RGBA", "LA"):
            img = img.convert("RGBA")

        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True)
        return output.getvalue()

    except Exception as e:
        logger.error(f"Failed to convert to PNG: {e}")
        return None


def convert_to_webp(
    image_bytes: bytes,
    quality: int = 85,
    lossless: bool = False,
) -> Optional[bytes]:
    """Convert image to WebP format.

    Args:
        image_bytes: Source image
        quality: Compression quality (ignored if lossless)
        lossless: Use lossless compression

    Returns:
        WebP bytes or None if conversion fails
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Ensure RGBA for transparency
        if img.mode not in ("RGBA", "LA"):
            img = img.convert("RGBA")

        output = io.BytesIO()
        img.save(output, format="WEBP", quality=quality, lossless=lossless)
        return output.getvalue()

    except Exception as e:
        logger.error(f"Failed to convert to WebP: {e}")
        return None


def get_image_info(image_bytes: bytes) -> Optional[dict]:
    """Get basic image information.

    Args:
        image_bytes: Image data

    Returns:
        Dict with width, height, format, mode or None if invalid
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "has_alpha": img.mode in ("RGBA", "LA", "PA"),
        }
    except Exception:
        return None
