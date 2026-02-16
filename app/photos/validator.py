"""Player Photo QA Gate.

Validates candidate photos before processing and storage.
Pattern follows app/logos/validator.py but adapted for player headshots.

Checks:
- Dimensions >= 200x200
- Aspect ratio 0.6-1.0 (portrait to square)
- File size 5KB-500KB
- Format PNG/JPEG/WebP
- SHA-256 hash for dedup
"""

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from app.photos.config import get_photos_settings, compute_content_hash

logger = logging.getLogger(__name__)
photos_settings = get_photos_settings()


@dataclass
class ValidationResult:
    """Result of player photo validation."""

    valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # Image metadata
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    mode: Optional[str] = None
    file_size_bytes: int = 0
    content_hash: Optional[str] = None

    def __str__(self) -> str:
        if self.valid:
            return f"Valid ({self.width}x{self.height}, {self.format}, {self.content_hash[:12] if self.content_hash else '?'})"
        return f"Invalid: {', '.join(self.errors)}"


def validate_player_photo(
    image_bytes: bytes,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
) -> ValidationResult:
    """Validate a candidate player photo.

    Args:
        image_bytes: Raw image bytes
        min_width: Minimum width (default from settings)
        min_height: Minimum height (default from settings)

    Returns:
        ValidationResult with valid flag, errors, metadata, and content_hash
    """
    min_width = min_width or photos_settings.PHOTOS_MIN_WIDTH
    min_height = min_height or photos_settings.PHOTOS_MIN_HEIGHT
    max_size = photos_settings.PHOTOS_MAX_FILE_SIZE_BYTES
    min_size = photos_settings.PHOTOS_MIN_FILE_SIZE_BYTES
    aspect_min = photos_settings.PHOTOS_ASPECT_RATIO_MIN
    aspect_max = photos_settings.PHOTOS_ASPECT_RATIO_MAX

    errors = []
    warnings = []

    # File size checks
    file_size = len(image_bytes)
    if file_size == 0:
        return ValidationResult(valid=False, errors=["Empty image data"])

    if file_size < min_size:
        errors.append(f"File size {file_size} bytes below minimum {min_size} bytes (likely placeholder)")
        return ValidationResult(valid=False, errors=errors, file_size_bytes=file_size)

    if file_size > max_size:
        errors.append(
            f"File size {file_size / 1024:.1f}KB exceeds limit {max_size / 1024:.1f}KB"
        )
        return ValidationResult(valid=False, errors=errors, file_size_bytes=file_size)

    # Compute content hash for dedup
    content_hash = compute_content_hash(image_bytes)

    # Load image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        errors.append(f"Image corrupted or invalid format: {e}")
        return ValidationResult(
            valid=False, errors=errors, file_size_bytes=file_size,
            content_hash=content_hash,
        )

    width, height = img.size
    img_format = img.format
    img_mode = img.mode

    # Format check
    if img_format not in ("PNG", "JPEG", "WEBP"):
        errors.append(f"Unsupported format '{img_format}' (need PNG/JPEG/WebP)")

    # Dimension checks
    if width < min_width:
        errors.append(f"Width {width}px below minimum {min_width}px")
    if height < min_height:
        errors.append(f"Height {height}px below minimum {min_height}px")

    # Aspect ratio check (portrait to square, not landscape)
    if width > 0 and height > 0:
        ratio = width / height
        if ratio < aspect_min:
            errors.append(f"Aspect ratio {ratio:.2f} too narrow (min {aspect_min})")
        if ratio > aspect_max:
            warnings.append(f"Aspect ratio {ratio:.2f} is landscape (max {aspect_max})")

    valid = len(errors) == 0

    if not valid:
        logger.debug(f"Photo validation failed: {errors}")
    elif warnings:
        logger.debug(f"Photo validation passed with warnings: {warnings}")

    return ValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        width=width,
        height=height,
        format=img_format,
        mode=img_mode,
        file_size_bytes=file_size,
        content_hash=content_hash,
    )
