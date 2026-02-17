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

_SKIN_CB_MIN = 77
_SKIN_CB_MAX = 127
_SKIN_CR_MIN = 133
_SKIN_CR_MAX = 173


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


def _mask_bbox(
    mask: np.ndarray,
    *,
    min_row_frac: float = 0.02,
    min_col_frac: float = 0.02,
) -> Optional[tuple[int, int, int, int]]:
    """Compute a robust bbox (top, bottom, left, right) for a boolean mask.

    Uses row/col occupancy thresholds to ignore speckle noise.
    Returns None if no meaningful foreground is found.
    """
    if mask is None or mask.size == 0:
        return None

    h, w = mask.shape
    if h < 2 or w < 2:
        return None

    row_min = max(1, int(w * float(min_row_frac)))
    col_min = max(1, int(h * float(min_col_frac)))

    row_sum = mask.sum(axis=1)
    col_sum = mask.sum(axis=0)

    rows = np.where(row_sum >= row_min)[0]
    cols = np.where(col_sum >= col_min)[0]

    if rows.size == 0 or cols.size == 0:
        return None

    top = int(rows.min())
    bottom = int(rows.max() + 1)
    left = int(cols.min())
    right = int(cols.max() + 1)
    return top, bottom, left, right


def _estimate_bg_rgb(rgb: np.ndarray, border: int = 12) -> tuple[np.ndarray, float]:
    """Estimate background RGB color from image borders.

    Returns (bg_rgb[3], bg_noise_p90).
    """
    h, w, _ = rgb.shape
    b = int(max(1, min(border, h // 10, w // 10)))

    # Sample border pixels (flattened)
    top = rgb[:b, :, :]
    bottom = rgb[-b:, :, :]
    left = rgb[:, :b, :]
    right = rgb[:, -b:, :]
    border_pixels = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    ).astype(np.float32)

    bg = np.median(border_pixels, axis=0)
    diffs = border_pixels - bg
    dist = np.sqrt((diffs * diffs).sum(axis=1))
    noise_p90 = float(np.percentile(dist, 90)) if dist.size else 0.0
    return bg, noise_p90


def _foreground_mask_from_bg(rgb: np.ndarray) -> np.ndarray:
    """Best-effort foreground mask for opaque portraits with uniform background."""
    bg, noise_p90 = _estimate_bg_rgb(rgb, border=12)
    diffs = rgb.astype(np.float32) - bg.reshape(1, 1, 3)
    dist = np.sqrt((diffs * diffs).sum(axis=2))

    # Adaptive threshold: be strict on clean backgrounds, looser on noisy ones.
    thr = max(22.0, noise_p90 * 3.0, 30.0 if noise_p90 > 8 else 0.0)
    return dist >= thr


def _skin_mask_ycbcr(img_rgb: Image.Image) -> np.ndarray:
    """Compute a skin-like mask in YCbCr (best-effort, no heavy deps)."""
    ycbcr = img_rgb.convert("YCbCr")
    arr = np.array(ycbcr)
    y = arr[:, :, 0].astype(np.int16)
    cb = arr[:, :, 1].astype(np.int16)
    cr = arr[:, :, 2].astype(np.int16)
    # Keep Y relaxed (dark skin/hair shadows), but avoid near-black noise.
    return (
        (y >= 35)
        & (cb >= _SKIN_CB_MIN)
        & (cb <= _SKIN_CB_MAX)
        & (cr >= _SKIN_CR_MIN)
        & (cr <= _SKIN_CR_MAX)
    )


def _clamp_crop(left: int, top: int, crop_size: int, w: int, h: int) -> tuple[int, int, int]:
    """Clamp crop box to image boundaries."""
    crop_size = int(max(1, min(crop_size, w, h)))
    left = int(max(0, min(left, w - crop_size)))
    top = int(max(0, min(top, h - crop_size)))
    return left, top, crop_size


def _compute_crop_from_face_box(
    *,
    face_box: tuple[int, int, int, int],
    w: int,
    h: int,
    scale: float = 2.35,
    y_bias: float = 0.60,
) -> tuple[int, int, int]:
    """Compute a square crop around a face-like bbox.

    y_bias controls vertical framing: center_y is placed at ~y_bias of crop height
    (higher y_bias => more forehead/space above).
    """
    top, bottom, left, right = face_box
    fw = max(1, int(right - left))
    fh = max(1, int(bottom - top))

    cx = int(round((left + right) / 2))
    cy = int(round((top + bottom) / 2))

    size = int(round(max(fw, fh) * float(scale)))
    size = max(64, min(size, w, h))

    crop_left = int(round(cx - size / 2))
    crop_top = int(round(cy - size * float(y_bias)))
    return _clamp_crop(crop_left, crop_top, size, w, h)


def crop_face(image_bytes: bytes, output_size: int = 512, player_name: str = "", player_ext_id: int = 0) -> Optional[bytes]:
    """Crop face close-up from a player portrait.

    Two modes:
    - Transparent bg: use alpha mask to locate content
    - Opaque bg: segment foreground from (mostly) uniform background

    Both paths then:
    - compute a robust content bbox
    - try a skin-like mask in the upper region to center the head
    - fall back to silhouette-based head strip if skin detection is weak

    Args:
        image_bytes: Source image bytes
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

        # For preview/headshots we only output <=512px; analyze on a smaller canvas for speed.
        max_dim = max(800, int(output_size) * 4)
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        w, h = img.size

        # Avoid reopening bytes: detect transparency from current RGBA image.
        alpha = np.array(img.split()[3])
        transparent_ratio = float(np.mean(alpha < 128))
        is_transparent = transparent_ratio >= 0.15

        if is_transparent:
            # --- Transparent mode: use alpha mask to locate head ---
            mask_fg = alpha > 128
        else:
            # --- Opaque mode: segment foreground from uniform background ---
            rgb_arr = np.array(img.convert("RGB"))
            mask_fg = _foreground_mask_from_bg(rgb_arr)

        # Content bbox (robust against speckle noise)
        bbox = _mask_bbox(mask_fg, min_row_frac=0.01, min_col_frac=0.01)
        if not bbox:
            # Fallback: full image
            bbox = (0, h, 0, w)
        content_top, content_bottom, content_left, content_right = bbox
        content_h = content_bottom - content_top
        content_w = content_right - content_left
        if content_h < 80 or content_w < 80:
            logger.warning(f"Content area too small ({content_w}x{content_h})")
            return None

        # Attempt 1: skin-based face box in the upper body region (works for opaque+transparent).
        # Restrict to top portion to avoid hands/arms lower in the frame.
        rgb_img = img.convert("RGB")
        skin = _skin_mask_ycbcr(rgb_img)

        roi_bottom = int(content_top + content_h * 0.60)
        roi_bottom = max(content_top + 1, min(roi_bottom, h))
        roi = np.zeros((h, w), dtype=bool)
        roi[content_top:roi_bottom, content_left:content_right] = True

        skin_roi = skin & roi & mask_fg
        ys, xs = np.where(skin_roi)
        face_box = None
        if ys.size >= 400:
            # Keep only the top part of skin pixels to avoid hands.
            y_cut = float(np.percentile(ys, 35))
            keep = ys <= y_cut
            ys2 = ys[keep]
            xs2 = xs[keep]
            if ys2.size >= 200:
                # Robust bounds (percentiles beat raw bbox on noisy masks)
                y1 = int(np.percentile(ys2, 5))
                y2 = int(np.percentile(ys2, 95))
                x1 = int(np.percentile(xs2, 5))
                x2 = int(np.percentile(xs2, 95))
                if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                    face_box = (y1, y2, x1, x2)

        if face_box:
            left, top, crop_size = _compute_crop_from_face_box(
                face_box=face_box,
                w=w,
                h=h,
                scale=2.35,
                y_bias=0.60,
            )
        else:
            # Attempt 2: head region from foreground silhouette in top strip of content bbox.
            strip_bottom = int(content_top + content_h * 0.38)
            strip_bottom = max(content_top + 1, min(strip_bottom, h))

            strip = mask_fg[content_top:strip_bottom, content_left:content_right]
            ys3, xs3 = np.where(strip)
            if xs3.size >= 200:
                x1 = int(np.percentile(xs3, 5))
                x2 = int(np.percentile(xs3, 95))
                y1 = int(np.percentile(ys3, 5))
                y2 = int(np.percentile(ys3, 95))
                head_box = (
                    content_top + y1,
                    content_top + y2,
                    content_left + x1,
                    content_left + x2,
                )
                left, top, crop_size = _compute_crop_from_face_box(
                    face_box=head_box,
                    w=w,
                    h=h,
                    scale=2.20,
                    y_bias=0.62,
                )
            else:
                # Last resort: conservative top-center crop from content bbox.
                size = int(min(content_w, content_h) * 0.42)
                size = max(96, min(size, w, h))
                cx = int((content_left + content_right) / 2)
                cy = int(content_top + content_h * 0.18)
                left, top, crop_size = _clamp_crop(
                    int(cx - size / 2),
                    int(cy - size * 0.62),
                    size,
                    w,
                    h,
                )

        logger.debug(f"crop_face: transparent={is_transparent}, crop=({left},{top},{crop_size}x{crop_size})")

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
