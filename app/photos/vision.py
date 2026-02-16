"""Gemini Flash Vision for Headshot Validation.

Uses Gemini Flash to validate that a candidate photo is:
- A professional headshot (not action shot, group photo, etc.)
- Single person visible
- Studio/official quality
"""

import base64
import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.photos.config import get_photos_settings

logger = logging.getLogger(__name__)
photos_settings = get_photos_settings()

GEMINI_VISION_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

VALIDATION_PROMPT = """Analyze this image and respond with a JSON object (no markdown):

{
  "is_headshot": true/false,
  "single_person": true/false,
  "professional_quality": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Criteria:
- is_headshot: True if this is a portrait or upper-body photo of a person (face clearly visible). Includes headshots, half-body portraits, and photos showing the person from the waist up. Action shots, full-body standing photos, and celebration photos should be false.
- single_person: True if exactly one person is the main subject (minor background figures are OK)
- professional_quality: True if official/studio quality (not blurry, not cropped badly, not a selfie, not a low-res thumbnail)
- confidence: Your confidence in the assessment
- reasoning: Brief explanation of your assessment"""


@dataclass
class VisionResult:
    """Result of Gemini Vision validation."""

    is_headshot: bool = False
    single_person: bool = False
    professional_quality: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    error: Optional[str] = None

    @property
    def passes(self) -> bool:
        return self.is_headshot and self.single_person and self.professional_quality


async def validate_with_vision(
    image_bytes: bytes,
    player_name: Optional[str] = None,
) -> VisionResult:
    """Validate photo using Gemini Flash Vision.

    Args:
        image_bytes: Raw image bytes (PNG/JPEG/WebP)
        player_name: Optional player name for context (not used in validation)

    Returns:
        VisionResult with headshot quality assessment
    """
    if not photos_settings.PHOTOS_GEMINI_VISION_ENABLED:
        logger.debug("Vision validation disabled, auto-passing")
        return VisionResult(
            is_headshot=True, single_person=True,
            professional_quality=True, confidence=0.0,
            reasoning="Vision validation disabled",
        )

    api_key = photos_settings.GEMINI_API_KEY
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, skipping vision validation")
        return VisionResult(error="GEMINI_API_KEY not configured")

    # Encode image to base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Detect MIME type from first bytes
    mime_type = "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        mime_type = "image/jpeg"
    elif image_bytes[:4] == b"RIFF":
        mime_type = "image/webp"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": VALIDATION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_image,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 256,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{GEMINI_VISION_URL}?key={api_key}",
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        result = json.loads(text)

        return VisionResult(
            is_headshot=result.get("is_headshot", False),
            single_person=result.get("single_person", False),
            professional_quality=result.get("professional_quality", False),
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning", ""),
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"Gemini Vision API error: {e.response.status_code}")
        return VisionResult(error=f"API error: {e.response.status_code}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Failed to parse Gemini Vision response: {e}")
        return VisionResult(error=f"Parse error: {e}")
    except Exception as e:
        logger.error(f"Vision validation failed: {e}")
        return VisionResult(error=str(e))
