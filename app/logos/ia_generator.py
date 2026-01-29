"""IA Logo Generator Interface.

Abstracts IA model calls for 3D logo generation.
Supports multiple backends: DALL-E 3, SDXL (Replicate).

Usage:
    generator = get_ia_generator("dall-e-3")
    image_bytes = await generator.generate(
        original_url="https://...",
        prompt="Transform this 2D shield..."
    )
"""

import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import Optional, Protocol

import httpx

from app.logos.config import get_logos_settings

logger = logging.getLogger(__name__)
logos_settings = get_logos_settings()


class IALogoGenerator(Protocol):
    """Protocol for IA logo generators."""

    @abstractmethod
    async def generate(
        self,
        original_image: bytes,
        prompt: str,
        size: str = "1024x1024",
    ) -> Optional[bytes]:
        """Generate 3D logo from original image.

        Args:
            original_image: Original logo image bytes
            prompt: Generation prompt
            size: Output size (default 1024x1024)

        Returns:
            Generated image bytes or None if failed
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the generator is available."""
        ...


class DallEGenerator:
    """DALL-E 3 generator using OpenAI API.

    Note: DALL-E 3 doesn't support image-to-image directly.
    Uses image variation endpoint for DALL-E 2 or creates from scratch with DALL-E 3.
    For true image-to-image, consider using GPT-4 Vision to describe + DALL-E 3.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    async def generate(
        self,
        original_image: bytes,
        prompt: str,
        size: str = "1024x1024",
    ) -> Optional[bytes]:
        """Generate image using DALL-E 3.

        Since DALL-E 3 doesn't support direct image editing, we use the prompt
        to describe what we want. For better results, consider:
        1. Using GPT-4 Vision to describe the original logo
        2. Including that description in the DALL-E prompt

        Args:
            original_image: Original logo (used for context if GPT-4V available)
            prompt: Full generation prompt
            size: Output size

        Returns:
            Generated image bytes or None
        """
        async with httpx.AsyncClient() as client:
            try:
                # DALL-E 3 image generation
                response = await client.post(
                    f"{self.base_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "dall-e-3",
                        "prompt": prompt,
                        "n": 1,
                        "size": size,
                        "quality": "hd",
                        "response_format": "b64_json",
                    },
                    timeout=120.0,
                )

                if response.status_code != 200:
                    error = response.json().get("error", {}).get("message", "Unknown")
                    logger.error(f"DALL-E API error: {error}")
                    return None

                data = response.json()
                b64_image = data["data"][0]["b64_json"]
                image_bytes = base64.b64decode(b64_image)

                logger.info(f"DALL-E generated {len(image_bytes)} bytes")
                return image_bytes

            except httpx.TimeoutException:
                logger.error("DALL-E timeout")
                return None
            except Exception as e:
                logger.error(f"DALL-E error: {e}")
                return None

    async def generate_with_edit(
        self,
        original_image: bytes,
        prompt: str,
        mask: Optional[bytes] = None,
        size: str = "1024x1024",
    ) -> Optional[bytes]:
        """Generate using DALL-E 2 edit endpoint (supports image input).

        DALL-E 2 can edit images but has lower quality than DALL-E 3.
        The image must be PNG with transparency for the mask.

        Args:
            original_image: Original logo PNG with alpha
            prompt: Edit prompt
            mask: Optional mask (transparent areas to edit)
            size: Output size

        Returns:
            Generated image bytes or None
        """
        async with httpx.AsyncClient() as client:
            try:
                files = {
                    "image": ("logo.png", original_image, "image/png"),
                    "prompt": (None, prompt),
                    "n": (None, "1"),
                    "size": (None, size),
                    "response_format": (None, "b64_json"),
                }

                if mask:
                    files["mask"] = ("mask.png", mask, "image/png")

                response = await client.post(
                    f"{self.base_url}/images/edits",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    timeout=120.0,
                )

                if response.status_code != 200:
                    error = response.json().get("error", {}).get("message", "Unknown")
                    logger.error(f"DALL-E edit API error: {error}")
                    return None

                data = response.json()
                b64_image = data["data"][0]["b64_json"]
                return base64.b64decode(b64_image)

            except Exception as e:
                logger.error(f"DALL-E edit error: {e}")
                return None

    async def health_check(self) -> bool:
        """Check OpenAI API availability."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10.0,
                )
                return response.status_code == 200
            except Exception:
                return False


class ImagenGenerator:
    """Google Imagen/Gemini generator.

    Uses Gemini API for image generation and editing.
    Supports img2img via Gemini's image understanding + generation.

    Tiers:
    - Free tier (LOGOS_USE_FREE_TIER=True): Google AI Studio, ~50 imgs/day limit
    - Paid tier (LOGOS_USE_FREE_TIER=False): Vertex AI, $0.03/img, no limit
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.use_free_tier = logos_settings.LOGOS_USE_FREE_TIER

        # Google AI Studio (free tier) - uses API key auth
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        # Use Gemini 2.0 Flash for img2img (image understanding + generation)
        self.model = "gemini-2.0-flash-exp"

        tier_name = "Google AI Studio (FREE)" if self.use_free_tier else "Vertex AI (PAID)"
        logger.info(f"ImagenGenerator initialized with {tier_name}")

    async def generate(
        self,
        original_image: bytes,
        prompt: str,
        size: str = "1024x1024",
    ) -> Optional[bytes]:
        """Generate image using Gemini img2img.

        Uses Gemini's multimodal capabilities:
        1. Send original image + prompt
        2. Ask to transform/regenerate based on prompt

        Args:
            original_image: Original logo image bytes
            prompt: Transformation prompt
            size: Output size (not directly controllable, uses 1:1)

        Returns:
            Generated image bytes or None
        """
        # Encode image to base64
        b64_image = base64.b64encode(original_image).decode("utf-8")

        # Determine mime type (assume PNG for logos)
        mime_type = "image/png"

        async with httpx.AsyncClient() as client:
            try:
                # Use Gemini's generateContent with image input
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": b64_image,
                                        }
                                    },
                                    {
                                        "text": f"Transform this logo image: {prompt}. Return ONLY the transformed image, no text."
                                    },
                                ]
                            }
                        ],
                        "generationConfig": {
                            "responseModalities": ["IMAGE", "TEXT"],
                            "responseMimeType": "image/png",
                        },
                    },
                    timeout=120.0,
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"Gemini API error {response.status_code}: {error_text}")
                    return None

                data = response.json()

                # Extract image from response
                candidates = data.get("candidates", [])
                if not candidates:
                    logger.error("Gemini returned no candidates")
                    return None

                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "inlineData" in part:
                        b64_result = part["inlineData"]["data"]
                        image_bytes = base64.b64decode(b64_result)
                        logger.info(f"Gemini generated {len(image_bytes)} bytes")
                        return image_bytes

                logger.error("Gemini response contains no image data")
                return None

            except httpx.TimeoutException:
                logger.error("Gemini timeout")
                return None
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                return None

    async def generate_text_to_image(
        self,
        prompt: str,
        size: str = "1024x1024",
    ) -> Optional[bytes]:
        """Generate image from text only using Imagen 4.

        For cases where no original image is available.

        Args:
            prompt: Generation prompt
            size: Output size

        Returns:
            Generated image bytes or None
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/models/imagen-4.0-generate-001:predict",
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "instances": [{"prompt": prompt}],
                        "parameters": {"sampleCount": 1},
                    },
                    timeout=120.0,
                )

                if response.status_code != 200:
                    logger.error(f"Imagen API error: {response.text}")
                    return None

                data = response.json()
                predictions = data.get("predictions", [])
                if predictions and "bytesBase64Encoded" in predictions[0]:
                    b64_image = predictions[0]["bytesBase64Encoded"]
                    return base64.b64decode(b64_image)

                logger.error("Imagen response contains no image")
                return None

            except Exception as e:
                logger.error(f"Imagen error: {e}")
                return None

    async def health_check(self) -> bool:
        """Check Gemini API availability."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    params={"key": self.api_key},
                    timeout=10.0,
                )
                return response.status_code == 200
            except Exception:
                return False


class SDXLGenerator:
    """SDXL generator using Replicate API.

    Supports image-to-image generation with better control.
    Uses SDXL img2img pipeline.
    """

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.replicate.com/v1"
        # SDXL img2img model
        self.model_version = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

    async def generate(
        self,
        original_image: bytes,
        prompt: str,
        size: str = "1024x1024",
    ) -> Optional[bytes]:
        """Generate image using SDXL img2img.

        Args:
            original_image: Original logo image bytes
            prompt: Generation prompt
            size: Output size (parsed for width/height)

        Returns:
            Generated image bytes or None
        """
        # Parse size
        try:
            width, height = map(int, size.split("x"))
        except ValueError:
            width, height = 1024, 1024

        # Encode image to base64 data URL
        b64_image = base64.b64encode(original_image).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_image}"

        async with httpx.AsyncClient() as client:
            try:
                # Start prediction
                response = await client.post(
                    f"{self.base_url}/predictions",
                    headers={
                        "Authorization": f"Token {self.api_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "version": self.model_version.split(":")[-1],
                        "input": {
                            "image": image_url,
                            "prompt": prompt,
                            "width": width,
                            "height": height,
                            "num_outputs": 1,
                            "guidance_scale": 7.5,
                            "prompt_strength": 0.8,
                            "num_inference_steps": 50,
                        },
                    },
                    timeout=30.0,
                )

                if response.status_code not in (200, 201):
                    logger.error(f"SDXL API error: {response.text}")
                    return None

                prediction = response.json()
                prediction_id = prediction["id"]

                # Poll for completion
                for _ in range(60):  # Max 2 minutes
                    import asyncio
                    await asyncio.sleep(2)

                    status_response = await client.get(
                        f"{self.base_url}/predictions/{prediction_id}",
                        headers={"Authorization": f"Token {self.api_token}"},
                        timeout=10.0,
                    )

                    status = status_response.json()

                    if status["status"] == "succeeded":
                        output_url = status["output"][0]
                        # Download the image
                        img_response = await client.get(output_url, timeout=30.0)
                        if img_response.status_code == 200:
                            logger.info(f"SDXL generated {len(img_response.content)} bytes")
                            return img_response.content
                        break

                    elif status["status"] == "failed":
                        logger.error(f"SDXL prediction failed: {status.get('error')}")
                        return None

                logger.error("SDXL prediction timeout")
                return None

            except Exception as e:
                logger.error(f"SDXL error: {e}")
                return None

    async def health_check(self) -> bool:
        """Check Replicate API availability."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models/stability-ai/sdxl",
                    headers={"Authorization": f"Token {self.api_token}"},
                    timeout=10.0,
                )
                return response.status_code == 200
            except Exception:
                return False


def get_ia_generator(model: Optional[str] = None) -> Optional[IALogoGenerator]:
    """Get IA generator instance for specified model.

    Args:
        model: Model name (imagen-3, dall-e-3, sdxl) or None for default

    Returns:
        Generator instance or None if not configured
    """
    model = model or logos_settings.LOGOS_IA_MODEL

    if model in ("imagen-3", "imagen-4", "gemini"):
        api_key = logos_settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("Imagen/Gemini requested but GEMINI_API_KEY not configured")
            return None
        return ImagenGenerator(api_key)

    elif model == "dall-e-3":
        api_key = logos_settings.OPENAI_API_KEY
        if not api_key:
            logger.warning("DALL-E requested but OPENAI_API_KEY not configured")
            return None
        return DallEGenerator(api_key)

    elif model == "sdxl":
        api_token = logos_settings.REPLICATE_API_TOKEN
        if not api_token:
            logger.warning("SDXL requested but REPLICATE_API_TOKEN not configured")
            return None
        return SDXLGenerator(api_token)

    else:
        logger.error(f"Unknown IA model: {model}")
        return None


async def generate_logo_variant(
    original_image: bytes,
    prompt: str,
    model: Optional[str] = None,
) -> Optional[bytes]:
    """Convenience function to generate a single logo variant.

    Args:
        original_image: Original logo bytes
        prompt: Generation prompt
        model: IA model to use (default from settings)

    Returns:
        Generated image bytes or None
    """
    generator = get_ia_generator(model)
    if not generator:
        return None

    return await generator.generate(original_image, prompt)
