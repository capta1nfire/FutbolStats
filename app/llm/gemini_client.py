"""
Google Gemini API client for narrative generation.

Provides async interface similar to RunPodClient for drop-in replacement.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"


@dataclass
class GeminiResult:
    """Result from a Gemini API call."""

    status: str  # COMPLETED, ERROR, TIMEOUT
    text: str
    tokens_in: int
    tokens_out: int
    exec_ms: int
    model_version: str
    raw_output: dict
    error: Optional[str] = None
    finish_reason: Optional[str] = None  # STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
    delay_ms: int = 0  # Queue/startup delay (0 for direct Gemini API, used by RunPod)
    # Compatibility with RunPodResult interface (used by narrative_generator.py)
    worker_id: str = ""  # Gemini has no worker concept
    job_id: str = ""  # Gemini has no job ID, could use request_id if needed


class GeminiError(Exception):
    """Error from Gemini API."""

    pass


class GeminiClient:
    """Async client for Google Gemini API."""

    def __init__(self):
        settings = get_settings()
        self.api_key = (settings.GEMINI_API_KEY or "").strip()
        self.model = getattr(settings, "GEMINI_MODEL", DEFAULT_MODEL)
        self.timeout = settings.NARRATIVE_LLM_TIMEOUT_SECONDS
        self.max_tokens = getattr(settings, "GEMINI_MAX_TOKENS", 1000)
        self.temperature = settings.NARRATIVE_LLM_TEMPERATURE
        self.top_p = settings.NARRATIVE_LLM_TOP_P

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> GeminiResult:
        """
        Generate text using Gemini API.

        Args:
            prompt: The prompt to send to the model.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.

        Returns:
            GeminiResult with generated text and metadata.
        """
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY not configured")

        client = await self._get_client()
        url = f"{GEMINI_BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "topP": top_p if top_p is not None else self.top_p,
            },
        }

        import time
        start_time = time.time()

        try:
            response = await client.post(url, json=payload)
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                error_text = response.text[:500]
                logger.error(f"Gemini API error {response.status_code}: {error_text}")
                return GeminiResult(
                    status="ERROR",
                    text="",
                    tokens_in=0,
                    tokens_out=0,
                    exec_ms=elapsed_ms,
                    model_version=self.model,
                    raw_output={},
                    error=f"HTTP {response.status_code}: {error_text}",
                )

            data = response.json()

            # Extract text and finish reason from response
            text, finish_reason = self._extract_text_and_reason(data)
            usage = data.get("usageMetadata", {})

            # Log warning if finish_reason indicates potential truncation
            if finish_reason and finish_reason != "STOP":
                logger.warning(
                    f"Gemini finishReason={finish_reason} (tokens_out={usage.get('candidatesTokenCount', 0)}, "
                    f"max_tokens={max_tokens or self.max_tokens}, text_len={len(text)})"
                )

            return GeminiResult(
                status="COMPLETED",
                text=text,
                tokens_in=usage.get("promptTokenCount", 0),
                tokens_out=usage.get("candidatesTokenCount", 0),
                exec_ms=elapsed_ms,
                model_version=data.get("modelVersion", self.model),
                raw_output=data,
                finish_reason=finish_reason,
            )

        except httpx.TimeoutException:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Gemini API timeout after {elapsed_ms}ms")
            return GeminiResult(
                status="TIMEOUT",
                text="",
                tokens_in=0,
                tokens_out=0,
                exec_ms=elapsed_ms,
                model_version=self.model,
                raw_output={},
                error="Request timed out",
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"Gemini API error: {e}")
            return GeminiResult(
                status="ERROR",
                text="",
                tokens_in=0,
                tokens_out=0,
                exec_ms=elapsed_ms,
                model_version=self.model,
                raw_output={},
                error=str(e),
            )

    def _extract_text_and_reason(self, response: dict) -> tuple[str, Optional[str]]:
        """Extract text and finishReason from Gemini response."""
        candidates = response.get("candidates", [])
        if not candidates:
            return "", None

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")

        content = candidate.get("content", {})
        parts = content.get("parts", [])

        if not parts:
            return "", finish_reason

        text = parts[0].get("text", "")
        return text, finish_reason
