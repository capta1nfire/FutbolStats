"""
RunPod Serverless v2 async client for Qwen/vLLM narrative generation.

Flow:
1. POST /run -> job_id
2. Poll /status/<job_id> until COMPLETED
3. Extract text from output[0].choices[0].tokens
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RunPodJobResult:
    """Result from a completed RunPod job."""

    job_id: str
    status: str
    text: str
    tokens_in: int
    tokens_out: int
    delay_ms: int
    exec_ms: int
    worker_id: str
    raw_output: dict


class RunPodError(Exception):
    """Error from RunPod API."""

    pass


class RunPodClient:
    """Async client for RunPod Serverless v2 API."""

    def __init__(self):
        settings = get_settings()
        # Strip whitespace/newlines from URL components to avoid httpx errors
        self.api_key = (settings.RUNPOD_API_KEY or "").strip()
        self.endpoint_id = (settings.RUNPOD_ENDPOINT_ID or "").strip()
        self.base_url = (settings.RUNPOD_BASE_URL or "").strip().rstrip("/")
        self.timeout = settings.NARRATIVE_LLM_TIMEOUT_SECONDS
        self.poll_interval = settings.NARRATIVE_LLM_POLL_INTERVAL_SECONDS
        self.max_tokens = settings.NARRATIVE_LLM_MAX_TOKENS
        self.temperature = settings.NARRATIVE_LLM_TEMPERATURE
        self.top_p = settings.NARRATIVE_LLM_TOP_P

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "authorization": self.api_key,  # No Bearer prefix
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def run_job(self, prompt: str) -> str:
        """
        Submit a job to RunPod with retries for transient errors.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            Job ID string.
        """
        client = await self._get_client()
        url = f"{self.base_url}/{self.endpoint_id}/run"

        payload = {
            "input": {
                "prompt": prompt,
                "sampling_params": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            }
        }
        logger.debug(f"RunPod payload: max_tokens={self.max_tokens}")

        # Retry with exponential backoff for transient errors
        max_retries = 2
        backoff_delays = [2, 6]  # seconds
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                job_id = data.get("id")
                if not job_id:
                    raise RunPodError(f"No job ID in response: {data}")
                logger.info(f"RunPod job submitted: {job_id}")
                return job_id
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                # Retry on 5xx server errors
                if status_code >= 500 and attempt < max_retries:
                    delay = backoff_delays[attempt]
                    logger.warning(f"RunPod server error {status_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                    last_error = e
                    continue
                raise RunPodError(f"RunPod API error: {status_code} - {e.response.text}")
            except httpx.TimeoutException as e:
                # Retry on timeouts
                if attempt < max_retries:
                    delay = backoff_delays[attempt]
                    logger.warning(f"RunPod timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                    last_error = e
                    continue
                raise RunPodError(f"RunPod request timed out after {max_retries + 1} attempts")
            except Exception as e:
                # Retry on connection errors
                if attempt < max_retries:
                    delay = backoff_delays[attempt]
                    logger.warning(f"RunPod error: {e}, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                    last_error = e
                    continue
                raise RunPodError(f"RunPod request failed: {e}")

        raise RunPodError(f"RunPod request failed after {max_retries + 1} attempts: {last_error}")

    async def poll_job(self, job_id: str) -> dict:
        """
        Poll a job until completion.

        Args:
            job_id: The job ID to poll.

        Returns:
            Complete response JSON when status=COMPLETED.

        Raises:
            RunPodError: If job fails or times out.
        """
        client = await self._get_client()
        url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"

        start_time = asyncio.get_event_loop().time()
        max_time = start_time + self.timeout

        while True:
            if asyncio.get_event_loop().time() > max_time:
                raise RunPodError(f"Job {job_id} timed out after {self.timeout}s")

            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                status = data.get("status")

                if status == "COMPLETED":
                    logger.info(f"RunPod job {job_id} completed")
                    return data
                elif status in ("FAILED", "CANCELLED"):
                    error = data.get("error", "Unknown error")
                    raise RunPodError(f"Job {job_id} {status}: {error}")
                elif status in ("IN_QUEUE", "IN_PROGRESS"):
                    await asyncio.sleep(self.poll_interval)
                else:
                    logger.warning(f"Unknown job status: {status}")
                    await asyncio.sleep(self.poll_interval)

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                # Transient 5xx errors - continue polling (don't mark as error)
                if status_code >= 500:
                    logger.warning(f"Poll transient error {status_code} for job {job_id}, will retry")
                    await asyncio.sleep(self.poll_interval)
                    continue
                # 4xx errors are permanent failures
                raise RunPodError(f"Poll failed: {status_code} - {e.response.text}")
            except RunPodError:
                raise
            except httpx.TimeoutException:
                # Transient timeout - continue polling
                logger.warning(f"Poll timeout for job {job_id}, will retry")
                await asyncio.sleep(self.poll_interval)
                continue
            except Exception as e:
                # Other transient errors (connection, etc) - continue polling
                logger.warning(f"Poll error for job {job_id} (will retry): {e}")
                await asyncio.sleep(self.poll_interval)

    def extract_text(self, completed_json: dict) -> str:
        """
        Extract generated text from completed job response.

        Expected path: output[0].choices[0].tokens (array of strings)

        Args:
            completed_json: The complete job response.

        Returns:
            Joined text string.

        Raises:
            RunPodError: If expected path not found.
        """
        try:
            output = completed_json.get("output")
            if not output or not isinstance(output, list) or len(output) == 0:
                raise RunPodError(f"Missing output array. Keys: {list(completed_json.keys())}")

            first_output = output[0]
            if not isinstance(first_output, dict):
                raise RunPodError(f"output[0] is not dict: {type(first_output)}")

            choices = first_output.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                raise RunPodError(f"Missing choices. Keys in output[0]: {list(first_output.keys())}")

            tokens = choices[0].get("tokens")
            if tokens is None:
                raise RunPodError(f"Missing tokens. Keys in choices[0]: {list(choices[0].keys())}")

            if isinstance(tokens, list):
                return "".join(str(t) for t in tokens)
            elif isinstance(tokens, str):
                return tokens
            else:
                raise RunPodError(f"Unexpected tokens type: {type(tokens)}")

        except RunPodError:
            raise
        except Exception as e:
            # Log structure for debugging (without full content)
            logger.error(f"extract_text failed: {e}. Response keys: {list(completed_json.keys())}")
            raise RunPodError(f"Failed to extract text: {e}")

    def extract_usage(self, completed_json: dict) -> tuple[int, int]:
        """
        Extract token usage from completed job.

        Returns:
            Tuple of (tokens_in, tokens_out).
        """
        try:
            output = completed_json.get("output", [{}])
            if output and isinstance(output, list) and len(output) > 0:
                usage = output[0].get("usage", {})
                return (
                    int(usage.get("input", 0)),
                    int(usage.get("output", 0)),
                )
        except Exception:
            pass
        return (0, 0)

    def extract_metadata(self, completed_json: dict) -> dict:
        """
        Extract execution metadata from completed job.

        Returns:
            Dict with delayTime, executionTime, workerId.
        """
        return {
            "delay_ms": int(completed_json.get("delayTime", 0)),
            "exec_ms": int(completed_json.get("executionTime", 0)),
            "worker_id": completed_json.get("workerId", ""),
        }

    async def generate(self, prompt: str) -> RunPodJobResult:
        """
        Complete flow: submit job, poll, extract result.

        Args:
            prompt: The prompt to send.

        Returns:
            RunPodJobResult with all extracted data.
        """
        job_id = await self.run_job(prompt)
        completed = await self.poll_job(job_id)
        text = self.extract_text(completed)
        tokens_in, tokens_out = self.extract_usage(completed)
        meta = self.extract_metadata(completed)

        return RunPodJobResult(
            job_id=job_id,
            status="ok",
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            delay_ms=meta["delay_ms"],
            exec_ms=meta["exec_ms"],
            worker_id=meta["worker_id"],
            raw_output=completed,
        )
