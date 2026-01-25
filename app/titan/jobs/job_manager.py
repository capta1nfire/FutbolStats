"""TITAN Job Manager - Idempotency and Dead Letter Queue handling.

Responsibilities:
1. Check idempotency before extraction (avoid duplicate work)
2. Save successful extractions to titan.raw_extractions
3. Send failed extractions to titan.job_dlq
4. Manage DLQ retry logic with exponential backoff
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID


def _utc_now() -> datetime:
    """Get current UTC timestamp (timezone-aware) for TIMESTAMPTZ compatibility."""
    return datetime.now(timezone.utc)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.titan.extractors.base import ExtractionResult
from app.titan.config import get_titan_settings

logger = logging.getLogger(__name__)
titan_settings = get_titan_settings()


class TitanJobManager:
    """Manages TITAN job lifecycle: idempotency, persistence, and DLQ.

    Usage:
        manager = TitanJobManager(session)

        # Check if already extracted
        if await manager.check_idempotency(idempotency_key):
            return  # Skip duplicate

        # Extract data
        result = await extractor.extract(...)

        # Save result
        if result.is_success:
            await manager.save_extraction(result)
        else:
            await manager.send_to_dlq(result)
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.schema = titan_settings.TITAN_SCHEMA

    async def check_idempotency(self, idempotency_key: str) -> bool:
        """Check if extraction already exists.

        Args:
            idempotency_key: 32-char SHA256 key

        Returns:
            True if extraction already exists (skip), False if new (proceed)
        """
        query = text(f"""
            SELECT 1 FROM {self.schema}.raw_extractions
            WHERE idempotency_key = :key
            LIMIT 1
        """)
        result = await self.session.execute(query, {"key": idempotency_key})
        return result.scalar() is not None

    async def save_extraction(self, result: ExtractionResult) -> int:
        """Save successful extraction to raw_extractions.

        Args:
            result: ExtractionResult from extractor

        Returns:
            extraction_id of inserted row

        Raises:
            IntegrityError: If idempotency_key already exists (race condition)
        """
        query = text(f"""
            INSERT INTO {self.schema}.raw_extractions (
                source_id, job_id, url, endpoint, params_hash, date_bucket,
                response_type, response_body, http_status, response_time_ms,
                captured_at, idempotency_key
            ) VALUES (
                :source_id, :job_id, :url, :endpoint, :params_hash, :date_bucket,
                :response_type, :response_body::jsonb, :http_status, :response_time_ms,
                :captured_at, :idempotency_key
            )
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING extraction_id
        """)

        import json
        result_dict = result.to_dict()
        # Convert response_body to JSON string for JSONB
        if result_dict["response_body"] is not None:
            result_dict["response_body"] = json.dumps(result_dict["response_body"])

        db_result = await self.session.execute(query, result_dict)
        await self.session.commit()

        row = db_result.fetchone()
        if row:
            return row[0]
        else:
            # ON CONFLICT triggered - already exists
            logger.info(f"Extraction already exists: {result.idempotency_key}")
            return 0

    async def send_to_dlq(
        self,
        result: ExtractionResult,
        max_attempts: Optional[int] = None,
    ) -> int:
        """Send failed extraction to Dead Letter Queue.

        If entry already exists (same idempotency_key), updates attempts counter.

        Args:
            result: ExtractionResult with error info
            max_attempts: Override default max attempts

        Returns:
            dlq_id of inserted/updated row
        """
        if max_attempts is None:
            max_attempts = titan_settings.TITAN_DLQ_MAX_ATTEMPTS

        # Calculate next retry time with exponential backoff
        now = _utc_now()
        base_seconds = titan_settings.TITAN_DLQ_RETRY_BASE_SECONDS
        max_seconds = titan_settings.TITAN_DLQ_RETRY_MAX_SECONDS

        # Check if entry already exists
        existing = await self._get_dlq_entry(result.idempotency_key)

        if existing:
            # Update existing entry
            attempts = existing["attempts"] + 1
            delay_seconds = min(base_seconds * (2 ** (attempts - 1)), max_seconds)
            next_retry = now + timedelta(seconds=delay_seconds) if attempts < max_attempts else None

            query = text(f"""
                UPDATE {self.schema}.job_dlq
                SET attempts = :attempts,
                    last_attempt = :now,
                    next_retry_at = :next_retry,
                    error_message = :error_message,
                    http_status = :http_status
                WHERE idempotency_key = :key
                RETURNING dlq_id
            """)

            db_result = await self.session.execute(query, {
                "attempts": attempts,
                "now": now,
                "next_retry": next_retry,
                "error_message": result.error_message,
                "http_status": result.http_status or None,
                "key": result.idempotency_key,
            })
            await self.session.commit()
            return db_result.fetchone()[0]

        else:
            # Insert new entry
            delay_seconds = base_seconds
            next_retry = now + timedelta(seconds=delay_seconds)

            import json
            # Store full replay info: endpoint, url, params_hash, date_bucket
            # This allows DLQ replay to reconstruct the exact request
            params_json = json.dumps({
                "url": result.url,
                "endpoint": result.endpoint,
                "params_hash": result.params_hash,
                "date_bucket": result.date_bucket.isoformat() if result.date_bucket else None,
            })

            query = text(f"""
                INSERT INTO {self.schema}.job_dlq (
                    job_id, source_id, idempotency_key, error_type, error_message,
                    http_status, attempts, max_attempts, endpoint, params, date_bucket,
                    first_attempt, last_attempt, next_retry_at
                ) VALUES (
                    :job_id, :source_id, :idempotency_key, :error_type, :error_message,
                    :http_status, 1, :max_attempts, :endpoint, :params::jsonb, :date_bucket,
                    :now, :now, :next_retry
                )
                RETURNING dlq_id
            """)

            db_result = await self.session.execute(query, {
                "job_id": result.job_id,
                "source_id": result.source_id,
                "idempotency_key": result.idempotency_key,
                "error_type": result.error_type or "unknown",
                "error_message": result.error_message,
                "http_status": result.http_status or None,
                "max_attempts": max_attempts,
                "endpoint": result.endpoint,
                "params": params_json,
                "date_bucket": result.date_bucket,
                "now": now,
                "next_retry": next_retry,
            })
            await self.session.commit()
            return db_result.fetchone()[0]

    async def _get_dlq_entry(self, idempotency_key: str) -> Optional[dict]:
        """Get existing DLQ entry by idempotency key."""
        query = text(f"""
            SELECT dlq_id, attempts, first_attempt, resolved_at
            FROM {self.schema}.job_dlq
            WHERE idempotency_key = :key
        """)
        result = await self.session.execute(query, {"key": idempotency_key})
        row = result.fetchone()
        if row:
            return {
                "dlq_id": row[0],
                "attempts": row[1],
                "first_attempt": row[2],
                "resolved_at": row[3],
            }
        return None

    async def get_pending_retries(
        self,
        source_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get DLQ entries ready for retry.

        Args:
            source_id: Filter by source (optional)
            limit: Max entries to return

        Returns:
            List of DLQ entries with retry info
        """
        now = _utc_now()
        params = {"now": now, "limit": limit}

        source_filter = ""
        if source_id:
            source_filter = "AND source_id = :source_id"
            params["source_id"] = source_id

        query = text(f"""
            SELECT dlq_id, job_id, source_id, idempotency_key, error_type,
                   endpoint, params, date_bucket, attempts, max_attempts
            FROM {self.schema}.job_dlq
            WHERE resolved_at IS NULL
              AND next_retry_at IS NOT NULL
              AND next_retry_at <= :now
              {source_filter}
            ORDER BY next_retry_at
            LIMIT :limit
        """)

        result = await self.session.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "dlq_id": row[0],
                "job_id": row[1],
                "source_id": row[2],
                "idempotency_key": row[3],
                "error_type": row[4],
                "endpoint": row[5],
                "params": row[6],
                "date_bucket": row[7],
                "attempts": row[8],
                "max_attempts": row[9],
            }
            for row in rows
        ]

    async def resolve_dlq_entry(
        self,
        dlq_id: int,
        resolution: str,
        resolved_by: str = "system",
    ) -> None:
        """Mark DLQ entry as resolved.

        Args:
            dlq_id: DLQ entry ID
            resolution: Resolution type ('retried_success', 'manual_skip', etc.)
            resolved_by: Who/what resolved it
        """
        query = text(f"""
            UPDATE {self.schema}.job_dlq
            SET resolved_at = :now,
                resolution = :resolution,
                resolved_by = :resolved_by
            WHERE dlq_id = :dlq_id
        """)

        await self.session.execute(query, {
            "now": _utc_now(),
            "resolution": resolution,
            "resolved_by": resolved_by,
            "dlq_id": dlq_id,
        })
        await self.session.commit()

    async def get_dlq_stats(self) -> dict:
        """Get DLQ statistics for dashboard.

        Returns:
            Dict with pending counts, error types, oldest entry, etc.
        """
        query = text(f"""
            SELECT
                COUNT(*) FILTER (WHERE resolved_at IS NULL) as pending_count,
                COUNT(*) FILTER (WHERE resolved_at IS NOT NULL) as resolved_count,
                COUNT(*) FILTER (WHERE resolved_at IS NULL AND next_retry_at IS NULL) as exhausted_count,
                MIN(created_at) FILTER (WHERE resolved_at IS NULL) as oldest_pending,
                COUNT(DISTINCT source_id) FILTER (WHERE resolved_at IS NULL) as sources_affected
            FROM {self.schema}.job_dlq
        """)

        result = await self.session.execute(query)
        row = result.fetchone()

        # Get error type breakdown for pending
        error_query = text(f"""
            SELECT error_type, COUNT(*) as count
            FROM {self.schema}.job_dlq
            WHERE resolved_at IS NULL
            GROUP BY error_type
            ORDER BY count DESC
        """)
        error_result = await self.session.execute(error_query)
        error_breakdown = {row[0]: row[1] for row in error_result.fetchall()}

        return {
            "pending_count": row[0] or 0,
            "resolved_count": row[1] or 0,
            "exhausted_count": row[2] or 0,
            "oldest_pending": row[3].isoformat() if row[3] else None,
            "sources_affected": row[4] or 0,
            "error_breakdown": error_breakdown,
        }
