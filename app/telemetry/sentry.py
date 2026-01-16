"""
Sentry integration for error tracking and performance monitoring.

Provides:
- Automatic exception capture with stacktrace
- FastAPI request context
- SQLAlchemy query errors
- Scheduler job context tagging

Security:
- All sensitive headers are scrubbed before sending
- Query strings with tokens are redacted
- Request bodies are NOT captured
- PII is disabled by default
"""

import os
import re
import logging
from typing import Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Module-level flag to track initialization
_sentry_initialized = False


def scrub_sensitive_data(event: dict, hint: dict) -> Optional[dict]:
    """
    Scrub sensitive data from Sentry events before sending.

    Removes:
    - API keys and tokens from headers
    - Token parameters from query strings
    - Authorization headers
    - Cookies
    """
    try:
        request = event.get("request") or {}

        # Scrub sensitive headers
        headers = request.get("headers") or {}
        sensitive_headers = [
            "x-api-key",
            "x-dashboard-token",
            "authorization",
            "cookie",
            "set-cookie",
            "x-forwarded-for",  # Privacy
        ]
        for header in sensitive_headers:
            if header in headers:
                headers[header] = "[REDACTED]"

        # Also check case-insensitive (headers can vary)
        headers_lower = {k.lower(): k for k in headers.keys()}
        for sensitive in sensitive_headers:
            if sensitive in headers_lower:
                original_key = headers_lower[sensitive]
                headers[original_key] = "[REDACTED]"

        request["headers"] = headers

        # Scrub query string parameters
        query_string = request.get("query_string")
        if isinstance(query_string, str) and query_string:
            # Redact common sensitive params
            query_string = re.sub(
                r'(?i)(token|api_key|dashboard_token|key|secret|password)=([^&]*)',
                r'\1=[REDACTED]',
                query_string
            )
            request["query_string"] = query_string

        # Remove request body entirely (extra safety)
        if "data" in request:
            request["data"] = "[SCRUBBED]"

        event["request"] = request

    except Exception as e:
        # Never fail scrubbing - just log and continue
        logger.warning(f"Sentry scrubbing error (continuing): {e}")

    return event


def init_sentry() -> bool:
    """
    Initialize Sentry SDK if SENTRY_DSN is configured.

    Returns True if Sentry was initialized, False otherwise.

    Environment variables:
    - SENTRY_DSN: Required. Sentry DSN from project settings.
    - SENTRY_TRACES_SAMPLE_RATE: Optional. Default 0.05 (5%).
    - SENTRY_ENABLED: Optional. Set to 'false' to disable even with DSN.
    - RAILWAY_ENVIRONMENT: Used as Sentry environment tag.
    - RAILWAY_GIT_COMMIT_SHA: Used as release version.
    """
    global _sentry_initialized

    if _sentry_initialized:
        logger.debug("Sentry already initialized, skipping")
        return True

    # Check kill switch
    if os.getenv("SENTRY_ENABLED", "true").lower() == "false":
        logger.info("Sentry disabled via SENTRY_ENABLED=false")
        return False

    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        logger.info("Sentry not configured (SENTRY_DSN not set)")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError as e:
        logger.warning(f"Sentry SDK not installed: {e}")
        return False

    # Configuration
    environment = os.getenv("RAILWAY_ENVIRONMENT", "development")
    release = os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown")
    traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.05"))

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,

        # Integrations
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
            LoggingIntegration(
                level=logging.ERROR,        # Capture ERROR logs
                event_level=logging.ERROR,  # Create events for ERROR
            ),
        ],

        # Sampling - conservative to stay within free tier
        traces_sample_rate=traces_sample_rate,  # 5% of transactions
        profiles_sample_rate=0.0,               # Disable profiling

        # Privacy & Security
        send_default_pii=False,
        before_send=scrub_sensitive_data,

        # Performance
        enable_tracing=True,

        # Ignore common non-errors
        ignore_errors=[
            KeyboardInterrupt,
            SystemExit,
        ],
    )

    _sentry_initialized = True
    logger.info(
        f"Sentry initialized: env={environment}, release={release[:8] if release != 'unknown' else 'unknown'}, "
        f"traces_sample_rate={traces_sample_rate}"
    )

    # One-time smoke test (controlled via env var, auto-disables after first run)
    if os.getenv("SENTRY_SMOKE_TEST", "").lower() == "true":
        sentry_sdk.capture_message(
            "SENTRY_SMOKE_TEST: Integration verified",
            level="warning"
        )
        logger.info("Sentry smoke test sent. Set SENTRY_SMOKE_TEST=false to disable.")

    return True


def is_sentry_enabled() -> bool:
    """Check if Sentry is initialized and active."""
    return _sentry_initialized


@contextmanager
def sentry_job_context(job_id: str, **extra_tags):
    """
    Context manager for scheduler jobs that sets Sentry tags and captures exceptions.

    Usage:
        with sentry_job_context("stats_backfill", model_version="v1.0.0"):
            # job code here
            pass

    If an exception occurs, it will be captured with full context before re-raising.
    """
    if not _sentry_initialized:
        yield
        return

    import sentry_sdk

    with sentry_sdk.push_scope() as scope:
        # Set job context tags
        scope.set_tag("job_id", job_id)
        scope.set_context("job", {"job_id": job_id, **extra_tags})

        # Set any extra low-cardinality tags
        for key, value in extra_tags.items():
            if value is not None:
                scope.set_tag(key, str(value))

        try:
            yield scope
        except Exception as e:
            # Capture with full context before re-raising
            sentry_sdk.capture_exception(e)
            raise


def capture_message(message: str, level: str = "info", **extra):
    """
    Capture a message to Sentry (useful for important events that aren't exceptions).

    Args:
        message: The message to capture
        level: One of 'debug', 'info', 'warning', 'error', 'fatal'
        **extra: Additional context to attach
    """
    if not _sentry_initialized:
        return

    import sentry_sdk

    with sentry_sdk.push_scope() as scope:
        for key, value in extra.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_message(message, level=level)


def set_user_context(user_id: str = None, **extra):
    """
    Set user context for the current scope (useful for API requests).

    Note: We don't capture PII, so this is mainly for correlation.
    """
    if not _sentry_initialized:
        return

    import sentry_sdk

    user_data = {}
    if user_id:
        user_data["id"] = user_id
    user_data.update(extra)

    if user_data:
        sentry_sdk.set_user(user_data)


def capture_exception(exception: Exception, job_id: str = None, **extra_context):
    """
    Capture an exception to Sentry with optional job context.

    Use this in scheduler job except blocks for explicit capture with context.

    Args:
        exception: The exception to capture
        job_id: Optional job identifier for tagging
        **extra_context: Additional context to attach
    """
    if not _sentry_initialized:
        return

    import sentry_sdk

    with sentry_sdk.push_scope() as scope:
        if job_id:
            scope.set_tag("job_id", job_id)
        for key, value in extra_context.items():
            scope.set_extra(key, value)
        sentry_sdk.capture_exception(exception)
