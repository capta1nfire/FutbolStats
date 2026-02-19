"""
Email alerting module for FutbolStats.

Sends SMTP alerts with cooldown to prevent spam.
"""

import asyncio
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts that can be sent."""

    PREDICTIONS_HEALTH_WARN = "predictions_health_warn"
    PREDICTIONS_HEALTH_RED = "predictions_health_red"


# In-memory cooldown tracking (resets on restart, which is acceptable)
_last_alert_times: dict[AlertType, datetime] = {}


def _can_send_alert(alert_type: AlertType) -> bool:
    """Check if enough time has passed since last alert of this type."""
    settings = get_settings()
    cooldown = timedelta(minutes=settings.ALERT_COOLDOWN_MINUTES)

    last_sent = _last_alert_times.get(alert_type)
    if last_sent is None:
        return True

    return datetime.utcnow() - last_sent >= cooldown


def _record_alert_sent(alert_type: AlertType) -> None:
    """Record that an alert was sent."""
    _last_alert_times[alert_type] = datetime.utcnow()


def _build_predictions_health_email(
    status: str,
    hours_since_last: Optional[float],
    ns_next_48h: int,
    ns_missing: int,
    coverage_pct: float,
) -> tuple[str, str]:
    """Build email subject and body for predictions health alert."""
    severity = "WARNING" if status == "warn" else "CRITICAL"
    hours_str = f"{hours_since_last:.1f}h" if hours_since_last else "N/A"

    subject = f"[Bon Jogo] {severity}: Predictions Health Alert"

    body = f"""
Bon Jogo Predictions Health Alert
===================================

Status: {status.upper()}
Severity: {severity}

Metrics:
--------
- Hours since last prediction saved: {hours_str}
- NS matches next 48h: {ns_next_48h}
- NS matches missing predictions: {ns_missing}
- Coverage: {coverage_pct:.1f}%

Thresholds:
-----------
- WARN: >6h since last prediction AND >0 NS matches
- RED: >12h since last prediction AND >0 NS matches

Action Required:
----------------
Check the predictions scheduler job and RunPod/Gemini LLM availability.

---
This is an automated alert from Bon Jogo.
"""

    return subject, body


async def send_alert_email(
    alert_type: AlertType,
    status: str = "",
    hours_since_last: Optional[float] = None,
    ns_next_48h: int = 0,
    ns_missing: int = 0,
    coverage_pct: float = 0.0,
) -> bool:
    """
    Send an alert email if cooldown allows.

    Args:
        alert_type: Type of alert to send.
        status: Health status (warn/red).
        hours_since_last: Hours since last prediction saved.
        ns_next_48h: Number of NS matches in next 48h.
        ns_missing: NS matches missing predictions.
        coverage_pct: Prediction coverage percentage.

    Returns:
        True if email was sent, False if skipped (cooldown/disabled/error).
    """
    settings = get_settings()

    if not settings.SMTP_ENABLED:
        logger.debug("[ALERT] SMTP disabled, skipping email")
        return False

    if not _can_send_alert(alert_type):
        cooldown_mins = settings.ALERT_COOLDOWN_MINUTES
        logger.info(f"[ALERT] Skipped {alert_type.value}: cooldown active ({cooldown_mins}min)")
        return False

    # Build email based on alert type
    if alert_type in (AlertType.PREDICTIONS_HEALTH_WARN, AlertType.PREDICTIONS_HEALTH_RED):
        subject, body = _build_predictions_health_email(
            status, hours_since_last, ns_next_48h, ns_missing, coverage_pct
        )
    else:
        logger.error(f"[ALERT] Unknown alert type: {alert_type}")
        return False

    # Send email in thread pool to avoid blocking
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _send_smtp_email,
            settings.SMTP_HOST,
            settings.SMTP_PORT,
            settings.SMTP_USERNAME,
            settings.SMTP_PASSWORD,
            settings.SMTP_FROM_EMAIL,
            settings.SMTP_TO_EMAIL,
            subject,
            body,
        )

        _record_alert_sent(alert_type)
        logger.info(f"[ALERT] Email sent: {alert_type.value} to {settings.SMTP_TO_EMAIL}")
        return True

    except Exception as e:
        logger.error(f"[ALERT] Failed to send email: {e}")
        return False


def _send_smtp_email(
    host: str,
    port: int,
    username: str,
    password: str,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
) -> None:
    """Synchronous SMTP send (runs in executor)."""
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(username, password)
        server.sendmail(from_email, to_email, msg.as_string())
