"""Alerting module for FutbolStats."""

from app.alerting.email import send_alert_email, AlertType

__all__ = ["send_alert_email", "AlertType"]
