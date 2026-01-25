"""TITAN Storage module - R2/S3-compatible object storage."""

from app.titan.storage.r2_client import R2Client, get_r2_client

__all__ = ["R2Client", "get_r2_client"]
