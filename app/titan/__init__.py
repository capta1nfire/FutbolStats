"""
TITAN OMNISCIENCE Module
========================
Enterprise scraping infrastructure with PIT (Point-in-Time) compliance.

Key components:
- extractors/: API wrappers that capture timestamps for PIT
- jobs/: Job manager with idempotency and DLQ
- materializers/: Feature matrix builders with insertion policy
- dashboard.py: Operational status for /dashboard/titan.json

Schema: titan.* (isolated from public.*)
Tables: raw_extractions, job_dlq, feature_matrix
"""

from app.titan.config import TitanSettings, get_titan_settings

__all__ = [
    "TitanSettings",
    "get_titan_settings",
]
