#!/usr/bin/env python3
"""Migrate simple and params-aware caches in dashboard_views_routes.py to SimpleCache.

Targets (11 caches):
- Simple: _pit_dashboard_cache, _feature_coverage_cache, _rollup_cache,
  _upcoming_matches_cache, _analytics_reports_cache, _audit_logs_cache, _team_logos_cache
- Params-aware: _sentry_issues_cache, _missing_preds_cache, _movement_recent_cache, _movement_top_cache

Skips (keyed/extended/dead): _matches_table_cache, _jobs_table_cache,
  _data_quality_cache, _data_quality_detail_cache, _ml_health_cache, _dashboard_predictions_cache
"""

import re

FILE = "app/dashboard/dashboard_views_routes.py"


def migrate():
    with open(FILE, "r") as f:
        content = f.read()

    # 1. Add import
    content = content.replace(
        "from app.security import verify_dashboard_token_bool",
        "from app.security import verify_dashboard_token_bool\nfrom app.utils.cache import SimpleCache",
    )

    # 2. Replace simple cache definitions
    simple_replacements = [
        # _pit_dashboard_cache (lines 46-50) - complex structure, skip for now
        # _feature_coverage_cache (line 305-309)
        ('_feature_coverage_cache: dict = {"data": None, "timestamp": 0, "ttl": 1800}',
         '_feature_coverage_cache = SimpleCache(ttl=1800)'),
        # _rollup_cache (line 987)
        ('_rollup_cache: dict = {"data": None, "timestamp": 0, "ttl": 60}',
         '_rollup_cache = SimpleCache(ttl=60)'),
        # _sentry_issues_cache (line 1058) - params-aware
        ('_sentry_issues_cache: dict = {"data": None, "timestamp": 0, "ttl": 90, "params": None}',
         '_sentry_issues_cache = SimpleCache(ttl=90)'),
        # _missing_preds_cache (line 1214) - params-aware
        ('_missing_preds_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}',
         '_missing_preds_cache = SimpleCache(ttl=60)'),
        # _movement_recent_cache (line 1366) - params-aware
        ('_movement_recent_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}',
         '_movement_recent_cache = SimpleCache(ttl=60)'),
        # _movement_top_cache (line 1509) - params-aware
        ('_movement_top_cache: dict = {"data": None, "timestamp": 0, "ttl": 60, "params": None}',
         '_movement_top_cache = SimpleCache(ttl=60)'),
        # _dashboard_predictions_cache (line 3652) - dead code but convert anyway
        ('_dashboard_predictions_cache: dict = {"data": None, "timestamp": 0, "ttl": 45}',
         '_dashboard_predictions_cache = SimpleCache(ttl=45)'),
        # _analytics_reports_cache (line 4040)
        ('_analytics_reports_cache: dict = {"data": None, "timestamp": 0, "ttl": 120}',
         '_analytics_reports_cache = SimpleCache(ttl=120)'),
        # _audit_logs_cache (line 4401) - TTL in constant
        ('_audit_logs_cache: dict = {"data": None, "timestamp": 0}',
         '_audit_logs_cache = SimpleCache(ttl=90)  # was _AUDIT_LOGS_TTL'),
        # _team_logos_cache (line 4683) - TTL in constant
        ('_team_logos_cache: dict = {"data": None, "timestamp": 0}',
         '_team_logos_cache = SimpleCache(ttl=3600)  # was _TEAM_LOGOS_TTL'),
    ]

    for old, new in simple_replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  Replaced definition: {new[:60]}...")
        else:
            print(f"  WARNING: Not found: {old[:60]}...")

    with open(FILE, "w") as f:
        f.write(content)

    print(f"\nDone. Definitions replaced.")
    print("NOTE: Cache access patterns (read/write) must be updated manually.")


if __name__ == "__main__":
    migrate()
