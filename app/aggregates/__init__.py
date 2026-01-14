"""
League aggregates module for narrative context.

Provides league baselines and team profiles for derived_facts.
"""

from app.aggregates.service import (
    AggregatesService,
    compute_league_baseline,
    compute_team_profiles,
    get_league_context,
    get_team_context,
)
from app.aggregates.refresh_job import (
    refresh_all_aggregates,
    refresh_single_league,
    get_aggregates_status,
)

__all__ = [
    "AggregatesService",
    "compute_league_baseline",
    "compute_team_profiles",
    "get_league_context",
    "get_team_context",
    "refresh_all_aggregates",
    "refresh_single_league",
    "get_aggregates_status",
]
