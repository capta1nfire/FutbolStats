#!/usr/bin/env python3
"""Extract public API routes from main.py (Step 5 - Final).

Reads main.py, extracts:
- Block A: lines 250-1270 (caches, constants, standings/prediction helpers)
- Block B: lines 1361-1620 (_train_model_background, _warmup_standings_cache, _predictions_catchup_on_startup)
- Block C: lines 1662-5098 (Pydantic models + 39 endpoints)

Applies @app.X → @router.X transformation.
Writes app/routes/api.py.
"""

import re

MAIN_PY = "app/main.py"
OUTPUT = "app/routes/api.py"

# 1-indexed inclusive boundaries
BLOCK_A_START = 250
BLOCK_A_END = 1270
BLOCK_B_START = 1361
BLOCK_B_END = 1620
BLOCK_C_START = 1662
BLOCK_C_END = None  # to end of file

HEADER = '''"""Public API endpoints — predictions, matches, standings, teams, ETL, recalibration.

39 endpoints under various paths (/, /predictions/*, /matches/*, /standings/*,
/teams/*, /etl/*, /model/*, /odds/*, /audit/*, /recalibration/*, /lineup/*).
Auth: mix of verify_api_key (protected) and public (no auth).
Extracted from main.py Step 5 (final extraction).
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import bindparam, func, select, text, column
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, get_async_session, get_pool_status
from app.etl import APIFootballProvider, ETLPipeline
from app.etl.competitions import ALL_LEAGUE_IDS, COMPETITIONS
from app.etl.sota_constants import SOFASCORE_SUPPORTED_LEAGUES, UNDERSTAT_SUPPORTED_LEAGUES
from app.features import FeatureEngineer
from app.ml.persistence import load_active_model, persist_model_snapshot
from app.models import (
    JobRun, Match, OddsHistory, OpsAlert, PITReport, PostMatchAudit,
    Prediction, PredictionOutcome, SensorPrediction, ShadowPrediction,
    Team, TeamAdjustment, TeamOverride,
)
from app.teams.overrides import preload_team_overrides, resolve_team_display
from app.scheduler import get_last_sync_time, get_sync_leagues, SYNC_LEAGUES, global_sync_window
from app.security import limiter, verify_api_key, verify_api_key_or_ops_session
from app.state import ml_engine, _telemetry, _incr, _live_summary_cache
from app.utils.standings import (
    select_standings_view, StandingsGroupNotFound, apply_zones,
    group_standings_by_name, select_default_standings_group,
    classify_group_type,
)

router = APIRouter(tags=["api"])

logger = logging.getLogger(__name__)
settings = get_settings()

'''


def extract():
    with open(MAIN_PY, "r") as f:
        lines = f.readlines()

    total = len(lines)
    block_c_end = BLOCK_C_END or total

    # Extract blocks (convert to 0-indexed)
    block_a = lines[BLOCK_A_START - 1 : BLOCK_A_END]
    block_b = lines[BLOCK_B_START - 1 : BLOCK_B_END]
    block_c = lines[BLOCK_C_START - 1 : block_c_end]

    combined = block_a + ["\n\n"] + block_b + ["\n\n"] + block_c

    # Apply transformations
    result = []
    for line in combined:
        # @app.method → @router.method
        line = re.sub(r"@app\.(get|post|put|patch|delete)\(", r"@router.\1(", line)

        # Auth alias replacements (main.py uses aliases from security)
        line = line.replace("_verify_dashboard_token(request)", "verify_dashboard_token_bool(request)")
        line = line.replace("_verify_debug_token(request)", "verify_debug_token(request)")

        result.append(line)

    # Build final content
    content = HEADER + "".join(result)

    # Strip trailing whitespace, ensure single newline
    content = content.rstrip() + "\n"

    with open(OUTPUT, "w") as f:
        f.write(content)

    print(f"Wrote {OUTPUT} ({len(content)} bytes, {content.count(chr(10))} lines)")

    # Count endpoints
    router_count = content.count("@router.")
    print(f"  @router.* decorators: {router_count}")

    # Check for remaining @app. references
    app_refs = [
        (i + 1, l.rstrip())
        for i, l in enumerate(content.split("\n"))
        if "@app." in l and not l.strip().startswith("#") and not l.strip().startswith('"')
    ]
    if app_refs:
        print(f"  WARNING: {len(app_refs)} remaining @app. references:")
        for num, l in app_refs[:10]:
            print(f"    L{num}: {l}")
    else:
        print("  No @app. references remaining (OK)")

    # Check for from app.main at top level (P0-11)
    top_level_main_imports = []
    in_function = False
    for i, l in enumerate(content.split("\n")):
        stripped = l.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            in_function = True
        elif not l.startswith(" ") and not l.startswith("\t") and stripped and not stripped.startswith("#") and not stripped.startswith("@"):
            if not stripped.startswith("def ") and not stripped.startswith("async def ") and not stripped.startswith("class "):
                in_function = False

        if "from app.main import" in l and not in_function:
            top_level_main_imports.append((i + 1, l.rstrip()))

    if top_level_main_imports:
        print(f"  P0-11 VIOLATION: {len(top_level_main_imports)} top-level 'from app.main import':")
        for num, l in top_level_main_imports:
            print(f"    L{num}: {l}")
    else:
        print("  P0-11: No top-level 'from app.main import' (OK)")


if __name__ == "__main__":
    extract()
