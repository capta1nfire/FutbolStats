#!/usr/bin/env python3
"""Extract dashboard_views_routes.py from main.py (Step 4b).

Reads main.py, extracts the 4b sections:
- Part 1: lines 5100-6976 (PIT/TITAN/V2/Feature Coverage/predictions triggers)
- Part 2: lines 6981-9817 (upcoming matches, matches, jobs, data_quality, etc.)

Applies mechanical transformations and writes app/dashboard/dashboard_views_routes.py.
"""

import re

MAIN_PY = "app/main.py"
OUTPUT = "app/dashboard/dashboard_views_routes.py"

# 1-indexed inclusive boundaries
PART1_START = 5100
PART1_END = 6976
PART2_START = 6981
PART2_END = 9817

HEADER = '''"""Dashboard Views API — PIT, TITAN, feature coverage, tables, predictions, analytics.

23 endpoints under /dashboard/* (various paths, no single prefix).
All protected by dashboard token auth.
Extracted from main.py Step 4b.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, get_async_session
from app.security import verify_dashboard_token_bool
from app.state import ml_engine

router = APIRouter(tags=["dashboard-views"])

logger = logging.getLogger(__name__)
settings = get_settings()


def _check_token(request: Request):
    """Verify dashboard token (header + session + query param dev)."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")


'''


def extract():
    with open(MAIN_PY, "r") as f:
        lines = f.readlines()

    # Extract parts (convert to 0-indexed)
    part1 = lines[PART1_START - 1 : PART1_END]
    part2 = lines[PART2_START - 1 : PART2_END]

    combined = part1 + ["\n\n"] + part2

    # Apply transformations
    result = []
    for line in combined:
        # @app.method → @router.method
        line = re.sub(r"@app\.(get|post|put|patch|delete)\(", r"@router.\1(", line)

        # Auth alias replacements
        line = line.replace("_verify_dashboard_token(request)", "verify_dashboard_token_bool(request)")
        line = line.replace("_verify_debug_token(request)", "verify_debug_token(request)")

        result.append(line)

    # Build final content
    content = HEADER + "".join(result)

    # Strip trailing whitespace from end of file, ensure single newline
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
        if "@app." in l
    ]
    if app_refs:
        print(f"  WARNING: {len(app_refs)} remaining @app. references:")
        for num, l in app_refs:
            print(f"    L{num}: {l}")
    else:
        print("  No @app. references remaining (OK)")

    # Check for remaining alias references
    for alias in ["_verify_dashboard_token", "_verify_debug_token"]:
        refs = [
            (i + 1, l.rstrip())
            for i, l in enumerate(content.split("\n"))
            if alias + "(" in l and not l.strip().startswith("#") and not l.strip().startswith('"')
        ]
        if refs:
            print(f"  WARNING: {len(refs)} remaining {alias}( references:")
            for num, l in refs:
                print(f"    L{num}: {l}")

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
