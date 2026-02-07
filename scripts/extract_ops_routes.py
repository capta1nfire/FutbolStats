#!/usr/bin/env python3
"""Extract ops_routes.py from main.py (Step 4a).

Reads main.py, extracts Block A (lines 6975-10762) and Block B (lines 13602-16689),
applies mechanical transformations, and writes app/dashboard/ops_routes.py.

Transformations:
- @app.get/post/patch/put/delete → @router.get/post/patch/put/delete
- _verify_dashboard_token(request) → verify_dashboard_token_bool(request)
- _verify_debug_token(request) → verify_debug_token(request)
- _has_valid_session(request) → _has_valid_ops_session(request)
- Depends(_verify_dashboard_token) → Depends(verify_dashboard_token_bool)
- Remove standalone 'from pathlib import Path' / 'from datetime import datetime as dt'
- dt.utcnow() → datetime.utcnow()
"""

import re

MAIN_PY = "app/main.py"
OUTPUT = "app/dashboard/ops_routes.py"

# 1-indexed inclusive boundaries
BLOCK_A_START = 6975
BLOCK_A_END = 10762
BLOCK_B_START = 13602
BLOCK_B_END = 16689

HEADER = '''"""Ops Dashboard API — debug, triggers, login, alerts, incidents.

46 endpoints across /dashboard/ops/*, /ops/*, /debug/*, /dashboard/incidents*.
Auth patterns: dashboard_token, debug_token, alerts_webhook, Depends, public+rate-limit.
Extracted from main.py Step 4a.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import AsyncSessionLocal, get_async_session
from app.models import OpsAlert
from app.security import (
    limiter,
    verify_dashboard_token_bool,
    verify_debug_token,
    _has_valid_ops_session,
)
from app.state import ml_engine

router = APIRouter(tags=["ops"])

logger = logging.getLogger(__name__)
settings = get_settings()

# Constants for ops log defaults (read from env, same as main.py log buffer)
OPS_LOG_DEFAULT_LIMIT = int(os.environ.get("OPS_LOG_DEFAULT_LIMIT", "200"))
OPS_LOG_DEFAULT_SINCE_MINUTES = int(os.environ.get("OPS_LOG_DEFAULT_SINCE_MINUTES", "1440"))  # 24h


def _check_token(request: Request):
    """Verify dashboard token (header + session + query param dev)."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")


'''


def extract():
    with open(MAIN_PY, "r") as f:
        lines = f.readlines()

    # Extract blocks (convert to 0-indexed)
    block_a = lines[BLOCK_A_START - 1 : BLOCK_A_END]
    block_b = lines[BLOCK_B_START - 1 : BLOCK_B_END]

    combined = block_a + ["\n\n"] + block_b

    # Apply transformations
    result = []
    for line in combined:
        # @app.method → @router.method
        line = re.sub(r"@app\.(get|post|put|patch|delete)\(", r"@router.\1(", line)

        # Auth alias replacements
        line = line.replace("_verify_dashboard_token(request)", "verify_dashboard_token_bool(request)")
        line = line.replace("_verify_debug_token(request)", "verify_debug_token(request)")
        line = line.replace("_has_valid_session(request)", "_has_valid_ops_session(request)")
        line = line.replace("Depends(_verify_dashboard_token)", "Depends(verify_dashboard_token_bool)")

        # Remove standalone import lines that are already in header
        stripped = line.strip()
        if stripped == "from pathlib import Path":
            continue
        if stripped == "from datetime import datetime as dt":
            continue

        # dt.utcnow() → datetime.utcnow()
        line = line.replace("dt.utcnow()", "datetime.utcnow()")

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

    # Check for remaining @app. references (should be zero)
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
    for alias in ["_verify_dashboard_token", "_verify_debug_token", "_has_valid_session"]:
        # Only flag if used as a function call (not in comments/strings)
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
    indent_stack = 0
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
