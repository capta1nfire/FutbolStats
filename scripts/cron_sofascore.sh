#!/bin/bash
# Cron wrapper for sofascore_stealth_sync.py
# Loads .env and runs with the specified mode, logging output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

MODE="${1:---all}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/sofascore_${MODE#--}_${TIMESTAMP}.log"

# Cron has minimal PATH — add homebrew + user bins
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Load environment
set -a
source "$PROJECT_DIR/.env"
set +a

export PYTHONPATH="$PROJECT_DIR"

echo "=== Sofascore Stealth Sync ($MODE) — $(date) ===" >> "$LOG_FILE"
/opt/homebrew/opt/python@3.12/libexec/bin/python3 "$SCRIPT_DIR/sofascore_stealth_sync.py" "$MODE" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE — $(date) ===" >> "$LOG_FILE"

# Keep only last 50 log files
ls -t "$LOG_DIR"/sofascore_*.log 2>/dev/null | tail -n +51 | xargs rm -f 2>/dev/null || true

exit $EXIT_CODE
