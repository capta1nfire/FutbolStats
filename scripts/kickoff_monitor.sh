#!/bin/bash
set -euo pipefail
# Monitor de Kickoff - Captura timestamp exacto NS → 1H
# Uso: ./scripts/kickoff_monitor.sh <MATCH_ID_INTERNO>

MATCH_ID="${1:-73238}"

# IMPORTANT: Never hardcode secrets in scripts committed to git.
# Provide these via environment variables instead:
#   FUTBOLSTATS_X_API_KEY=... FUTBOLSTATS_BASE_URL=... ./scripts/kickoff_monitor.sh 123
API_KEY="${FUTBOLSTATS_X_API_KEY:-}"
BASE_URL="${FUTBOLSTATS_BASE_URL:-https://web-production-f2de9.up.railway.app}"

if [ -z "$API_KEY" ]; then
  echo "ERROR: FUTBOLSTATS_X_API_KEY is not set (required to call /live-summary)" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="${ROOT_DIR}/logs/kickoff_${MATCH_ID}_$(date -u +%Y%m%d_%H%M%S).json"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  KICKOFF MONITOR v2 - Match ID: $MATCH_ID                  ║"
echo "║  Output: $OUTPUT_FILE"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Inicializar archivo de log
echo "{" > "$OUTPUT_FILE"
echo "  \"match_id\": $MATCH_ID," >> "$OUTPUT_FILE"
echo "  \"monitor_start\": \"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)\"," >> "$OUTPUT_FILE"
echo "  \"polls\": [" >> "$OUTPUT_FILE"

FIRST_POLL=true
KICKOFF_DETECTED=false
PREV_STATUS="NS"
POLL_COUNT=0
MAX_POLLS=720  # 1 hora a 5 segundos por poll

while [ $POLL_COUNT -lt $MAX_POLLS ]; do
    POLL_COUNT=$((POLL_COUNT + 1))
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)

    # Fetch live-summary
    RESPONSE=$(curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/live-summary")

    # Extraer datos del partido usando el ID interno
    MATCH_DATA=$(echo "$RESPONSE" | jq -r ".matches[\"$MATCH_ID\"] // null")

    if [ "$MATCH_DATA" != "null" ] && [ -n "$MATCH_DATA" ]; then
        STATUS=$(echo "$MATCH_DATA" | jq -r '.s')
        ELAPSED=$(echo "$MATCH_DATA" | jq -r '.e')
        ELAPSED_EXTRA=$(echo "$MATCH_DATA" | jq -r '.ex')
        HOME_GOALS=$(echo "$MATCH_DATA" | jq -r '.h')
        AWAY_GOALS=$(echo "$MATCH_DATA" | jq -r '.a')
        EVENTS=$(echo "$MATCH_DATA" | jq -c '.ev // []')

        # Detectar transición NS → 1H (kickoff)
        if [ "$PREV_STATUS" = "NS" ] && [ "$STATUS" = "1H" ]; then
            KICKOFF_DETECTED=true
            echo ""
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║  🎯 KICKOFF DETECTED!                                      ║"
            echo "╠════════════════════════════════════════════════════════════╣"
            echo "║  Timestamp captura: $TIMESTAMP"
            echo "║  Status: $STATUS"
            echo "║  Elapsed: $ELAPSED"
            echo "║  Elapsed Extra: $ELAPSED_EXTRA"
            echo "╚════════════════════════════════════════════════════════════╝"
        fi

        # Log entry
        if [ "$FIRST_POLL" = true ]; then
            FIRST_POLL=false
        else
            echo "," >> "$OUTPUT_FILE"
        fi

        echo "    {\"ts\": \"$TIMESTAMP\", \"poll\": $POLL_COUNT, \"status\": \"$STATUS\", \"elapsed\": $ELAPSED, \"elapsed_extra\": $ELAPSED_EXTRA, \"home\": $HOME_GOALS, \"away\": $AWAY_GOALS, \"events\": $EVENTS}" >> "$OUTPUT_FILE"

        # Console output
        echo "[$TIMESTAMP] Poll #$POLL_COUNT | Status: $STATUS | Elapsed: ${ELAPSED}'+${ELAPSED_EXTRA} | Score: $HOME_GOALS-$AWAY_GOALS"

        PREV_STATUS=$STATUS

        # Terminar si el partido terminó
        if [ "$STATUS" = "FT" ] || [ "$STATUS" = "AET" ] || [ "$STATUS" = "PEN" ]; then
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "  Partido finalizado. Status: $STATUS"
            echo "═══════════════════════════════════════════════════════════"
            break
        fi
    else
        # Partido aún no en live-summary (NS)
        if [ "$FIRST_POLL" = true ]; then
            FIRST_POLL=false
        else
            echo "," >> "$OUTPUT_FILE"
        fi
        echo "    {\"ts\": \"$TIMESTAMP\", \"poll\": $POLL_COUNT, \"status\": \"NS\", \"in_live_summary\": false}" >> "$OUTPUT_FILE"
        echo "[$TIMESTAMP] Poll #$POLL_COUNT | Status: NS (esperando kickoff...)"
    fi

    sleep 5
done

# Cerrar JSON
echo "" >> "$OUTPUT_FILE"
echo "  ]," >> "$OUTPUT_FILE"
echo "  \"monitor_end\": \"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)\"," >> "$OUTPUT_FILE"
echo "  \"total_polls\": $POLL_COUNT," >> "$OUTPUT_FILE"
echo "  \"kickoff_detected\": $KICKOFF_DETECTED" >> "$OUTPUT_FILE"
echo "}" >> "$OUTPUT_FILE"

echo ""
echo "Monitor finalizado. Log guardado en: $OUTPUT_FILE"
