#!/bin/bash
set -euo pipefail
# Monitor de Transiciones HT/2H - Captura detallada para Auditor
# Uso: ./scripts/transition_monitor.sh <MATCH_ID>

MATCH_ID="${1:-73238}"

# IMPORTANT: Never hardcode secrets in scripts committed to git.
# Provide these via environment variables instead:
#   FUTBOLSTATS_X_API_KEY=... FUTBOLSTATS_BASE_URL=... ./scripts/transition_monitor.sh 123
API_KEY="${FUTBOLSTATS_X_API_KEY:-}"
BASE_URL="${FUTBOLSTATS_BASE_URL:-https://web-production-f2de9.up.railway.app}"

if [ -z "$API_KEY" ]; then
  echo "ERROR: FUTBOLSTATS_X_API_KEY is not set (required to call /live-summary)" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="${ROOT_DIR}/logs/transitions_${MATCH_ID}_$(date -u +%Y%m%d_%H%M%S).log"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  TRANSITION MONITOR - Match ID: $MATCH_ID                  ║"
echo "║  Capturing: 1H→HT and HT→2H transitions                    ║"
echo "║  Output: $OUTPUT_FILE"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Initialize log
echo "=== TRANSITION MONITOR LOG ===" > "$OUTPUT_FILE"
echo "Match ID: $MATCH_ID" >> "$OUTPUT_FILE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

PREV_STATUS=""
PREV_ELAPSED=0
HT_CAPTURED=false
H2_CAPTURED=false
PRE_TRANSITION_BUFFER=()
POST_HT_COUNT=0
POST_2H_COUNT=0

while true; do
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Fetch live-summary
    RESPONSE=$(curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/live-summary")
    MATCH_DATA=$(echo "$RESPONSE" | jq -r ".matches[\"$MATCH_ID\"] // null")

    if [ "$MATCH_DATA" != "null" ] && [ -n "$MATCH_DATA" ]; then
        STATUS=$(echo "$MATCH_DATA" | jq -r '.s')
        ELAPSED=$(echo "$MATCH_DATA" | jq -r '.e')
        EXTRA=$(echo "$MATCH_DATA" | jq -r '.ex')
        HOME=$(echo "$MATCH_DATA" | jq -r '.h')
        AWAY=$(echo "$MATCH_DATA" | jq -r '.a')

        POLL_LINE="[$TIMESTAMP] Status=$STATUS | Elapsed=$ELAPSED'+$EXTRA | Score=$HOME-$AWAY"

        # Keep buffer of last 5 polls for pre-transition context
        PRE_TRANSITION_BUFFER+=("$POLL_LINE")
        if [ ${#PRE_TRANSITION_BUFFER[@]} -gt 5 ]; then
            PRE_TRANSITION_BUFFER=("${PRE_TRANSITION_BUFFER[@]:1}")
        fi

        # Detect 1H → HT transition
        if [ "$PREV_STATUS" = "1H" ] && [ "$STATUS" = "HT" ] && [ "$HT_CAPTURED" = false ]; then
            echo "" >> "$OUTPUT_FILE"
            echo "═══════════════════════════════════════════════════════════" >> "$OUTPUT_FILE"
            echo "  A) TRANSICIÓN 1H → HT DETECTADA" >> "$OUTPUT_FILE"
            echo "═══════════════════════════════════════════════════════════" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo "--- 5 polls ANTES del HT ---" >> "$OUTPUT_FILE"
            for line in "${PRE_TRANSITION_BUFFER[@]}"; do
                echo "$line" >> "$OUTPUT_FILE"
            done
            echo "" >> "$OUTPUT_FILE"
            echo "--- Transición detectada ---" >> "$OUTPUT_FILE"
            echo "$POLL_LINE  ← HT START" >> "$OUTPUT_FILE"
            HT_CAPTURED=true
            POST_HT_COUNT=0

            echo ""
            echo "🎯 HT DETECTED at $TIMESTAMP"
        fi

        # Capture 5 polls after HT
        if [ "$HT_CAPTURED" = true ] && [ "$POST_HT_COUNT" -lt 5 ] && [ "$STATUS" = "HT" ]; then
            POST_HT_COUNT=$((POST_HT_COUNT + 1))
            if [ "$POST_HT_COUNT" -le 5 ]; then
                echo "$POLL_LINE" >> "$OUTPUT_FILE"
            fi
            if [ "$POST_HT_COUNT" -eq 5 ]; then
                echo "" >> "$OUTPUT_FILE"
                echo "--- Fin captura post-HT ---" >> "$OUTPUT_FILE"
            fi
        fi

        # Detect HT → 2H transition
        if [ "$PREV_STATUS" = "HT" ] && [ "$STATUS" = "2H" ] && [ "$H2_CAPTURED" = false ]; then
            echo "" >> "$OUTPUT_FILE"
            echo "═══════════════════════════════════════════════════════════" >> "$OUTPUT_FILE"
            echo "  B) TRANSICIÓN HT → 2H DETECTADA" >> "$OUTPUT_FILE"
            echo "═══════════════════════════════════════════════════════════" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo "--- Primer poll 2H ---" >> "$OUTPUT_FILE"
            echo "$POLL_LINE  ← 2H START" >> "$OUTPUT_FILE"
            H2_CAPTURED=true
            POST_2H_COUNT=0

            echo ""
            echo "🎯 2H DETECTED at $TIMESTAMP (elapsed=$ELAPSED)"
        fi

        # Capture 3 polls after 2H start
        if [ "$H2_CAPTURED" = true ] && [ "$POST_2H_COUNT" -lt 3 ] && [ "$STATUS" = "2H" ]; then
            POST_2H_COUNT=$((POST_2H_COUNT + 1))
            if [ "$POST_2H_COUNT" -le 3 ]; then
                echo "$POLL_LINE" >> "$OUTPUT_FILE"
            fi
            if [ "$POST_2H_COUNT" -eq 3 ]; then
                echo "" >> "$OUTPUT_FILE"
                echo "--- Fin captura 2H (3 polls) ---" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"
                echo "Monitor completo. Transiciones capturadas." >> "$OUTPUT_FILE"
                echo ""
                echo "✅ MONITOR COMPLETO - Ambas transiciones capturadas"
                echo "Log guardado en: $OUTPUT_FILE"
                exit 0
            fi
        fi

        # Console output
        echo "$POLL_LINE"

        PREV_STATUS=$STATUS
        PREV_ELAPSED=$ELAPSED

        # Exit if match ended
        if [ "$STATUS" = "FT" ] || [ "$STATUS" = "AET" ] || [ "$STATUS" = "PEN" ]; then
            echo "" >> "$OUTPUT_FILE"
            echo "Partido finalizado: $STATUS" >> "$OUTPUT_FILE"
            echo ""
            echo "Partido finalizado: $STATUS"
            exit 0
        fi
    else
        echo "[$TIMESTAMP] Esperando partido en live-summary..."
    fi

    sleep 5
done
