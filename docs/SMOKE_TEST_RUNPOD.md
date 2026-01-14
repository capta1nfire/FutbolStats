# RunPod Smoke Test

Script para probar prompts/payloads directamente contra RunPod sin depender del backend.

## Requisitos

```bash
# Variable de entorno obligatoria
export RUNPOD_API_KEY="rpa_..."

# Opcional (default: a49n0iddpgsv7r)
export RUNPOD_ENDPOINT_ID="..."
```

## Uso Básico

```bash
# Test async (default) - más robusto
python scripts/runpod_smoke_test.py --payload logs/payloads/payload_70509.json

# Test sync - más rápido para pruebas individuales
python scripts/runpod_smoke_test.py --payload logs/payloads/payload_70509.json --sync

# Con parámetros custom
python scripts/runpod_smoke_test.py \
    --payload logs/payloads/payload_6648.json \
    --max-tokens 2048 \
    --temperature 0.5 \
    --top-p 0.95
```

## Payloads de Ejemplo

| Archivo | Match | Características |
|---------|-------|-----------------|
| `logs/payloads/payload_70509.json` | Cultural Leonesa vs Athletic Club | Red card, AET, penalties, bet_lost |
| `logs/payloads/payload_6648.json` | Dortmund vs Bremen | xG data, clean win, bet_won |

## Estructura del Payload

```json
{
  "match_id": 70509,
  "date": "2026-01-13T20:00:00",
  "home_team": "Cultural Leonesa",
  "away_team": "Athletic Club",
  "home_goals": 3,
  "away_goals": 4,
  "league_name": "Copa del Rey",
  "venue": {"city": "Leon", "name": "Estadio Reino de León"},
  "events": [...],
  "stats": {
    "home": {"total_shots": 5, "ball_possession": 59.0, "shots_on_goal": 3},
    "away": {"total_shots": 7, "ball_possession": 41.0, "shots_on_goal": 4}
  },
  "prediction": {
    "correct": false,
    "confidence": 0.5288,
    "probabilities": {"away": 0.2918, "draw": 0.1794, "home": 0.5288},
    "predicted_result": "home"
  },
  "market_odds": {}
}
```

**Nota**: El script genera `derived_facts` automáticamente si no está presente en el payload.

## Output

El script genera:
1. **Console output** - Resumen del test
2. **JSON file** - `logs/runpod_smoke_<match_id>_<timestamp>.json`

### Ejemplo de Output

```
============================================================
SMOKE TEST RESULTS
============================================================
Job ID:         sync-62237860-15da-4c53-be20-aff380df2e59-u1
Status:         COMPLETED
Delay Time:     37273 ms
Execution Time: 4608 ms
Client Elapsed: 42000 ms
Tokens In/Out:  1850 / 512

Validation:     OK
Claim Issues:   0

Title:          Victoria visitante en prórroga
Word Count:     185
Tone:           mitigate_loss

Body Preview:
  Athletic Club logró imponerse en una eliminatoria vibrante...
============================================================
```

## Validaciones Ejecutadas

El script ejecuta todas las validaciones del backend:

1. **Schema validation** - Estructura JSON correcta
2. **Claim validation** - Red cards, penalties, goal minutes
3. **Derived facts validation** - HT score, stats leaders, team attribution
4. **Control token sanitization** - Elimina tokens internos del body

## Exit Codes

| Code | Significado |
|------|-------------|
| 0 | Validation OK (o solo warnings) |
| 1 | Error (no se pudo generar/parsear) |
| 2 | Rejected (claim validation failed) |

## Crear Payloads Custom

### Desde la DB

```sql
-- Extraer payload existente
SELECT llm_prompt_input_json
FROM post_match_audits pma
JOIN prediction_outcomes po ON pma.outcome_id = po.id
WHERE po.match_id = {match_id}
ORDER BY pma.created_at DESC LIMIT 1;
```

### Manual

Crear un archivo JSON siguiendo la estructura del ejemplo. Campos mínimos requeridos:

- `match_id`
- `home_team`, `away_team`
- `home_goals`, `away_goals`
- `events` (puede estar vacío: `[]`)
- `stats.home`, `stats.away` (necesita `ball_possession`, `total_shots`, `shots_on_goal`)
- `prediction` con `correct`, `confidence`, `probabilities`, `predicted_result`

## Troubleshooting

### "RUNPOD_API_KEY not set"
```bash
export RUNPOD_API_KEY="rpa_YOUR_API_KEY_HERE"
```

### "Failed to parse JSON"
El LLM generó output malformado. Revisar `raw_output_preview` en el JSON de resultados.

### "No text in LLM response"
El worker puede estar frío. Reintentar o verificar health:
```bash
curl -s "https://api.runpod.ai/v2/a49n0iddpgsv7r/health" \
  -H "authorization: $RUNPOD_API_KEY"
```

### Validation "rejected"
Revisar `claim_errors` en el output. Posibles causas:
- Red card attribution incorrecta
- Stats leader contradiction
- Unsupported claims (eventos sin evidencia)

## Notas Técnicas

- **Auth header**: `authorization: <API_KEY>` (sin Bearer)
- **Prompt version**: v7 con derived_facts (PROMPT_VERSION en claim_validator.py)
- **Default timeout**: 120s para sync, 60 polls × 2s para async
