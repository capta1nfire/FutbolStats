Ejecuta un smoke test de release (sin hacer cambios) y reporta si el sistema está OK.

## Config
- `API_BASE`: usar `FUTBOLSTATS_API_BASE` o pedirlo.
- `DASHBOARD_TOKEN`: `FUTBOLSTATS_DASHBOARD_TOKEN` (no imprimir completo).
- `API_KEY`: `FUTBOLSTATS_API_KEY` (no imprimir completo).

## Pasos (Bash)
1) Health:
   - `curl -s "$API_BASE/health" | jq`

2) Ops dashboard:
   - `curl -s -H "X-Dashboard-Token: $DASHBOARD_TOKEN" "$API_BASE/dashboard/ops.json" | jq`
   - Resumir: jobs OK (sin errores recientes), uptime/ts, flags relevantes (shadow_mode, sensor_b).

3) Endpoint de predicciones (autenticado):
   - `curl -s -H "X-API-Key: $API_KEY" "$API_BASE/predictions/upcoming" | jq`
   - Validar: responde 200, JSON parseable, lista no vacía (si aplica).

4) Señales de degradación (solo lectura):
   - Si algo falla, trae `railway logs -n 80 --filter "error"` y adjunta 10-20 líneas relevantes.

## Output (formato fijo)
- “**Checklist**”
  - health: OK/WARN/FAIL
  - ops dashboard: OK/WARN/FAIL
  - predictions/upcoming: OK/WARN/FAIL
  - logs: OK/WARN/FAIL (si se consultó)
- “**Detalle de fallos**” (solo si hay WARN/FAIL)
- “**Siguiente acción**” (1 recomendación concreta)
