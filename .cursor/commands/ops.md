Ejecuta un **diagnóstico operacional** rápido del backend (producción o el `API_BASE` que yo indique).

## Inputs
- Si el usuario escribió argumentos después de `/ops`, trátalos como `API_BASE` (ej: `/ops https://...`).
- Si no hay argumentos, usa `API_BASE` desde variable de entorno `FUTBOLSTATS_API_BASE`.
- Si tampoco existe, **pregunta** cuál `API_BASE` usar (no inventes URLs).

## Autenticación
- Para `GET /dashboard/ops.json` se requiere header `X-Dashboard-Token`.
- Busca el token en `FUTBOLSTATS_DASHBOARD_TOKEN` (env var).
- Si falta, **pide al usuario** el token. No lo imprimas completo en la salida (si lo necesitas para curl, úsalo pero no lo “echoees”).

## Pasos (Bash)
1) Health check:
   - `curl -s "$API_BASE/health" | jq` (si `jq` no está, imprime raw)

2) Ops dashboard (filtrado):
   - `curl -s -H "X-Dashboard-Token: $DASHBOARD_TOKEN" "$API_BASE/dashboard/ops.json" | jq '.data // . | {jobs, shadow_mode, sensor_b, incidents, errors, version, timestamp}'`
   - Si ese shape no existe, imprime `keys` y muestra un resumen razonable.

3) Evaluación de **staleness** de jobs (crítico):
   - **Regla #1 (preferida): usar `next_run` si existe.**
     - Si el payload trae `next` / `next_run` (timestamp UTC), calcula:
       - `lateness = now_utc - next_run` (en minutos)
     - Marca:
       - **OK** si `lateness <= 0` (todavía no vence / está “entre ciclos”)
       - **WARN** si `0 < lateness <= 30 min` (ligeramente atrasado)
       - **FAIL** si `lateness > 30 min` o si hay error explícito
     - Nota: con `next_run`, **NO uses** “edad/intervalo” para decidir WARN; eso genera falsos positivos.

   - **Regla #2 (fallback): si NO hay `next_run`, usa intervalos esperados.**
     - Asume frecuencias esperadas (UTC):
       - `global_sync`: 1 min
       - `live_tick`: 10 seg
       - `stats_backfill`: 60 min
       - `odds_sync`: 6 horas
       - `fastpath`: 2 min
     - Extrae “última ejecución” y calcula `age` (minutos).
     - Marca:
       - **OK** si \(age \le 1.5 \times interval\)
       - **WARN** si \(1.5 \times interval < age \le 4 \times interval\)
       - **FAIL** si \(age > 4 \times interval\) o si hay error explícito

   - Si el dashboard dice “OK” pero hay **FAIL** por lateness/age, prioriza FAIL (puede ser drift/atasco).

4) Logs rápidos (solo si el usuario lo pide o si detectas error explícito o **lateness > 0** o FAIL):
   - `railway logs -n 50`
   - y/o `railway logs -n 50 --filter "error"`
   - y/o `railway logs -n 50 --filter "FASTPATH"`
   - Si `stats_backfill` está WARN/FAIL:
     - `railway logs -n 120 --filter "stats_backfill"`
     - Luego **NO pegues todo**: muestra solo 10–20 líneas relevantes (las más recientes que contengan `error|exception|timeout|traceback|failed|retry|skipped|rate limit|429|502|503`).
   - Si `odds_sync` está WARN/FAIL:
     - `railway logs -n 120 --filter "odds_sync"`
     - Luego **NO pegues todo**: muestra solo 10–20 líneas relevantes (mismos keywords).
   - Si no hay líneas “relevantes”, muestra únicamente las últimas 15 líneas y di “no error keywords found”.
   - Cierra con “**Hipótesis de causa**” (1–3 bullets) basado en esas líneas.

## Salida
- Devuelve una **tabla resumen** (OK/WARN/FAIL) para: `health`, `scheduler/jobs`, `shadow_mode`, `sensor_b`, `incidents/errors`.
- Añade una tabla “**Jobs staleness**” con columnas: job, expected interval, last run age, status (OK/WARN/FAIL), note (si hay mismatch con dashboard).
- Si hubo WARN/FAIL en `stats_backfill` u `odds_sync`, añade sección “**Logs relevantes (snippet)**” + “**Hipótesis de causa**”.
- Luego una sección “**Acciones sugeridas**” con 3-5 bullets (si aplica).
