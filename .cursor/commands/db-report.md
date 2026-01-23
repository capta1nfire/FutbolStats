Genera un reporte **read-only** (SQL SELECT) sobre un tema de negocio/ops y lo devuelve en tabla.

## Inputs
- Lo que venga después de `/db-report` es el tema (ej: `/db-report narrativas pendientes`).
- Si no hay tema, pregunta por uno.

## Fuente
- Preferido: usar el MCP de Postgres (Railway) si está disponible (`railway-postgres`).

## Guardrails (obligatorio)
- **Solo SELECT**. Prohibido: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `GRANT`, `REVOKE`.
- Si el usuario pide algo que implique mutación, rechaza y ofrece alternativa read-only.

## Temas sugeridos (no inventar campos: descubrir schema si hace falta)
- “predicciones recientes”: últimas 20 predicciones + match status/result si existe.
- “accuracy por liga (7d/14d)”: usando tablas de reportes si existen; si no, explicar limitación.
- “partidos sin odds”: NS próximos 48h con odds faltantes.
- “narrativas pendientes”: FT últimos 7d sin `llm_narrative_status=ok` en `post_match_audits` (si existe).
- “jobs health fallback”: últimas ejecuciones en `job_runs` por job_name.

## Procedimiento
1) Si no conoces las tablas/campos, primero corre queries a `information_schema` para ubicar:
   - `matches`, `predictions`, `post_match_audits`, `job_runs`, `odds_history` (si existen).
2) Genera 1–3 queries SELECT pequeñas (con límites).
3) Devuelve:
   - SQL usado
   - tabla markdown con resultados (top 20)
   - interpretación breve (2–4 bullets)
