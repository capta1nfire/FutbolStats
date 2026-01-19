# 🚀 Bitácora de Desarrollo & Ideas

> "Documento vivo para capturar ideas sin interrumpir el flujo actual."

## 🧠 Brain Dump (Zona de Aterrizaje)
*Escribe aquí cualquier idea rápida, bug o mejora tal como llegue a tu mente.*

## ⏳ Pendiente
- [x] Dashboard: agregar links **visibles** a JSON por sección (Ops/PIT/History/Logs) y aclarar qué JSON se está viendo.

- [x] PIT report: agregar bloque `interpretation` (fix escala `skill_vs_market`, guard-rail `low_n_predictions` con umbral `<30` explícito). Commits: baf71e3, 81a403d

- [x] PIT: Brier vs market vs uniform + ROI/EV con CI (cuando N lo permita) expuesto para auditoría (daily + weekly).

- [x] Dashboard: "Progreso hacia re-test/Alpha" con snapshots históricos en DB (AlphaProgressSnapshot + endpoints + job diario 09:10 UTC) + baseline #0 capturado. Commits: 69683f9, 4498d73

- [x] Auditoría con "Composer" (P0 ImportError weekly_pit_report + P1 anti-leakage `<` + hardening `_save_pit_report_to_db`) aplicada. Commit: 7014f29

- [x] Actualizar el codigo fuente en la App de IOS para que refleje los nuevos cambios en el codigo.
  - ValueBet fields hechos opcionales para manejar datos incompletos del backend
  - APIClient: agregado caso `emptyResponse` para respuestas vacías
  - Token hardcodeado removido de AppConfiguration.swift
  - scheduler.py normalizado para generar value_bets en formato consistente con engine.py
  - Legacy value_bets (6) permanecen; UI muestra "—" en campos faltantes; no rompe
  - Script de migración creado (scripts/migrate_legacy_value_bets.py) - opcional, requiere acceso a producción

- [ ] Monitorear narrativa y comparar con un review de un humano (analista deportivo o periodista) para validar/reforzar prescicion de nuestra narrativa.

- [ ] crear un checklist y mapa (ayer datos iban a pasar por fuera de "ML Engine")

- [ ] verificar que cada equipo de cada liga esta recibiendo predicciones, odds, estadisticas, etc.

- [ ] Reestructura xgboots y features basado en analisis de Google (pineado)

- [ ] Por que desaparecen los ppartidos de la App? El sabado 10 de enero inicialmente aparecieron 35, luego aparecieron 34 y hoy solo aparecen 8? Se estan filtrando o eliminando?


- [ ] Pandas vs Polars (por el momento no)

- [ ] Mejorar narrativa

- [x] Para el cierre formal (operacional) solo faltan los 2 pendientes y ya:
Screenshot de 1 alerta "Firing" (prueba controlada bajando temporalmente el umbral y luego revertir).
Confirmación de estabilidad 24h (screenshot del panel de series/requests mostrando que no hubo spikes).
**CLOSED 2026-01-14**: Informe 24h generado en docs/MONITORING_REPORT_20260113.md - 0 alertas, sistema estable.

- [x] Supervision de telemetria de las ultimas 24 horas
**CLOSED 2026-01-14**: Ver docs/MONITORING_REPORT_20260113.md - LLM 6/6 ok, 0 rejected, DQ sin anomalias.

- [ ] Revisar los payloads de las narrativas que ya estan persistiendo: "A partir de ahora, cuando una narrativa “desmejore”, el flujo correcto es:
abrir /dashboard/ops/llm_audit/{match_id}.json?token=...,
revisar llm_prompt_input_json + llm_validation_errors,
y ahí sabremos en 1 minuto si fue data incompleta o salida no soportada del modelo."

- [x] Perfecto. Con esos dos checks, ya no tengo observaciones: está bien instrumentado y auditable.
Para cerrar el caso, el criterio es exactamente el que pusiste:
llm_narrative_status=ok para audit 99,
llm_prompt_input_json.stats_summary poblado correctamente,
y si hubo normalización, queda en llm_validation_errors.
Cuando veas el próximo tick y esos campos, lo marcamos CLOSED. (esto es respecto a las narrativas LLMs).
**CLOSED 2026-01-14**: Audits 98-103 verificados con llm_narrative_status=ok. Ver docs/MONITORING_REPORT_20260113.md

- [x] P0: iOS "Bookie" null por desync odds_snapshots → matches.odds_*
  - RCA: odds_snapshots se populaba pero matches.odds_* quedaba NULL
  - Fix: write-through en scheduler.py + backfill script
  - Backfill ejecutado: 8 matches actualizados
  - Commit: 0b4b29d
  **CLOSED 2026-01-14**

- [x] Estaba pensando a futuro en como enriquecer la narrativa de qwem, habia pensado en que la narrativa aportara datos estadisticos puntuales basado en promedios generales de cada liga de cada pais.
  **IMPLEMENTED 2026-01-14**: League aggregates system created.
  - Tablas: `league_season_baselines` (promedios liga) + `league_team_profiles` (perfil equipo)
  - Métricas P0: goals_avg, over_X_pct, btts, clean_sheet, corners, cards
  - Métricas P0.3: ranks (best_attack, worst_defense, goal_difference)
  - Métricas P1: by_time goals (0-15min, 76-90+min)
  - Integrado en derived_facts: league_context, team_context, relative_context
  - Job de refresh: `app/aggregates/refresh_job.py`
  - Verificación: `scripts/verify_aggregates.py`
  - Ref: `docs/API_FOOTBALL_PAYLOAD_INVENTORY.md`

- [ ] Gol de camerino (goles en los primeros minutos)

- [ ]configurar alerta desde grafana en lugar de email en el codigo.

- [ ] Mensaje "Apuesta con responsabilidad en el footer de la narrativa me gusta", podemos manejar mas mensajes bajo este concepto.

- [ ]Revisar tabla de posiciones de inicio de ligas de sudamerica, deben volver a "Cero".
Monitorear (Ajustes aplicados:

Last eval + next run: Last eval: — | Next: every 30m
Errors 24h: Contador agregado en ops.json, solo se muestra en card si > 0
Labels claros: Pending → Awaiting FT para indicar que esperan partidos terminados, no que faltan generar
El deploy se propagará automáticamente. El card ahora distingue claramente:

"Awaiting FT: 1576" = predicciones shadow generadas, esperando que los partidos terminen
"Evaluated: 0" = partidos ya terminados y evaluados
"Last eval: —" = aún no ha corrido evaluación (vs job roto que mostraría timestamp antiguo)))

- [ ] El formato Copy se llama "triple backticks"

- [ ] Quiero cards con promedio de acierto de modelo A y modelo B.

- [ ] Auditoria a shadow + model B

- [ ] Se estan cargando equipos de la liga pero por la copa del rey y no por la liga, agregar esos torneos tambien.

- [ ] OJO LIGA COLOMBIANA! Para poblar manualmente cuando API-Football publique 2026:
curl "https://web-production-f2de9.up.railway.app/standings/239?season=2026"

- [x] Actualizar la equidad por internacional de bogota

- [ ] El P1 diferido (unificar en _LEAGUE_RULES) está bien planteado: lo activaría únicamente cuando crezca el número de ligas con reglas especiales o cuando haya más de un consumidor de esas reglas.

- [ ] Monitorear aleatoriamente logs en el backend.

- [ ] OK, todos los partidos NS de las próximas 48h sí tienen predicciones. El health check reporta WARN porque la columna created_at de esas predicciones es antigua (hace 19h), no porque falten predicciones.

El catch-up hizo upsert y encontró que ya existían predicciones, así que el created_at no se actualizó (es el timestamp original de cuando se crearon).

Diagnóstico:

Las predicciones existen y están correctas (Coverage 100%)
Missing FT: 0/22 - ningún partido terminado sin predicción
El WARN es porque el timestamp de la última predicción guardada es antiguo
El sistema está funcionando correctamente, pero la métrica de "staleness" es confusa porque:

Mira MAX(created_at) de la tabla predictions
Si el catch-up hace upsert de predicciones existentes, created_at no cambia
Reporta WARN aunque las predicciones están al día
Esto es un falso positivo del health check. Las predicciones están actualizadas, solo que el timestamp de creación es el original.

¿Quieres que ajuste la lógica del health check para considerar este caso, o prefieres dejarlo así ya que el próximo daily_save_predictions (a las 7:00 UTC) actualizará naturalmente las métricas?

- [ ] Betting Odds formato (+185 vs 2.54)

- [ ] medir predicciones con empates ajustado desde el dia 15 enero hora 5:50pm hora LA.

- [ ] monitoria en dashboard de todos los jobs

- [ ] configurar alertas en grafana

- [ ] Revisar claude.md para validar los pendientes.

- [ ] Supervisar Shadow Mode (2 semanas a partir de Jan 14/25)


--- Intención Sensor B: re-entrenar periódicamente con una ventana deslizante (p.ej. últimos N partidos FT) para ajustarse a la tendencia reciente; con ese ajuste, generar predicciones internas “hacia adelante” y, cuando esos partidos terminen, comparar Modelo A vs Realidad vs Modelo B (head-to-head + calibración).

tabla de posiciones aun sigue La Equidad.

- [ ] Perfecto. Resumen del estado actual:

ML Leagues Onboarding: COMPLETO

Liga	Volumen	Odds	Stats	Valor ML
Championship (40)	37 NS	32%	Pending	Alto (volumen)
Eredivisie (88)	3K+	47%	100%	Alto (mejor candidato)
Primeira Liga (94)	3K+	7%	100%	Medio (sin market)
Conference (848)	2K	0%	100%	Bajo (sin market)
Monitoreo 48h establecido:

jobs_health (stats_backfill, odds_sync)
Sentry (league_id 40/88/94)
Coverage por liga
Criterio backfill P1: Solo si FT sin stats o coverage persistentemente 0%.

- [ ] P1 (más adelante)
Cuando tengamos aire, migramos a un esquema más limpio (DB-only o S3/GCS). Pero hoy el objetivo es que no se caiga el job.
GO.

- [ ] Cuando tengamos una liga con tabla de posicion en fallback, notificarlo para seguimiento.

eliminar medallas "Tier" (y codigo) de oro, plata bronce en las predicciones de cada partido o revisar el codigo e implementarlo bien.

