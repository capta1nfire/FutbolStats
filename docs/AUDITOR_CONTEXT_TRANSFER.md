# Transferencia de Contexto para Nuevo Auditor Técnico

**Fecha**: 2026-01-17
**Proyecto**: FutbolStats

---

## Definición de la Tríada de Trabajo (CRÍTICO)

### Identidad: Auditor Técnico (este agente)
- Rol: **Auditor Técnico / Director de Ingeniería**. Dirige estrategias, valida decisiones, audita lógica, define criterios de aceptación (AC), redacta prompts/instrucciones para ejecución.
- Restricción: **NO escribe código de producción**. No hace commits ni aplica parches. No ejecuta deploys. Todo cambio lo implementa **Master**.
- Responsabilidad: Mantener trazabilidad, evitar regressions, exigir guardrails (seguridad, PIT integrity, money-safe), y asegurar observabilidad (OPS/Sentry/Prometheus).

### David (Usuario)
- Rol: **Dueño del producto**. Visión, prioridades, aprobación final. Conocimientos básicos de desarrollo.
- Necesidad central: "**Si no lo veo, no sé si existe/funciona**" → cockpit/OPS Dashboard con herramientas de debug (Copy button), y UX "live" comparable a la competencia.

### Master (Agente Coder)
- Rol: **Único implementador** de código. Hace PRs, migraciones, cambios en backend/iOS, deploy a Railway.
- El Auditor le entrega: prompts con requisitos, guardrails, AC, queries de verificación y orden de prioridades.

---

## Estructura del Proyecto

### Archivos Clave
```
FutbolStats/
├── CLAUDE.md                    # Instrucciones globales para Claude (tokens, URLs, queries)
├── app/
│   ├── main.py                  # Monolito FastAPI (~8000 líneas)
│   ├── scheduler.py             # APScheduler jobs (stats_backfill, odds_sync, fastpath, live_tick)
│   ├── database.py              # SQLAlchemy async engine + session helpers
│   ├── config.py                # Settings via pydantic
│   ├── telemetry/
│   │   ├── metrics.py           # Prometheus counters/gauges/histograms
│   │   └── sentry.py            # Sentry SDK init
│   └── ml/                      # ML prediction logic
├── models/                      # ML artifacts (xgb_v1.0.0_*.json) - commiteados
├── docs/
│   ├── COMPETITION_ONBOARDING.md
│   ├── OPS_RUNBOOK.md
│   ├── GRAFANA_ALERTS_CHECKLIST.md
│   ├── PIT_EVALUATION_PROTOCOL.md
│   └── PROPOSAL_LIVE_SCORE_OPTIMIZATION.md
├── scripts/
│   └── evaluate_pit_live_only.py
└── ios/FutbolStats/FutbolStats/
    ├── FutbolStatsApp.swift     # Entry point, scenePhase handling
    ├── Services/
    │   ├── APIEnvironment.swift # Backend URL config
    │   ├── AppConfiguration.swift # API keys (Info.plist)
    │   ├── LiveScoreManager.swift # Live polling singleton
    │   ├── MatchCache.swift     # Local cache overlay
    │   └── ImageCache.swift
    ├── ViewModels/
    │   └── PredictionsViewModel.swift
    └── Views/
        ├── PredictionsListView.swift # Parrilla con LeagueCard
        └── MatchDetailView.swift     # Detalle partido
```

### Comandos de Verificación Rápida
```bash
# Health check producción
curl -s "https://web-production-f2de9.up.railway.app/health"

# OPS dashboard JSON
curl -s -H "X-Dashboard-Token: ops_c902abbbc239904c96f9ae37db4b882a" \
  "https://web-production-f2de9.up.railway.app/dashboard/ops.json" | jq '.data.jobs_health'

# Live summary (requiere API key)
curl -s -H "X-API-Key: <YOUR_API_KEY>" \
  "https://web-production-f2de9.up.railway.app/live-summary" | jq '.'

# Railway logs
railway logs -n 50
railway logs -n 30 --filter "FASTPATH"
```

---

## Estado Actual del Desarrollo

### Terminado / Operativo (✅)

#### Observabilidad y Operación
- **OPS Dashboard**: `/dashboard/ops.json` (JSON API, consumed by Next.js dashboard).
- **Jobs Health Monitoring (P0)**: stats_backfill, odds_sync, fastpath instrumentados con métricas Prometheus.
- **Sentry backend**: `sentry-sdk[fastapi]==1.40.0`, scrubbing de tokens, jobs instrumentados.

#### Data/ETL/Competitions
- Protocolo de onboarding: `docs/COMPETITION_ONBOARDING.md`.
- Competiciones activas: Copa del Rey (143), Championship (40), Eredivisie (88), Primeira Liga (94), Belgian Pro League (144), Saudi Pro League (307), Colombia Superliga (713).
- **Odds sync job**: Cada 6h, ventana 48h, freshness 6h, max fixtures.
- **Stats backfill**: SQL json bugs corregidos, jobs health evita fallos silenciosos.

#### ML/Predicciones
- Shadow Mode + Sensor B implementados (gating min_samples=50).
- Rerun de predicciones NS con tabla `prediction_reruns` y endpoints OPS.
- PIT evaluation protocol v2: ROI/EV con IC95%, resultado actual HOLD.

#### LLM Narratives
- Schema v3.2 normalización corregida, PROMPT_VERSION v1.7.
- Selección de predicción fijada a baseline MODEL_VERSION.

#### Team Identity Overrides
- "La Equidad" → "Internacional de Bogotá" implementado via `TeamOverride` + migración.

#### iOS Live UX
- **Live Score Optimization completado (2026-01-17)**:
  - Endpoint `/live-summary`: Auth X-API-Key, rate limit 60 req/min, cache L1 5s, cap 50 matches.
  - iOS `LiveScoreManager`: Gating (15s si hay live, 60s backoff), scenePhase-aware.
  - Métricas: `live_summary_requests_total`, `live_summary_latency_ms`, `live_summary_matches_count`.
- **UI estética (2026-01-17)**:
  - `LeagueCard` agrupa partidos por liga en una sola tarjeta glass con separadores.
  - `GlassCardModifier` para iOS 26+ con fallback a fondo oscuro.

### A medias / En monitoreo (🟡)
- **Shadow Mode**: eval count bajo por gating 50. No es bug, falta muestra.
- **Sensor B**: Reporta "LEARNING"; falta retrain para producir b_probs.
- **Value bets monetización**: HOLD; requiere N>=100/200 post-fix y mejorar skill_vs_market.

### Roto / Riesgos activos (🔴)
- Ninguno crítico confirmado.

---

## Stack Tecnológico

### Backend
- Python (FastAPI), APScheduler, PostgreSQL, SQLAlchemy/SQLModel.
- Prometheus `/metrics`, Grafana, Sentry.
- ETL con API-Football.
- Deploy: Railway (auto-deploy en push a main).

### iOS
- Swift/SwiftUI, polling controlado, cache overlay local.
- Target: iOS 17+ con features iOS 26 (glassEffect) con fallback.

### Objetivo Final
- Predicciones probabilísticas calibradas + motor monetizable de value bets.
- Gobernanza ML: canary/shadow/sensor, PIT integrity, ROI/EV con IC.
- Operabilidad: cockpit OPS con controles + audit log + debug pack.

---

## Decisiones Arquitectónicas Críticas (NO cambiar)

1. **No jobs por partido** para live → `live_tick` global + iOS gating + `/live-summary`.
2. **Short-polling vs WebSockets** → Short-polling por simplicidad/debuggability.
3. **Rerun/Two-stage rollout** → No promover MODEL_ARCHITECTURE; usar canary con `PREFER_RERUN_PREDICTIONS`.
4. **Team rebranding** → No reescribir histórico; overrides por `effective_from`.
5. **Competition onboarding** → Siempre seguir `docs/COMPETITION_ONBOARDING.md`.
6. **PIT protocol v2** → ROI/EV con IC95% es métrica primaria; GO requiere `IC95%_ROI_lower > 0`.

---

## Reglas de Negocio y Restricciones

- **API-Football**: Respetar rate limits (~30 req/min), budget ~7500/día.
- **Odds**: Captura via `odds_sync_upcoming` cada 6h. `live_tick` NO actualiza odds.
- **Stats**: Backfill post-FT; SQL json comparisons con `stats::text != '{}'`.
- **Live**: `/live-summary` requiere API key; iOS backoff si 0 live.
- **Security**: Nunca pegar tokens en chat; rotar si se expusieron.

---

## Deuda Técnica y Riesgos

1. **Value bet performance**: `skill_vs_market` negativo (~-12.7%). Necesita segmentación y N mayor.
2. **Shadow/Sensor gating**: Riesgo de "no se mueve" si no se comunica estado.
3. **Rate limiting por IP**: Carrier NAT puede causar falsos positivos; migrar a per-API-key si necesario.
4. **Narratives**: Si se sirve twostage, necesitará flag "served_model_version".

---

## Trabajo en Progreso (Sesión 2026-01-17)

### Completado hoy
1. **Live Score Optimization**: Endpoint `/live-summary` + iOS `LiveScoreManager` - DEPLOYED y verificado.
2. **UI iOS**: `LeagueCard` agrupa partidos por liga en tarjeta glass única con separadores Divider.

### Pendiente
- **Xcode target**: `LiveScoreManager.swift` necesita agregarse manualmente al target (error de compilación reportado).
- **API Key iOS**: Configurar en `Info.plist` para producción.

---

## Próximos Pasos Inmediatos

1. **PIT monetización**: Re-test semanal post-fix era (>=2026-01-13), segmentación por liga.
2. **Sensor B**: Confirmar que retrain corre y `evaluated_with_b` sube.
3. **Shadow evaluation**: Asegurar eval lag bajo y `total_evaluated` aumenta.
4. **Live-summary hardening**: Evaluar rate limit per API-key (P1).

---

## NOTAS ADICIONALES DEL USUARIO (David)

1. **Config iOS:** La API Key y Dashboard Token se configuran en `Info.plist` (keys `API_KEY` y `DASHBOARD_TOKEN`) o via UserDefaults para dev local. Ver `ios/FutbolStats/FutbolStats/Services/AppConfiguration.swift`. La URL del backend está en `APIEnvironment` (hardcoded por environment).

2. **Modelos ML:** Los binarios blessed residen en `models/` (ej. `xgb_v1.0.0_20260102.json`). Se commitean al repo para que Railway los tenga en deploy.

3. **Testing:** NO hay coverage formal de unit tests. Se valida manualmente con curls y verificación en OPS dashboard antes de considerar algo "deployed". Railway hace auto-deploy en push a main.

4. **Xcode Target Membership:** Archivos nuevos en iOS (ej. `LiveScoreManager.swift`) deben agregarse manualmente al target en Xcode - no basta con que existan en el filesystem.

---

*Documento generado 2026-01-17 para transferencia de contexto a nuevo Auditor.*
