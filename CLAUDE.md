# FutbolStats

Sistema de predicciones de fútbol con ML y narrativas LLM.

## Roles del Proyecto
- **David**: Owner/Usuario del producto
- **ABE** (Auditor Backend): Director técnico del backend. No escribe código.
- **ADB** (Auditor Dashboard): Director técnico del dashboard. No escribe código.
- **Master**: Codificador backend - ejecuta código, deploys, debugging según instrucciones de ABE
- **Claude**: Codificador dashboard - implementa UI según instrucciones de ADB

> **Para auditores nuevos**: Ver `docs/AUDITOR_ONBOARDING.md` para contexto completo, roles, flujo de comunicación y estado del proyecto.

## Stack Tecnológico
- **Backend**: Python 3.12, FastAPI, SQLAlchemy/SQLModel
- **Database**: PostgreSQL (Railway)
- **ML**: XGBoost (predicciones 1X2)
- **LLM**: Gemini 2.0 Flash (narrativas post-partido)
- **iOS**: Swift/SwiftUI
- **Infra**: Railway (deploy), Prometheus/Grafana (métricas), Sentry (errores)

## Estructura del Proyecto
```
app/
├── main.py              # FastAPI monolito (~8000 líneas)
├── scheduler.py         # APScheduler jobs (sync, backfill, fastpath)
├── database.py          # Async SQLAlchemy engine
├── config.py            # Settings (pydantic)
├── ml/                  # Modelo ML y predicciones
├── llm/                 # Narrativas con Gemini
├── etl/                 # ETL desde API-Football
└── telemetry/           # Prometheus metrics, Sentry

models/                  # ML artifacts (commiteados, ej: xgb_v1.0.0_*.json)

ios/FutbolStats/
├── Services/            # API client, LiveScoreManager, MatchCache
├── ViewModels/          # PredictionsViewModel
└── Views/               # SwiftUI views

docs/                    # Protocolos operacionales
scripts/                 # Scripts de evaluación y utilidades
```

## Herramientas Disponibles

### MCP Servers

| Server | Descripción | Uso |
|--------|-------------|-----|
| `railway-postgres` | Queries read-only a PostgreSQL | `mcp__railway-postgres__query` con SQL |

### Commands (`.cursor/commands/`)
Invocables con `/nombre` en chat.

| Command | Descripción |
|---------|-------------|
| `/ops` | Estado operacional del sistema (jobs, shadow, sentry, budget) |
| `/logs [filtro]` | Logs de Railway con filtro opcional |
| `/match <equipo>` | Buscar partido por nombre de equipo |
| `/verify` | Smoke test post-deploy |
| `/model-sanity` | Verificar que el modelo ML carga y predice |
| `/db-report` | Reportes ad-hoc de base de datos |
| `/deploy` | Guía de deploy (manual only) |

### Skills (`.cursor/skills/`)
Capacidades especializadas que el agente puede invocar.

| Skill | Descripción | Modo |
|-------|-------------|------|
| `secrets-scan` | Escanear repo por secretos expuestos | Manual only |
| `api-contract` | Validar contrato iOS vs backend | Read-only |

### Subagents (`.cursor/skills/`)
Agentes especializados para tareas delegadas.

| Subagent | Descripción | Restricciones |
|----------|-------------|---------------|
| `ops-triage` | Diagnóstico operacional read-only | Sin edición, sin deploy |
| `db-analyst-ro` | Consultas DB solo SELECT | Sin INSERT/UPDATE/DELETE |

### Railway CLI
```bash
railway logs -n 50                    # Logs recientes
railway logs --filter "error"         # Solo errores
railway logs --filter "FASTPATH"      # Filtrar por componente
railway status                        # Estado del servicio
```

### Git
```bash
git push origin main                  # Deploy automático en Railway
```

## API Endpoints Principales

### Endpoints públicos (sin auth)
- `GET /health` - Health check

### Endpoints protegidos (requieren `X-API-Key`)
**CRÍTICO**: Estos endpoints NO son públicos. Requieren header `X-API-Key` válido.

| Endpoint | Descripción |
|----------|-------------|
| `GET /predictions/upcoming` | Predicciones próximos partidos |
| `GET /predictions/match/{id}` | Predicción específica |
| `GET /teams` | Lista de equipos |
| `GET /matches` | Lista de partidos |
| `GET /competitions` | Lista de competiciones |
| `GET /teams/{id}/history` | Historial de equipo |
| `GET /matches/{id}/details` | Detalles de partido |
| `GET /matches/{id}/insights` | Insights/narrativa de partido |
| `GET /matches/{id}/timeline` | Timeline de partido |
| `GET /matches/{id}/odds-history` | Historial de cuotas |
| `GET /matches/{id}/lineup` | Alineaciones |
| `GET /standings/{league_id}` | Tabla de posiciones |
| `GET /live-summary` | Partidos en vivo |

### Endpoints de dashboard (requieren `X-Dashboard-Token`)
- `GET /dashboard/ops.json` - Dashboard operacional
- `GET /dashboard/pit.json` - Dashboard PIT

## Protocolos Operacionales
Consultar `docs/` antes de operaciones críticas:

| Protocolo | Archivo |
|-----------|---------|
| Agregar liga/copa | `docs/COMPETITION_ONBOARDING.md` |
| Troubleshooting | `docs/OPS_RUNBOOK.md` |
| Alertas Grafana | `docs/GRAFANA_ALERTS_CHECKLIST.md` |
| Evaluar modelo | `docs/PIT_EVALUATION_PROTOCOL.md` |
| **SOTA Pendientes** | `docs/SOTA_PENDIENTES.md` |

## Convenciones
- **Timestamps**: Siempre UTC (`datetime.utcnow()`)
- **Commits**: Conventional commits (`feat:`, `fix:`, `docs:`)
- **Deploy**: Push a `main` = deploy automático en Railway

## Arquitectura ML (Resumen)
- **Modelo activo**: XGBoost v1.0.0 (14 features)
- **Shadow mode**: Two-stage (en evaluación, NO en producción)
- **Sensor B**: Diagnóstico de calibración (no afecta predicciones)

Para detalles técnicos de ML, shadow mode y sensor B, ver `docs/ML_ARCHITECTURE.md`.

## Jobs del Scheduler
| Job | Frecuencia | Función |
|-----|------------|---------|
| `global_sync` | 1 min | Sync partidos desde API-Football |
| `live_tick` | 10 seg | Actualizar partidos en vivo |
| `stats_backfill` | 60 min | Capturar stats de partidos FT |
| `odds_sync` | 6 horas | Sync odds para partidos próximos |
| `fastpath` | 2 min | Generar narrativas LLM |
