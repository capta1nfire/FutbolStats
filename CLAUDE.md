# FutbolStats

Sistema de predicciones de fútbol con ML y narrativas LLM.

## Roles del Proyecto
- **David**: Owner/Usuario del producto
- **Auditor**: Director de Ingeniería - valida arquitectura, define requisitos/guardrails/criterios de aceptación, audita riesgos. No escribe código de producción.
- **Claude (Master)**: Implementador - ejecuta código, deploys, debugging según instrucciones del Auditor

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

### MCP Server: PostgreSQL
Queries directas a la base de datos con lenguaje natural.
Ejemplo: "Busca los últimos partidos de América de Cali"

### Railway CLI
Acceso a logs, deployments y estado del backend.
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
- `GET /health` - Health check
- `GET /predictions/upcoming` - Predicciones próximos partidos
- `GET /predictions/match/{id}` - Predicción específica
- `GET /dashboard/ops.json` - Dashboard operacional (requiere token)
- `GET /live-summary` - Partidos en vivo (requiere API key)

## Protocolos Operacionales
Consultar `docs/` antes de operaciones críticas:

| Protocolo | Archivo |
|-----------|---------|
| Agregar liga/copa | `docs/COMPETITION_ONBOARDING.md` |
| Troubleshooting | `docs/OPS_RUNBOOK.md` |
| Alertas Grafana | `docs/GRAFANA_ALERTS_CHECKLIST.md` |
| Evaluar modelo | `docs/PIT_EVALUATION_PROTOCOL.md` |

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
