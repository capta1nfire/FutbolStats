# FutbolStats

Sistema de predicciones de fútbol con ML y narrativas LLM.

## Roles del Proyecto
- **David**: Owner/Usuario del producto
- **ABE** (Auditor Backend): Director técnico del backend. No escribe código.
- **ATI** (Auditor TITAN): Director técnico de TITAN Omniscience. No escribe código. Audita y dirige temas de TITAN y temas backend relacionados/inferidos a TITAN (ingesta, scraping, matching, aliases, PIT, feature_matrix).
- **ADB** (Auditor Dashboard): Director técnico del dashboard. No escribe código.
- **Master**: Codificador backend - ejecuta código, deploys, debugging según instrucciones de ABE **y/o ATI** (cuando el tema es TITAN o relacionado)
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
│   ├── steamchaser.py   # SteamChaser: line movement predictor (Phase 2, shadow mode)
│   └── policy.py        # Market Anchor + betting policy
├── events/              # Phase 2: Event-driven lineup cascade
│   ├── bus.py           # EventBus (asyncio.Queue) + Sweeper Queue (2min)
│   └── handlers.py      # cascade_handler (steel degradation, 5s timeout)
├── features/            # Feature engineering (PTS, VORP, talent delta, xi_continuity)
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

## Environment Setup

Todas las variables de entorno están en `.env` (raíz del proyecto, en `.gitignore`).

```bash
# Cargar variables antes de ejecutar scripts
source .env                              # O: set -a; source .env; set +a
python scripts/mi_script.py             # Ahora tiene DATABASE_URL, API keys, etc.
```

Variables disponibles en `.env`:
- `DATABASE_URL` — PostgreSQL sync (para psql, SQLAlchemy sync)
- `DATABASE_URL_ASYNC` — PostgreSQL async (asyncpg)
- `PGPASSWORD`, `PGHOST`, `PGPORT` — Para `psql` directo sin URL
- `API_FOOTBALL_KEY` / `RAPIDAPI_KEY` — API-Football
- `FUTBOLSTATS_API_KEY` — API key del backend
- `DASHBOARD_TOKEN` — Token para endpoints `/dashboard/*`
- `GEMINI_API_KEY` — Google Gemini
- `RUNPOD_API_KEY` — RunPod LLM
- `PYTHONPATH` — Apunta a raíz del proyecto

**NUNCA pasar credenciales inline** en comandos. Usar `source .env` primero.

## Database Access

- **MCP** (`mcp__railway-postgres__query`): **READ-ONLY**. Solo SELECT. Nunca INSERT/UPDATE/DELETE.
- **Escrituras**: Usar `psql` o scripts Python con `DATABASE_URL` del `.env`.
- **Hostname**: Usar siempre la URL **PÚBLICA** (`maglev.proxy.rlwy.net:24997`). El hostname interno (`postgres.railway.internal`) NO es alcanzable desde local.
- **psql directo**: Con `.env` cargado, basta `psql` (usa `PGPASSWORD`, `PGHOST`, etc.).

## Herramientas Disponibles

### MCP Servers

| Server | Descripción | Uso |
|--------|-------------|-----|
| `railway-postgres` | Queries read-only a PostgreSQL | `mcp__railway-postgres__query` con SQL |

### Regla Obligatoria: Verificar Schema antes de SQL

**ANTES de escribir cualquier query SQL nueva**, consultar el schema real:

```sql
-- Columnas de una tabla
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'matches' AND table_schema = 'public'
ORDER BY ordinal_position;

-- Para tablas en schema titan
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'feature_matrix' AND table_schema = 'titan'
ORDER BY ordinal_position;
```

**Tablas principales**: `matches`, `teams`, `predictions`, `titan.feature_matrix`

No asumir nombres de columnas (ej: usar `home_goals` no `home_score`).

### Commands (`.claude/commands/`)
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

### Skills (`.claude/skills/`)
Capacidades especializadas que el agente puede invocar.

| Skill | Descripción | Modo |
|-------|-------------|------|
| `secrets-scan` | Escanear repo por secretos expuestos | Manual only |
| `api-contract` | Validar contrato iOS vs backend | Read-only |
| `titan-ops-audit` | Auditoría operacional TITAN OMNISCIENCE | Read-only |

### Subagents (`.claude/skills/`)
Agentes especializados para tareas delegadas.

| Subagent | Descripción | Restricciones |
|----------|-------------|---------------|
| `ops-triage` | Diagnóstico operacional read-only | Sin edición, sin deploy |
| `db-analyst-ro` | Consultas DB solo SELECT | Sin INSERT/UPDATE/DELETE |
| `titan-ops-audit` | Audit TITAN: PIT, tiers, SOTA→TITAN | Solo SELECT, sin mutations |

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

## Backfill & Data Operations Protocol

Todas las operaciones de backfill deben seguir este protocolo:
1. **Pre-check**: Verificar row counts y estado actual vía MCP antes de mutar
2. **Branch**: Crear branch dedicada (`backfill/<scope>-<date>`)
3. **Execute**: Correr script con `source .env` cargado
4. **Post-verify**: Verificar row counts, date ranges, y que no haya contaminación cruzada
5. **PR**: Abrir PR con evidencia de verificación antes de merge

Anti-contaminación:
- Verificar que el backfill solo afectó la liga/región objetivo
- Comparar counts antes/después
- Revisar que no se sobreescribieron datos existentes

## Convenciones
- **Timestamps**: Siempre UTC (`datetime.utcnow()`)
- **Commits**: Conventional commits (`feat:`, `fix:`, `docs:`)
- **Deploy**: Push a `main` = deploy automático en Railway

## Arquitectura ML (Resumen)
- **Modelo activo**: XGBoost v1.0.0 (14 features)
- **Shadow mode**: Two-stage (en evaluación, NO en producción)
- **Sensor B**: Diagnóstico de calibración (no afecta predicciones)
- **SteamChaser**: Modelo secundario XGBoost binario — predice line movement post-lineup (shadow/data collection only)
- **CLV**: Closing Line Value 3-way log-odds contra canonical bookmaker (849 matches scored)
- **Market Anchor**: Blend model + market probs para ligas low-signal (Argentina α=1.0)

Para detalles técnicos de ML, shadow mode y sensor B, ver `docs/ML_ARCHITECTURE.md`.
Para Phase 2 (cascade, CLV, SteamChaser), ver `docs/PHASE2_ARCHITECTURE.md` y `docs/PHASE2_EVALUATION.md`.

## Jobs del Scheduler
| Job | Frecuencia | Función |
|-----|------------|---------|
| `global_sync` | 1 min | Sync partidos desde API-Football |
| `live_tick` | 10 seg | Actualizar partidos en vivo |
| `stats_backfill` | 60 min | Capturar stats de partidos FT |
| `odds_sync` | 6 horas | Sync odds para partidos próximos |
| `fastpath` | 2 min | Generar narrativas LLM |
| `lineup_monitoring` | 60-90 seg | Detectar lineups confirmadas, capturar odds frescos, emit LINEUP_CONFIRMED |
| `event_bus_sweeper` | 2 min | Reconciliar lineups sin cascade prediction (FOR UPDATE SKIP LOCKED) |
| `lineup_relative_movement` | 3 min | Capturar odds relativos a lineup detection (L0, L+5, L+10) |

## Tablas Clave (Phase 2)

| Tabla | Propósito |
|-------|-----------|
| `prediction_clv` | CLV 3-way log-odds por predicción (post-match scoring) |
| `player_id_mapping` | Mapeo API-Football ↔ Sofascore IDs (Hungarian bipartite) |
| `market_movement_snapshots` | Snapshots T5/T15/T30/T60 pre-kickoff |
| `lineup_movement_snapshots` | Snapshots L0/L+5/L+10 post-lineup |
| `match_lineups.lineup_detected_at` | Timestamp de detección de lineup (PIT anchor para cascade) |
| `predictions.asof_timestamp` | PIT anchor: momento exacto de generación de predicción |
