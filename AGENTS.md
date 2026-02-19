---
name: bon-jogo
version: "2.0"
updated: 2026-02-03
description: Sistema de predicciones de fútbol con ML y narrativas LLM
language: es
---

# Bon Jogo - Guía para Agentes de Código

## Visión General del Proyecto

**Bon Jogo** es un sistema de predicciones de resultados de partidos de fútbol (Victoria Local, Empate, Victoria Visitante) con probabilidades para identificar oportunidades de value betting. Incluye narrativas generadas por LLM, dashboard operacional y arquitectura DB-first.

### Stack Tecnológico

| Capa | Tecnología |
|------|------------|
| **Backend** | Python 3.12+, FastAPI (monolito en `app/main.py`) |
| **Database** | PostgreSQL (Railway), SQLAlchemy 2.0 + SQLModel (async) |
| **ML** | XGBoost multi-class classifier |
| **LLM** | Gemini 2.0 Flash (primario), RunPod/Qwen (fallback) |
| **Frontend Dashboard** | Next.js 16 + React 19 + TypeScript + Tailwind CSS 4 |
| **iOS Client** | Swift/SwiftUI |
| **Deploy** | Railway (auto-deploy en push a main) |
| **Observability** | Prometheus + Grafana + Sentry |

### Estructura de Directorios

```
/Users/inseqio/FutbolStats/
├── app/                           # Backend FastAPI
│   ├── main.py                    # Monolito principal (~8000 líneas)
│   ├── config.py                  # Pydantic Settings (env vars)
│   ├── database.py                # Async SQLAlchemy engine
│   ├── models.py                  # SQLModel tables (~1500 líneas)
│   ├── scheduler.py               # APScheduler jobs
│   ├── security.py                # Auth, rate limiting (SlowAPI)
│   ├── etl/                       # API-Football, Understat, Sofascore
│   ├── ml/                        # XGBoost engine, shadow mode, sensor B
│   ├── llm/                       # Narrativas con Gemini/RunPod
│   ├── features/                  # Feature engineering (SOTA + baseline)
│   ├── telemetry/                 # Prometheus, Sentry
│   ├── titan/                     # TITAN OMNISCIENCE feature store
│   ├── dashboard/                 # Endpoints de dashboard
│   ├── logos/                     # Gestión de escudos equipos (R2)
│   └── jobs/                      # Jobs programados
├── models/                        # ML artifacts (XGBoost .json files)
├── dashboard/                     # Next.js frontend
├── scripts/                       # Scripts de evaluación y backfill
├── tests/                         # Tests unitarios + tests/titan/
├── migrations/                    # Migraciones SQL
├── docs/                          # Protocolos operacionales
└── ios/                           # App iOS Swift/SwiftUI
```

## Configuración y Variables de Entorno

### Archivos de Configuración Clave

| Archivo | Propósito |
|---------|-----------|
| `requirements.txt` | Dependencias Python |
| `Procfile` | Comando de inicio Railway |
| `railway.json` | Configuración de deploy NIXPACKS |
| `.env.example` | Template de variables de entorno |
| `app/config.py` | Pydantic Settings con defaults |

### Variables de Entorno Requeridas

```bash
# Database (obligatorio)
DATABASE_URL=postgresql://user:pass@host:port/db

# API-Football (RapidAPI) - obligatorio
RAPIDAPI_KEY=your_key_here
RAPIDAPI_HOST=api-football-v1.p.rapidapi.com

# API Security (producción)
API_KEY=your_api_key
DASHBOARD_TOKEN=your_dashboard_token

# Sentry (opcional pero recomendado)
SENTRY_DSN=your_sentry_dsn

# LLM Providers
GEMINI_API_KEY=your_gemini_key
RUNPOD_API_KEY=your_runpod_key

# Email Alerts (SMTP)
SMTP_ENABLED=false
SMTP_USERNAME=
SMTP_PASSWORD=

# OPS Console (web login)
OPS_ADMIN_PASSWORD=
OPS_SESSION_SECRET=
```

## Comandos de Desarrollo

### Setup Local

```bash
# Crear virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar environment
cp .env.example .env
# Editar .env con credenciales

# Iniciar servidor de desarrollo
uvicorn app.main:app --reload --port 8000
```

### Testing

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/test_predictions.py -v
pytest tests/titan/ -v

# Tests con coverage
pytest tests/ --cov=app --cov-report=html
```

### Scripts Útiles

```bash
# Backfill standings
python scripts/backfill_standings.py

# Backfill events (últimos 7 días)
python scripts/backfill_events.py --days 7

# Evaluar modelo PIT
python scripts/evaluate_pit_v3.py

# Diagnóstico de modelo
python scripts/audit_fase1_apples_to_apples.py

# Backfill TITAN dataset
python scripts/build_titan_dataset.py
```

## Arquitectura del Sistema

### 1. Sistema de Predicciones (ML Engine)

**Archivos**: `app/ml/engine.py`, `app/ml/shadow.py`, `app/ml/sensor.py`

- **Modelo Producción**: XGBoost baseline (v1.0.1-league-only)
- **Shadow Mode**: Two-stage classifier (en evaluación, NO en producción)
- **Sensor B**: Diagnóstico de calibración (no afecta predicciones)
- **Features**: 14 features (rolling averages, rest days, time decay, match weighting)

### 2. Sistema TITAN OMNISCIENCE

**Archivos**: `app/titan/`, `scripts/build_titan_dataset.py`

- Feature store y tier system para ML avanzado
- **Tablas**: `titan.feature_matrix`, `titan.tier_coverage`, `titan.sota_to_titan`
- **Tiers de Datos**:
  - Tier A: Fixture + time + score + lineup + coverage completo
  - Tier B: Fixture + time + score + lineup (no coverage)
  - Tier C: Fixture + time + score (no lineup, no coverage)
  - Tier D: Fixture + time (no score, no lineup, no coverage)

### 3. Arquitectura DB-First

**Principio**: Los datos históricos siempre se sirven desde la BD. Provider fallback solo para misses.

**Flujo de Datos**:
```
Request → L1 Cache (memory 30min) → L2 (DB) → L3 (API-Football) → Persist to DB
```

**Reglas de Match Status**:
- `FT/AET/PEN`: Nunca llamar provider en request path (backfill only)
- `NS/LIVE`: Provider permitido pero cachear/persistir resultados

### 4. Scheduler y Jobs

**Archivo**: `app/scheduler.py` (APScheduler)

| Job | Frecuencia | Descripción |
|-----|------------|-------------|
| `global_sync` | 1 min | Sync partidos API-Football |
| `live_tick` | 10 seg | Actualizar partidos en vivo |
| `stats_backfill` | 60 min | Capturar stats FT |
| `odds_sync` | 6 horas | Sync cuotas |
| `fastpath` | 2 min | Generar narrativas LLM |
| `shadow_eval` | 30 min | Evaluar shadow models |
| `sensor_retrain` | 6 horas | Re-entrenar Sensor B |

## Modelos de Base de Datos

### Tablas Principales (schema public)

| Tabla | Descripción |
|-------|-------------|
| `teams` | Equipos (nacionales y clubs) |
| `matches` | Partidos (status: NS/LIVE/FT/AET/PEN) |
| `predictions` | Predicciones ML con frozen odds |
| `prediction_outcomes` | Resultados para evaluación |
| `post_match_audits` | Auditoría post-partido con LLM narratives |
| `odds_history` | Historial de cuotas con timestamps |
| `shadow_predictions` | Predicciones shadow para A/B testing |
| `sensor_predictions` | Predicciones Sensor B (diagnóstico) |
| `league_standings` | Tablas de posiciones cacheadas |
| `model_snapshots` | Modelos almacenados en BD (BYTEA) |

### Schema TITAN

| Tabla | Descripción |
|-------|-------------|
| `titan.feature_matrix` | Feature store completo |
| `titan.raw_extractions` | Extracciones raw de providers |
| `titan.sota_to_titan` | Mapeo SOTA → TITAN |

**Nota importante**: Antes de queries nuevas, verificar schema:
```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'matches' AND table_schema = 'public'
ORDER BY ordinal_position;
```

## Endpoints de API

### Públicos (sin auth)
- `GET /health` - Health check

### Protegidos (requieren X-API-Key)
- `GET /predictions/upcoming` - Predicciones próximos partidos
- `GET /predictions/match/{id}` - Predicción específica
- `GET /matches` - Lista partidos
- `GET /matches/{id}/details` - Detalles partido (DB-first)
- `GET /matches/{id}/insights` - Insights/narrativa
- `GET /matches/{id}/timeline` - Timeline eventos
- `GET /matches/{id}/odds-history` - Historial cuotas
- `GET /standings/{league_id}` - Tabla posiciones
- `GET /live-summary` - Partidos en vivo

### Dashboard (requieren X-Dashboard-Token u OPS session)
- `GET /dashboard/ops.json` - Dashboard operacional
- `GET /dashboard/pit.json` - Dashboard PIT
- `GET /dashboard/model-benchmark` - Benchmark de modelos

## Convenciones de Código

### Python

- **Async/await** para operaciones I/O
- **SQLModel** para modelos de BD
- **structlog** para logging estructurado
- **Pydantic** para validación

### Timestamps

Siempre usar UTC:
```python
from datetime import datetime
datetime.utcnow()  # Para timestamps
```

### Nombres de Columnas

- `home_goals` / `away_goals` (NO `home_score`)
- `external_id` para IDs de API externas
- `created_at` / `updated_at` para timestamps de registro

### Commits

Usar conventional commits:
```
feat: add fastpath job for llm narratives
fix: correct shadow prediction evaluation
refactor: extract feature engineering to module
docs: update AGENTS.md
```

## Testing Strategy

### Estrategia de Tests

1. **Tests Unitarios**: `tests/test_*.py`
2. **Tests TITAN**: `tests/titan/test_*.py`
3. **PIT Compliance Tests**: `tests/test_feature_engineering_pit.py`

### PIT (Point-In-Time) Compliance

**OBLIGATORIO** para datasets de ML:
- Solo usar datos con `match_date < t0` (kickoff)
- Snapshot features requieren `captured_at < t0`
- Missing data → `*_missing=1` flags, no crashes

### Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Tests específicos de TITAN
pytest tests/titan/test_pit_compliance.py -v
pytest tests/titan/test_idempotency.py -v
```

## Seguridad

### Rate Limiting (SlowAPI)

```python
from app.security import limiter

@app.get("/predictions/upcoming")
@limiter.limit("60/minute")
async def get_predictions(request: Request):
    ...
```

### API Key Authentication

- Header: `X-API-Key`
- En producción: `API_KEY` debe estar configurado (fail-closed)
- En desarrollo: API_KEY vacío permite todas las requests

### OPS Session Auth

Para endpoints de dashboard web:
- Login vía `/ops/login`
- Session cookie con TTL configurable (`OPS_SESSION_TTL_HOURS`)
- Verificación vía `verify_api_key_or_ops_session()`

## Deployment

### Railway (Producción)

1. Push a `main` = deploy automático
2. Verificar `/health` post-deploy
3. Monitorear logs: `railway logs -n 50`

### Variables de Producción Requeridas

```bash
RAILWAY_ENVIRONMENT=production
DATABASE_URL=postgresql://...
RAPIDAPI_KEY=...
API_KEY=...
SENTRY_DSN=...
```

### Checklist Pre-Deploy

- [ ] Tests pasan: `pytest tests/`
- [ ] No hay secrets expuestos
- [ ] Migraciones aplicadas
- [ ] Health check responde 200

## Protocolos Operacionales

### Antes de Operaciones Críticas

Consultar documentación en `docs/`:

| Operación | Documento |
|-----------|-----------|
| Agregar competición | `docs/COMPETITION_ONBOARDING.md` |
| Troubleshooting | `docs/OPS_RUNBOOK.md` |
| Evaluar modelo | `docs/PIT_EVALUATION_PROTOCOL.md` |
| Onboarding de agentes | `docs/AUDITOR_ONBOARDING.md` |

### Comandos Útiles (/cursor/commands/)

Los siguientes comandos están disponibles para agentes:

- `/ops` - Estado operacional
- `/logs [filtro]` - Logs de Railway
- `/match <equipo>` - Buscar partido
- `/verify` - Smoke test post-deploy
- `/model-sanity` - Verificar modelo ML
- `/db-report` - Reportes DB ad-hoc

## Roles del Proyecto

| Rol | Nombre | Responsabilidad |
|-----|--------|-----------------|
| **Owner** | David | Dueño del producto |
| **Auditor Backend** | ABE | Director técnico backend. NO escribe código. |
| **Auditor TITAN** | ATI | Director técnico TITAN. NO escribe código. |
| **Auditor Dashboard** | ADB | Director técnico dashboard. NO escribe código. |
| **Auditor Data** | ADA | Data & Swarm Specialist. NO escribe código. |
| **Codificador Backend** | Master | Ejecuta código backend según instrucciones |
| **Codificador Dashboard** | Claude | Implementa UI según instrucciones |

## Notas para Agentes

1. **Siempre verificar schema antes de SQL queries**
2. **Usar comandos `/ops`, `/logs`, `/verify` para debugging**
3. **Preguntar a ABE/ATI antes de cambios significativos**
4. **Mantener PIT compliance para datasets de ML**
5. **Documentar decisiones en commits y docs**
6. **Follow DB-first architecture principles**
7. **No deployar sin verificar health endpoint**
8. **Manejar tiers A/B/C/D en queries de TITAN**
9. **Mantener AGENTS.md actualizado**

---

*Última actualización: 2026-02-03*
