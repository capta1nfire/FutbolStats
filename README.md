# Bon Jogo

Football prediction system with ML models and LLM-generated narratives. Covers 25+ domestic leagues across Europe, South America, North America, and the Middle East. Predicts match outcomes (Home/Draw/Away) with calibrated probabilities for value betting analysis.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, SQLAlchemy (async) |
| **Database** | PostgreSQL (Railway) |
| **ML** | XGBoost (baseline + Family S tier model) |
| **LLM** | Gemini 2.0 Flash (post-match narratives) |
| **Dashboard** | Next.js, TypeScript, Tailwind CSS, shadcn/ui |
| **iOS** | Swift, SwiftUI |
| **Scheduler** | APScheduler (in-process) |
| **Observability** | Prometheus, Grafana, Sentry |
| **Infra** | Railway (deploy), Cloudflare R2 (storage) |
| **Data Sources** | API-Football, FotMob, Understat, OddsPortal, Football-Data.co.uk |

## Project Structure

```
app/
├── main.py                  # FastAPI app factory + startup
├── config.py                # Pydantic settings
├── database.py              # Async SQLAlchemy engine
├── scheduler.py             # APScheduler job definitions
├── routes/
│   ├── core.py              # Health, root
│   └── api.py               # Predictions, matches, teams, odds, model endpoints
├── dashboard/
│   ├── ops_routes.py        # Operational dashboard (ops.json, PIT, budget)
│   ├── dashboard_views_routes.py  # Matches, predictions, standings views
│   ├── football_routes.py   # Football navigation (countries, leagues, teams)
│   ├── admin_routes.py      # League config, rules, zones
│   ├── settings_routes.py   # Feature flags, model settings
│   ├── benchmark_matrix.py  # Model vs Pinnacle benchmark
│   └── model_benchmark.py   # Model accuracy over time
├── ml/
│   ├── engine.py            # XGBoost train/predict (baseline)
│   ├── family_s.py          # Family S engine (Tier 3 MTV model)
│   ├── league_router.py     # Tier-based model routing
│   ├── policy.py            # Market anchor, draw cap
│   ├── shadow.py            # Shadow model evaluation
│   ├── sensor.py            # Calibration diagnostics
│   ├── consensus.py         # Multi-source consensus
│   ├── devig.py             # Odds de-vigging
│   └── persistence.py       # Model snapshot management
├── features/
│   ├── engineering.py       # Feature engineering pipeline
│   └── player_features.py   # Player-level features
├── etl/
│   ├── api_football.py      # API-Football provider
│   ├── fotmob_provider.py   # FotMob xG + refs
│   ├── understat_provider.py # Understat xG
│   ├── sofascore_provider.py # SofaScore lineups
│   ├── competitions.py      # League/competition registry
│   └── pipeline.py          # ETL orchestrator
├── llm/
│   ├── gemini_client.py     # Gemini API client
│   ├── narrative_generator.py # Post-match narratives
│   ├── fastpath.py          # Fast narrative pipeline
│   └── claim_validator.py   # Fact-checking narratives
└── telemetry/
    ├── metrics.py           # Prometheus metrics
    └── sentry.py            # Sentry error tracking

dashboard/                   # Next.js frontend
├── app/                     # App router pages
├── components/              # React components (shadcn/ui)
├── lib/                     # Types, API client, hooks
└── public/                  # Static assets

ios/FutbolStats/             # iOS app
├── Services/                # API client, LiveScoreManager
├── ViewModels/              # PredictionsViewModel
└── Views/                   # SwiftUI views

scripts/                     # Backfill, evaluation, scraping utilities
docs/                        # Operational protocols and runbooks
models/                      # ML model artifacts (.json)
data/                        # Static data files (aliases, mappings)
```

## Supported Leagues

### Tier 1 — Top 5 European
| League | ID | Country |
|--------|-----|---------|
| Premier League | 39 | England |
| La Liga | 140 | Spain |
| Serie A | 135 | Italy |
| Bundesliga | 78 | Germany |
| Ligue 1 | 61 | France |

### Tier 2 — Secondary European + Americas
| League | ID | Country |
|--------|-----|---------|
| Championship | 40 | England |
| Argentina Primera | 128 | Argentina |
| Brazil Serie A | 71 | Brazil |
| Colombia Primera A | 239 | Colombia |
| Liga MX | 262 | Mexico |
| MLS | 253 | USA |

### Tier 3 — Family S (MTV-enhanced model)
| League | ID | Country |
|--------|-----|---------|
| Eredivisie | 88 | Netherlands |
| Primeira Liga | 94 | Portugal |
| Belgian Pro League | 144 | Belgium |
| Süper Lig | 203 | Turkey |
| Primera División | 265 | Chile |

### Additional Leagues
Bolivia, Ecuador, Paraguay, Peru, Uruguay, Venezuela, Saudi Pro League.

### International
FIFA World Cup, WC Qualifiers (all confederations), Copa America, UEFA Euro, Nations League, Champions League, Europa League, Conference League.

## ML Architecture

### Baseline Model
- **Algorithm**: XGBoost multi-class classifier (Home/Draw/Away)
- **Features**: 17 core (rolling averages, rest days, strength gaps) + 3 odds
- **Training**: `league_only=True`, min date 2023-01-01, 3-fold stratified CV
- **Evaluation**: Brier Score, LogLoss, Skill % vs Pinnacle

### Family S (Tier 3)
- **Scope**: 5 leagues where MTV (Market-to-Talent Value) improves predictions
- **Features**: 24 (17 core + 3 odds + 4 MTV from Transfermarkt squad values)
- **Serving**: Cascade handler writes to DB, serving layer overlays onto baseline
- **Activation**: `LEAGUE_ROUTER_MTV_ENABLED` env var, instant rollback

### xG Data
- **Understat**: Top 5 European leagues (EPL, La Liga, Serie A, Bundesliga, Ligue 1)
- **FotMob**: 16 additional leagues (Argentina, Brazil, Colombia, Eredivisie, etc.)
- **Coverage**: 15/25 leagues have >90% xG coverage since 2023

### Market Anchor
Blends model probabilities with de-vigged bookmaker odds for leagues where the market outperforms the model. Configurable per-league via `LEAGUE_OVERRIDES`.

## Scheduler Jobs

| Job | Frequency | Description |
|-----|-----------|-------------|
| `global_sync` | 1 min | Sync match data from API-Football |
| `live_tick` | 10 sec | Update live match scores |
| `stats_backfill` | 60 min | Capture post-match statistics |
| `odds_sync` | 6 hours | Sync odds for upcoming matches |
| `fastpath` | 2 min | Generate LLM narratives |
| `xg_sync` | 6 hours | FotMob xG ingestion |
| `refs_sync` | 12 hours | FotMob referee data |
| `daily_predictions` | daily | Pre-compute and save predictions |
| `shadow_recalibration` | weekly (Tue) | Evaluate shadow model candidates |

## API Endpoints

All protected endpoints require `X-API-Key` header.

### Predictions
| Endpoint | Description |
|----------|-------------|
| `GET /predictions/upcoming` | Upcoming match predictions (7-day window) |
| `GET /predictions/match/{id}` | Single match prediction |

### Matches & Teams
| Endpoint | Description |
|----------|-------------|
| `GET /matches` | Match list with filters |
| `GET /matches/{id}/details` | Full match details + prediction |
| `GET /matches/{id}/insights` | LLM narrative for match |
| `GET /matches/{id}/timeline` | Match event timeline |
| `GET /matches/{id}/odds-history` | Historical odds movement |
| `GET /matches/{id}/lineup` | Confirmed lineups |
| `GET /teams` | Team list |
| `GET /teams/{id}/history` | Team match history |
| `GET /competitions` | Competition list |
| `GET /standings/{league_id}` | League standings |
| `GET /live-summary` | Live match scores |

### Dashboard (requires `X-Dashboard-Token`)
| Endpoint | Description |
|----------|-------------|
| `GET /dashboard/ops.json` | Operational dashboard data |
| `GET /dashboard/pit.json` | Point-in-Time evaluation |
| `GET /dashboard/matches` | Matches view with model columns |
| `GET /dashboard/predictions` | Predictions with benchmark matrix |
| `GET /dashboard/benchmark-matrix` | Model vs Pinnacle comparison |
| `GET /dashboard/model-benchmark` | Model accuracy over time |

## Setup

### Prerequisites
- Python 3.12+
- PostgreSQL
- Node.js 18+ (for dashboard)

### Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Configure credentials
uvicorn app.main:app --reload
```

### Dashboard

```bash
cd dashboard
npm install
npm run dev
```

### Environment Variables

All configuration is in `.env` (see `.env.example`). Key variables:
- `DATABASE_URL` / `DATABASE_URL_ASYNC` — PostgreSQL connection
- `API_FOOTBALL_KEY` — API-Football data source
- `FUTBOLSTATS_API_KEY` — Backend API authentication
- `DASHBOARD_TOKEN` — Dashboard endpoint authentication
- `GEMINI_API_KEY` — Google Gemini for narratives
- `LEAGUE_ROUTER_MTV_ENABLED` — Family S activation flag
- `MARKET_ANCHOR_ENABLED` — Market anchor blending

## Deployment

Push to `main` triggers automatic deploy on Railway.

```bash
git push origin main
```

Railway configuration: `Procfile` + `railway.json`.

## Documentation

| Document | Description |
|----------|-------------|
| `docs/OPS_RUNBOOK.md` | Operational troubleshooting guide |
| `docs/COMPETITION_ONBOARDING.md` | Adding new leagues |
| `docs/PIT_EVALUATION_PROTOCOL.md` | Model evaluation protocol |
| `docs/FAMILY_S_TRAINING_RESULTS.md` | Family S training report |
| `docs/ML_ARCHITECTURE.md` | ML system architecture |
| `docs/GRAFANA_ALERTS_CHECKLIST.md` | Monitoring alerts setup |

## License

MIT
