# FutbolStat MVP

Football Prediction System for FIFA World Cup. Predicts match outcomes (Home Win, Draw, Away Win) with probability percentages for value betting.

## Tech Stack

- **Backend**: Python 3.10+, FastAPI
- **Database**: PostgreSQL (Railway)
- **ML**: XGBoost
- **Data Source**: API-Football (RapidAPI)

## Project Structure

```
futbolstat/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes
│   ├── config.py            # Environment variables
│   ├── database.py          # Async PostgreSQL connection
│   ├── models.py            # DB tables
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract DataProvider class
│   │   ├── api_football.py  # API-Football implementation
│   │   ├── competitions.py  # Competition IDs and configs
│   │   └── pipeline.py      # ETL orchestrator
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py   # Feature calculations
│   └── ml/
│       ├── __init__.py
│       ├── engine.py        # XGBoost train/predict
│       └── metrics.py       # Brier Score + ROI simulation
├── models/                  # Saved model files (.json)
├── Procfile                 # Railway deployment
├── railway.json             # Railway config
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone and Install

```bash
cd futbolstat
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `RAPIDAPI_KEY`: Your API-Football key from RapidAPI

### 3. Run Locally

```bash
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for API documentation.

## API Endpoints

### Health Check
```
GET /health
```

### ETL Sync
```
POST /etl/sync
{
  "league_ids": [1, 28, 32],
  "season": 2024,
  "fetch_odds": true
}
```

### Train Model
```
POST /model/train
{
  "min_date": "2018-01-01",
  "league_ids": [1, 28, 32, 29, 30, 31]
}
```

### Get Predictions
```
GET /predictions/upcoming?league_ids=1,28
```

### List Competitions
```
GET /competitions
```

## Supported Competitions

### Priority: HIGH (Mandatory)
| Competition | League ID |
|-------------|-----------|
| FIFA World Cup | 1 |
| WC Qualifiers - CONMEBOL | 28 |
| WC Qualifiers - UEFA | 32 |
| WC Qualifiers - CONCACAF | 29 |
| WC Qualifiers - AFC | 30 |
| WC Qualifiers - CAF | 31 |

### Priority: MEDIUM
| Competition | League ID |
|-------------|-----------|
| Copa América | 9 |
| UEFA Euro | 4 |
| UEFA Nations League | 5 |
| CONCACAF Gold Cup | 22 |

### Priority: LOW
| Competition | League ID |
|-------------|-----------|
| International Friendlies | 10 |

## Features

### Feature Engineering
- **Rolling Averages** (Last 5 matches): Goals scored/conceded, shots, corners
- **Rest Days**: Days since last match for each team
- **Time Decay**: Exponential decay weighting for national teams (λ=0.01)
- **Match Weighting**: Official matches (1.0) vs Friendlies (0.6)

### ML Model
- **Algorithm**: XGBoost multi-class classifier
- **Target**: Home Win (0), Draw (1), Away Win (2)
- **Validation**: TimeSeriesSplit (no data leakage)
- **Output**: Probabilities + Fair Odds

### Evaluation Metrics
- **Brier Score**: Probability calibration
- **ROI Simulation**: Value betting with 5% edge threshold

## Deployment (Railway)

1. Connect your GitHub repo to Railway
2. Add PostgreSQL addon
3. Set environment variables
4. Deploy!

The `Procfile` and `railway.json` are already configured.

## Usage Example

```python
import httpx

# Sync World Cup qualifiers
response = httpx.post("http://localhost:8000/etl/sync", json={
    "league_ids": [28, 32, 29, 30, 31],
    "season": 2024
})

# Train model
response = httpx.post("http://localhost:8000/model/train")
print(f"Brier Score: {response.json()['brier_score']}")

# Get predictions
response = httpx.get("http://localhost:8000/predictions/upcoming")
for pred in response.json()["predictions"]:
    print(f"{pred['home_team']} vs {pred['away_team']}")
    print(f"  Home: {pred['probabilities']['home']:.1%}")
    print(f"  Draw: {pred['probabilities']['draw']:.1%}")
    print(f"  Away: {pred['probabilities']['away']:.1%}")
```

## Architecture: DB-First + Provider Fallback

The backend follows a **DB-first architecture** where data is always served from our database. External providers (API-Football) are only used as fallback on cache miss.

### Principles

1. **DB-First**: For historical/immutable data (events, standings, stats), always serve from DB if exists
2. **Provider Fallback**: Only call external API on miss, then persist to DB
3. **Match Status Rules**:
   - `FT/AET/PEN`: Never call provider in request path (backfill only)
   - `NS/LIVE`: Provider allowed but results should be cached/persisted

### Data Flow

```
Request → L1 Cache (memory, 30min) → L2 DB → L3 Provider Fallback → Persist to DB
```

### Endpoints Compliance

| Endpoint | Status | Source |
|----------|--------|--------|
| `/matches/{id}/details` | DB-first | cache → DB → skip (non-blocking) |
| `/matches/{id}/timeline` | DB-first | DB → provider fallback with persist |
| `/standings/{league_id}` | DB-first | cache → DB → provider fallback |
| `/predictions/upcoming` | Cache | cache → compute |

### Backfill Scripts

- `scripts/backfill_standings.py` - Populate league_standings table
- `scripts/backfill_events.py` - Populate match.events for timeline

### Running Backfills

```bash
# Backfill standings for active leagues
python scripts/backfill_standings.py

# Backfill events for last 7 days
python scripts/backfill_events.py --days 7

# Dry run (no writes)
python scripts/backfill_standings.py --dry-run
```

## License

MIT
