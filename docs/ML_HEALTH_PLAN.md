# ML Health Dashboard - Plan de Implementación

**Endpoint: GET /dashboard/ml_health.json**

| Metadata | Valor |
|----------|-------|
| Versión | 1.1 |
| Fecha | 2026-01-26 |
| Estado | **APROBADO CON CAMBIOS (ATI)** |
| Solicitado por | ATI (Auditor TITAN) |
| Ejecuta | Master (Backend) |

---

## 1. Contexto: Hallazgo "Vuelo a Ciegas"

El modelo XGBoost v1.0.0 operó meses con "combustible incompleto":
- 14 features declaradas, pero shots/corners con 0% coverage en 23/24
- Stats backfill se agregó tarde (2026-01-09) con solo 72h lookback
- **Causa raíz**: No había visibilidad de coverage por feature/temporada

**Principio del Owner**: "Lo que no puedo VER, no existe en mi mundo operativo."

---

## 2. Objetivo

Endpoint único que responda en <2s:
- ¿Qué feature falló?
- ¿En qué liga/temporada?
- ¿Desde cuándo?
- ¿En qué etapa del pipeline?

---

## 3. Estructura de Respuesta (v1.1 - ATI Approved)

```json
{
  "generated_at": "2026-01-26T15:30:00Z",
  "cached": false,
  "cache_age_seconds": null,
  "health": "ok|partial|error",
  "data": {
    "fuel_gauge": {
      "status": "ok|warn|error",
      "reasons": ["All systems nominal"],
      "as_of_utc": "2026-01-26T15:30:00Z"
    },

    "sota_stats_coverage": {
      "by_season": {
        "23/24": {
          "total_matches_ft": 1140,
          "with_stats_pct": 0.0,
          "marked_no_stats_pct": 100.0,
          "shots_present_pct": 0.0
        },
        "24/25": {
          "total_matches_ft": 950,
          "with_stats_pct": 45.2,
          "marked_no_stats_pct": 54.8,
          "shots_present_pct": 45.2
        },
        "25/26": {
          "total_matches_ft": 320,
          "with_stats_pct": 92.5,
          "marked_no_stats_pct": 7.5,
          "shots_present_pct": 92.5
        }
      },
      "by_league": [
        {"league_id": 140, "name": "La Liga", "with_stats_pct": 85.3},
        {"league_id": 39, "name": "Premier League", "with_stats_pct": 82.1},
        {"league_id": 135, "name": "Serie A", "with_stats_pct": 78.9}
      ],
      "status": "warn"
    },

    "titan_coverage": {
      "by_season": {
        "25/26": {
          "tier1": {"complete": 19, "total": 19, "pct": 100.0},
          "tier1b": {"complete": 4, "total": 19, "pct": 21.1},
          "tier1c": {"complete": 3, "total": 19, "pct": 15.8},
          "tier1d": {"complete": 2, "total": 19, "pct": 10.5}
        }
      },
      "by_league": [
        {"league_id": 140, "tier1_pct": 100.0, "tier1b_pct": 25.0}
      ],
      "status": "ok"
    },

    "pit_compliance": {
      "total_rows": 19,
      "violations": 0,
      "violation_pct": 0.0,
      "status": "ok"
    },

    "freshness": {
      "age_hours_now": {
        "odds": {"p50": 12.3, "p95": 48.5, "max": 72.1},
        "xg": {"p50": 24.1, "p95": 96.2, "max": 168.0}
      },
      "lead_time_hours": {
        "odds": {"p50": 20.6, "p95": 44.5, "max": 44.8},
        "xg": {"p50": 19.0, "p95": 37.9, "max": 40.9}
      },
      "status": "ok"
    },

    "prediction_confidence": {
      "entropy": {"avg": 0.901, "p25": 0.861, "p50": 0.927, "p75": 0.968, "p95": 0.993},
      "tier_distribution": {"gold": 214, "silver": 296, "copper": 107},
      "sample_n": 617,
      "window_days": 30
    },

    "top_regressions": {
      "status": "not_ready",
      "note": "Requires baseline snapshot - will compare current vs previous window after 48h of data"
    }
  }
}
```

---

## 4. Thresholds P0 (Fuel Gauge) - ATI v1.1

| Métrica | Warn | Error |
|---------|------|-------|
| PIT violations | >0 | >0 |
| SOTA stats coverage (current season) | <70% | <50% |
| TITAN tier1 coverage | <80% | <50% |
| Freshness age_hours_now p95 (odds) | >6h | >24h |
| Freshness age_hours_now p95 (xG) | >24h | >72h |

```python
def _compute_fuel_gauge(data: dict, degraded_sections: list[str]) -> dict:
    """
    Compute fuel gauge from collected metrics.

    ATI v1.1: Uses correct paths from data dict, includes SOTA stats coverage,
    uses age_hours_now for staleness detection.
    """
    reasons = []
    status = "ok"

    # Degraded sections (fail-soft triggered)
    if degraded_sections:
        status = "warn"
        for section in degraded_sections:
            reasons.append(f"Degraded section: {section}")

    # PIT violations (crítico)
    pit = data.get("pit_compliance", {})
    if pit.get("violations", 0) > 0:
        status = "error"
        reasons.append(f"PIT violations: {pit['violations']}")

    # SOTA stats coverage (current season 25/26) - CRÍTICO para XGBoost
    sota = data.get("sota_stats_coverage", {}).get("by_season", {}).get("25/26", {})
    sota_pct = sota.get("with_stats_pct", 0)
    if sota_pct < 50:
        status = "error"
        reasons.append(f"SOTA stats coverage critical: {sota_pct}%")
    elif sota_pct < 70 and status != "error":
        status = "warn"
        reasons.append(f"SOTA stats coverage low: {sota_pct}%")

    # TITAN tier1 coverage
    titan = data.get("titan_coverage", {}).get("by_season", {}).get("25/26", {}).get("tier1", {})
    tier1_pct = titan.get("pct", 0)
    if tier1_pct < 50:
        status = "error"
        reasons.append(f"TITAN tier1 coverage critical: {tier1_pct}%")
    elif tier1_pct < 80 and status != "error":
        status = "warn"
        reasons.append(f"TITAN tier1 coverage low: {tier1_pct}%")

    # Freshness - age_hours_now (early warning for pipeline down)
    freshness = data.get("freshness", {}).get("age_hours_now", {})
    odds_p95 = freshness.get("odds", {}).get("p95")
    if odds_p95 and odds_p95 > 24:
        status = "error"
        reasons.append(f"Odds staleness critical: p95={odds_p95}h ago")
    elif odds_p95 and odds_p95 > 6 and status != "error":
        status = "warn"
        reasons.append(f"Odds staleness elevated: p95={odds_p95}h ago")

    xg_p95 = freshness.get("xg", {}).get("p95")
    if xg_p95 and xg_p95 > 72:
        if status != "error":
            status = "error"
        reasons.append(f"xG staleness critical: p95={xg_p95}h ago")
    elif xg_p95 and xg_p95 > 24 and status != "error":
        status = "warn"
        reasons.append(f"xG staleness elevated: p95={xg_p95}h ago")

    if not reasons:
        reasons = ["All systems nominal"]

    return {
        "status": status,
        "reasons": reasons,
        "as_of_utc": datetime.utcnow().isoformat() + "Z",
    }
```

---

## 5. Queries SQL Clave (ATI v1.1)

### 5.1 SOTA Stats Coverage por Temporada (P0 - Causa Raíz)

```sql
-- Coverage de stats en matches (fuente del problema "vuelo a ciegas")
SELECT
    CASE
        WHEN date >= '2023-08-01' AND date < '2024-08-01' THEN '23/24'
        WHEN date >= '2024-08-01' AND date < '2025-08-01' THEN '24/25'
        WHEN date >= '2025-08-01' AND date < '2026-08-01' THEN '25/26'
    END as season,
    COUNT(*) as total_matches_ft,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE stats IS NOT NULL
        AND stats != '{}'::jsonb
        AND (stats->>'_no_stats') IS NULL
    ) / NULLIF(COUNT(*), 0), 1) as with_stats_pct,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE (stats->>'_no_stats')::boolean = true
    ) / NULLIF(COUNT(*), 0), 1) as marked_no_stats_pct,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE stats->'home'->>'total_shots' IS NOT NULL
    ) / NULLIF(COUNT(*), 0), 1) as shots_present_pct
FROM matches
WHERE status IN ('FT', 'AET', 'PEN')
  AND date >= '2023-08-01'
  AND league_id IN (140, 39, 135, 78, 61)  -- Top 5 leagues
GROUP BY 1
ORDER BY 1
```

### 5.2 SOTA Stats Coverage por Liga

```sql
SELECT
    league_id,
    l.name,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE m.stats IS NOT NULL
        AND m.stats != '{}'::jsonb
        AND (m.stats->>'_no_stats') IS NULL
    ) / NULLIF(COUNT(*), 0), 1) as with_stats_pct
FROM matches m
JOIN leagues l ON m.league_id = l.external_id
WHERE m.status IN ('FT', 'AET', 'PEN')
  AND m.date >= '2025-08-01'  -- Current season only
  AND m.league_id IN (140, 39, 135, 78, 61)
GROUP BY league_id, l.name
ORDER BY with_stats_pct DESC
```

### 5.3 TITAN Coverage por Temporada

```sql
SELECT
    CASE
        WHEN kickoff_utc >= '2025-08-01' AND kickoff_utc < '2026-08-01' THEN '25/26'
    END as season,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE tier1_complete) as tier1_count,
    COUNT(*) FILTER (WHERE tier1b_complete) as tier1b_count,
    COUNT(*) FILTER (WHERE tier1c_complete) as tier1c_count,
    COUNT(*) FILTER (WHERE tier1d_complete) as tier1d_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE tier1_complete) / NULLIF(COUNT(*), 0), 1) as tier1_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE tier1b_complete) / NULLIF(COUNT(*), 0), 1) as tier1b_pct
FROM titan.feature_matrix
WHERE kickoff_utc >= '2025-08-01'
GROUP BY 1
```

### 5.4 PIT Compliance

```sql
SELECT
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE pit_max_captured_at >= kickoff_utc) as violations,
    ROUND(100.0 * COUNT(*) FILTER (WHERE pit_max_captured_at >= kickoff_utc) / NULLIF(COUNT(*), 0), 2) as violation_pct
FROM titan.feature_matrix
WHERE kickoff_utc >= '2025-08-01'
```

### 5.5 Freshness - Age Hours Now (Early Warning)

```sql
-- age_hours_now: tiempo desde captura hasta AHORA (detecta pipeline caído)
SELECT
    'odds' as tier,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (NOW() - odds_captured_at))/3600
    )::numeric, 1) as p50,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (NOW() - odds_captured_at))/3600
    )::numeric, 1) as p95,
    ROUND(MAX(EXTRACT(EPOCH FROM (NOW() - odds_captured_at))/3600)::numeric, 1) as max
FROM titan.feature_matrix
WHERE odds_captured_at IS NOT NULL
  AND kickoff_utc >= NOW() - INTERVAL '7 days'

UNION ALL

SELECT
    'xg' as tier,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (NOW() - xg_captured_at))/3600
    )::numeric, 1) as p50,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (NOW() - xg_captured_at))/3600
    )::numeric, 1) as p95,
    ROUND(MAX(EXTRACT(EPOCH FROM (NOW() - xg_captured_at))/3600)::numeric, 1) as max
FROM titan.feature_matrix
WHERE xg_captured_at IS NOT NULL
  AND kickoff_utc >= NOW() - INTERVAL '7 days'
```

### 5.6 Freshness - Lead Time Hours (Contexto)

```sql
-- lead_time_hours: tiempo desde captura hasta kickoff (contexto operacional)
SELECT
    'odds' as tier,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (kickoff_utc - odds_captured_at))/3600
    )::numeric, 1) as p50,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (kickoff_utc - odds_captured_at))/3600
    )::numeric, 1) as p95,
    ROUND(MAX(EXTRACT(EPOCH FROM (kickoff_utc - odds_captured_at))/3600)::numeric, 1) as max
FROM titan.feature_matrix
WHERE odds_captured_at IS NOT NULL
  AND kickoff_utc >= NOW() - INTERVAL '7 days'
```

### 5.7 Prediction Confidence (Entropía)

```sql
WITH prediction_entropy AS (
    SELECT
        -1 * (home_prob * LN(home_prob) + draw_prob * LN(draw_prob) + away_prob * LN(away_prob)) / LN(3) as normalized_entropy,
        frozen_confidence_tier
    FROM predictions
    WHERE home_prob > 0 AND draw_prob > 0 AND away_prob > 0
      AND created_at > NOW() - INTERVAL '30 days'
)
SELECT
    COUNT(*) as sample_n,
    ROUND(AVG(normalized_entropy)::numeric, 3) as avg,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p25,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p50,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p75,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY normalized_entropy)::numeric, 3) as p95,
    COUNT(*) FILTER (WHERE frozen_confidence_tier = 'gold') as gold,
    COUNT(*) FILTER (WHERE frozen_confidence_tier = 'silver') as silver,
    COUNT(*) FILTER (WHERE frozen_confidence_tier = 'copper') as copper
FROM prediction_entropy
```

---

## 6. Implementación (ATI v1.1)

### 6.1 Arquitectura de Archivos

```
app/
├── ml/
│   └── health.py              # NUEVO: Core queries y lógica ML Health
└── main.py                    # Solo endpoint + auth + cache (delgado)
```

**Decisión ATI**: No duplicar lógica. Reutilizar patrones de `app/titan/dashboard.py` pero en módulo separado `app/ml/health.py`.

### 6.2 Cache In-Memory con Snapshot para Regressions

```python
_ml_health_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 60,  # 60 segundos
    "previous_snapshot": None,  # Para calcular top_regressions
    "previous_snapshot_at": None,
}
```

### 6.3 Endpoint (main.py - delgado)

```python
@app.get("/dashboard/ml_health.json")
async def dashboard_ml_health_json(request: Request):
    if not _verify_dashboard_token(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")

    now = time.time()

    # Check cache
    if _ml_health_cache["data"] and (now - _ml_health_cache["timestamp"]) < _ml_health_cache["ttl"]:
        cached_data = _ml_health_cache["data"]
        return {
            "generated_at": cached_data["generated_at"],
            "cached": True,
            "cache_age_seconds": round(now - _ml_health_cache["timestamp"], 1),
            "health": cached_data["health"],
            "data": cached_data["data"],
        }

    # Build fresh
    from app.ml.health import build_ml_health_data
    async with AsyncSessionLocal() as session:
        result = await build_ml_health_data(session)

    # Update cache + snapshot for regressions
    _ml_health_cache["previous_snapshot"] = _ml_health_cache["data"]
    _ml_health_cache["previous_snapshot_at"] = _ml_health_cache["timestamp"]
    _ml_health_cache["data"] = result
    _ml_health_cache["timestamp"] = now

    return {
        "generated_at": result["generated_at"],
        "cached": False,
        "cache_age_seconds": None,
        "health": result["health"],
        "data": result["data"],
    }
```

### 6.4 Core Logic (app/ml/health.py)

```python
"""
ML Health Dashboard - Core Logic

ATI v1.1: Queries SOTA stats coverage + TITAN coverage por season/league,
freshness con age_hours_now, fail-soft por sección.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Season definitions (fixed ranges)
SEASONS = {
    "23/24": ("2023-08-01", "2024-08-01"),
    "24/25": ("2024-08-01", "2025-08-01"),
    "25/26": ("2025-08-01", "2026-08-01"),
}

TOP_LEAGUES = [140, 39, 135, 78, 61]  # La Liga, PL, Serie A, Bundesliga, Ligue 1


async def build_ml_health_data(session: AsyncSession) -> dict:
    """Build complete ML health data with fail-soft per section."""

    degraded_sections = []
    data = {}

    # 1. SOTA Stats Coverage (P0 - causa raíz)
    data["sota_stats_coverage"] = await _safe_query(
        "sota_stats_coverage",
        lambda: _query_sota_stats_coverage(session),
        {"by_season": {}, "by_league": [], "status": "unknown"},
        degraded_sections,
    )

    # 2. TITAN Coverage por season
    data["titan_coverage"] = await _safe_query(
        "titan_coverage",
        lambda: _query_titan_coverage(session),
        {"by_season": {}, "by_league": [], "status": "unknown"},
        degraded_sections,
    )

    # 3. PIT Compliance
    data["pit_compliance"] = await _safe_query(
        "pit_compliance",
        lambda: _query_pit_compliance(session),
        {"total_rows": 0, "violations": 0, "violation_pct": 0, "status": "unknown"},
        degraded_sections,
    )

    # 4. Freshness (age_hours_now + lead_time_hours)
    data["freshness"] = await _safe_query(
        "freshness",
        lambda: _query_freshness(session),
        {"age_hours_now": {}, "lead_time_hours": {}, "status": "unknown"},
        degraded_sections,
    )

    # 5. Prediction Confidence
    data["prediction_confidence"] = await _safe_query(
        "prediction_confidence",
        lambda: _query_prediction_confidence(session),
        {"entropy": {}, "tier_distribution": {}, "sample_n": 0, "window_days": 30},
        degraded_sections,
    )

    # 6. Top Regressions (placeholder hasta tener baseline)
    data["top_regressions"] = {
        "status": "not_ready",
        "note": "Requires baseline snapshot - will compare current vs previous window after 48h of data",
    }

    # 7. Compute Fuel Gauge
    data["fuel_gauge"] = _compute_fuel_gauge(data, degraded_sections)

    # 8. Overall health
    health = "ok"
    if degraded_sections:
        health = "partial"
    if data["fuel_gauge"]["status"] == "error":
        health = "error"

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "health": health,
        "data": data,
    }


async def _safe_query(
    section_name: str,
    query_fn,
    default_value: dict,
    degraded_sections: list,
) -> dict:
    """Execute query with fail-soft: on error, return default + mark degraded."""
    try:
        return await query_fn()
    except Exception as e:
        logger.warning(f"ML Health section '{section_name}' degraded: {e}")
        degraded_sections.append(section_name)
        return {**default_value, "_degraded": True, "_error": str(e)[:100]}
```

### 6.5 Fail-Soft: Health Level a Root

```python
# En la respuesta final:
{
    "health": "ok|partial|error",  # partial si alguna sección degraded
    "data": {
        "fuel_gauge": {
            "reasons": ["Degraded section: freshness", ...]  # incluye secciones caídas
        }
    }
}
```

---

## 7. Archivos a Modificar

| Archivo | Cambios |
|---------|---------|
| `app/ml/health.py` | **NUEVO**: Core queries (~200 líneas) |
| `app/main.py` | Endpoint + cache (~50 líneas) |

---

## 8. Performance

| Query | Estimación | Mitigación |
|-------|------------|------------|
| SOTA stats coverage | ~100ms | GROUP BY season, índice en date |
| TITAN coverage | ~50ms | Índice en kickoff_utc |
| PIT compliance | ~30ms | Simple COUNT |
| Freshness | ~80ms | PERCENTILE_CONT |
| Prediction entropy | ~150ms | Limitar a 30 días |
| **Total sin cache** | ~500ms | Cache 60s mitiga |
| **Con cache** | <5ms | 99% de requests |

---

## 9. Orden de Implementación

```
[1] Crear app/ml/health.py                   10 min
[2] _query_sota_stats_coverage()             25 min
[3] _query_titan_coverage()                  20 min
[4] _query_pit_compliance()                  10 min
[5] _query_freshness() (age + lead_time)     25 min
[6] _query_prediction_confidence()           15 min
[7] _compute_fuel_gauge() (ATI v1.1)         20 min
[8] Endpoint en main.py + cache              15 min
[9] Tests manuales con curl                  20 min
────────────────────────────────────────────
Total estimado:                             ~2.5-3 horas
```

---

## 10. Verificación

```bash
# Test básico
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ml_health.json" | jq

# Verificar fuel gauge
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ml_health.json" | jq '.data.fuel_gauge'

# Verificar SOTA stats coverage (causa raíz)
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ml_health.json" | jq '.data.sota_stats_coverage'

# Verificar freshness age_hours_now
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ml_health.json" | jq '.data.freshness.age_hours_now'

# Verificar health level
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ml_health.json" | jq '.health'

# Test fail-soft (forzar error en una query viendo partial)
curl -s -H "X-Dashboard-Token: $TOKEN" \
  "https://web-production-f2de9.up.railway.app/dashboard/ml_health.json" | jq 'select(.health == "partial")'
```

---

## 11. Criterios de Aceptación (ATI v1.1)

- [ ] Endpoint disponible y responde <2s
- [ ] `health` field a nivel root: `ok|partial|error`
- [ ] `sota_stats_coverage` por season (23/24, 24/25, 25/26) y por liga
- [ ] `titan_coverage` por season y por liga
- [ ] `freshness.age_hours_now` (early warning real)
- [ ] `fuel_gauge` usa paths correctos del data dict
- [ ] `top_regressions` marcado como `not_ready` (no `[]` vacío)
- [ ] Fail-soft: si una query falla, `health: "partial"` y sección marcada `_degraded`
- [ ] Cache funciona (60s TTL)

---

## 12. Cambios ATI Incorporados

| # | Cambio Obligatorio | Estado |
|---|-------------------|--------|
| 1 | SOTA stats coverage por season/liga | ✅ Incorporado |
| 2 | Segmentación por temporada (no solo 7 días) | ✅ Incorporado |
| 3 | Freshness age_hours_now (early warning) | ✅ Incorporado |
| 4 | Mismatch de claves en fuel_gauge | ✅ Corregido |
| 5 | top_regressions marcado not_ready | ✅ Incorporado |
| 6 | Módulo separado app/ml/health.py | ✅ Incorporado |
| 7 | health: partial a nivel root | ✅ Incorporado |

---

## Aprobación

| Rol | Nombre | Fecha | Estado |
|-----|--------|-------|--------|
| Auditor TITAN | ATI | 2026-01-26 | ✅ **APROBADO CON CAMBIOS** |
| Owner | David | - | PENDIENTE |
