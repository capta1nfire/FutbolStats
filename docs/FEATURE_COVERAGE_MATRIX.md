## Feature Coverage Matrix (SOTA) — Contrato y Semántica

### Objetivo
Visualizar la **cobertura real** (no-NULL) de features del modelo por **liga** y **ventana temporal** para decidir qué ligas incluir en entrenamiento (evitar imputar faltantes con 0 y meter ruido).

### Endpoint
- **Path**: `GET /dashboard/feature-coverage.json`
- **Auth**: `X-Dashboard-Token`
- **Cache**: server-side TTL (actualmente 30 min)

### Ventanas (UTC)
- **23/24**: 2023-08-01 ≤ kickoff/date < 2024-08-01
- **24/25**: 2024-08-01 ≤ kickoff/date < 2025-08-01
- **total**: combinación de ambas ventanas

### Semántica de cobertura (CRÍTICO)
Cada celda reporta:
\[
\text{pct} = 100 \cdot \frac{n}{\text{denominator}}
\]

Donde:
- **n**: cantidad de matches donde el feature es **NO-NULL**
- **denominator depende del tier**:
  - **Tier 1 (PROD)**: `denominator = matches_total_ft` (partidos FT en `matches`)
  - **Tiers TITAN (1b/1c/1d)**: `denominator = matches_total_titan` (filas en `titan.feature_matrix`)

Esto evita mezclar denominadores (FT vs feature_matrix) y hace que `total` sea consistente.

### Contrato de respuesta (shape)

```json
{
  "generated_at": "ISO timestamp",
  "cached": true,
  "cache_age_seconds": 123.4,
  "data": {
    "windows": [
      { "key": "23/24", "from": "2023-08-01", "to": "2024-07-31" },
      { "key": "24/25", "from": "2024-08-01", "to": "2025-07-31" }
    ],
    "tiers": [
      { "id": "tier1", "label": "[PROD] Tier 1 - Core", "badge": "PROD" },
      { "id": "tier1b", "label": "[TITAN] Tier 1b - xG", "badge": "TITAN" },
      { "id": "tier1c", "label": "[TITAN] Tier 1c - Lineup", "badge": "TITAN" },
      { "id": "tier1d", "label": "[TITAN] Tier 1d - XI Depth", "badge": "TITAN" }
    ],
    "features": [
      { "key": "home_goals_scored_avg", "tier_id": "tier1", "badge": "PROD", "source": "public.matches" }
    ],
    "leagues": [
      { "league_id": 39, "name": "Premier League" }
    ],
    "league_summaries": {
      "39": {
        "23/24": { "matches_total_ft": 380, "matches_total_titan": 0, "avg_pct": 33.3 },
        "24/25": { "matches_total_ft": 380, "matches_total_titan": 0, "avg_pct": 31.9 },
        "total": { "matches_total_ft": 760, "matches_total_titan": 0, "avg_pct": 32.6 }
      }
    },
    "coverage": {
      "home_goals_scored_avg": {
        "39": {
          "23/24": { "pct": 100.0, "n": 380 },
          "24/25": { "pct": 100.0, "n": 380 },
          "total": { "pct": 100.0, "n": 760 }
        }
      }
    }
  }
}
```

### Notas operacionales
- Si `matches_total_titan = 0`, los % TITAN se reportan como **0%** (evita división por cero). Esto típicamente indica que `titan.feature_matrix` aún no tiene filas para esa liga/ventana.
- Para Tier 1 shots/corners, la fuente actual es `matches.stats` (JSON). Si el JSON está vacío, la cobertura real será ~0%.

