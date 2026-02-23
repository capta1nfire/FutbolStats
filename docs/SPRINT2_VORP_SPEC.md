# Sprint 2 — VORP Feature Engineering Tech Spec

**Fecha**: 2026-02-23
**Autor**: GDT Deep Alpha Roadmap
**Prerequisito**: Sprint 1 completado (v1.3.0-latam-first, snapshot_id=19)
**Status**: PROPOSAL — pendiente aprobación ABE/GDT

---

## 1. Contexto y Lecciones de Experimentos Previos

### PTS Delta (KILLED — 2026-02-19)
- **Resultado**: CV SIGNAL 0/7 ligas
- **Causa raíz**: Rating de Sofascore es proxy demasiado ruidoso (σ ~0.8 sobre μ ~6.5). La varianza rating-a-rating dentro del mismo equipo domina la señal entre equipos.
- **Lección**: No usar rating absoluto como feature. Usar **rankings relativos** (percentil dentro del equipo/liga) o **deltas de calidad de reemplazo**.

### O-VORP (evaluate_o_vorp.py — Inconclusivo)
- Walk-forward con OLS expanding-window + XGBoost
- Orthogonaliza injury shock respecto a fortaleza del equipo
- **Problema**: Requiere historial profundo de Sofascore (solo Nov 2025+, ~4 meses)
- **Potencial**: La ortogonalización es correcta conceptualmente pero los datos son insuficientes

### Infraestructura Existente (REUTILIZABLE)
| Componente | Ubicación | Estado |
|------------|-----------|--------|
| PTS batch computation | `app/features/engineering.py:732-799` | Producción |
| VORP P25 priors | `app/features/engineering.py:802-877` | Producción |
| Cascade PTS (4 niveles) | `app/features/engineering.py:880-952` | Producción |
| Talent delta (XI real vs expected) | `app/features/engineering.py:1015-1095` | Producción |
| Player ID mapping (Hungarian) | `player_id_mapping` table | 85.4% active |
| match_player_stats | 1.1M+ rows, 3+ años | Producción |

---

## 2. Datos Disponibles: match_player_stats

### Schema Útil para Features
```
rating          — Sofascore/API-Football rating (6.5 ± 0.8)
minutes         — Minutos jugados (0-90)
position        — G, D, M, F
is_substitute   — Titular vs suplente
goals, assists  — Producción ofensiva
passes_key      — Pases clave (proxy creatividad)
tackles, interceptions, blocks — Producción defensiva
duels_won, duels_total — Proxy presión física
dribbles_success, dribbles_attempts — Proxy habilidad individual
```

### Cobertura por Liga LATAM
| Liga | Rows | Matches | Rango |
|------|------|---------|-------|
| Argentina (128) | 61,802 | 1,352 | 2023-01 a 2026-02 |
| Colombia (239) | 50,577 | 1,389 | 2023-01 a 2026-02 |
| Brazil (71) | ~45K+ | ~1,100+ | 2023-04 a 2026-02 |
| Chile (265) | ~25K+ | ~700+ | 2023+ |
| Ecuador (242) | ~20K+ | ~600+ | 2023+ |

---

## 3. Hipótesis de Features (5 candidatos)

### H1: Squad Depth Index (SDI)
**Idea**: Equipos con más profundidad de plantel absorben lesiones mejor.
```
SDI = mean(rating_starter) - P25(rating_squad, same_position)
```
- Alto SDI = poca distancia entre titulares y suplentes = resiliente
- Bajo SDI = gran caída de calidad al rotar = vulnerable
- **Feature**: `home_sdi`, `away_sdi`, `sdi_diff`
- **Ventaja**: No depende de lesiones actuales (siempre computable)
- **Fuente**: Rolling window 10 matches de match_player_stats

### H2: Cumulative Missing Talent (CMT)
**Idea**: Cuánto talento falta por lesiones activas, medido en calidad de reemplazo.
```
CMT = Σ (starter_rating - replacement_rating) para cada lesionado
    = talent_delta (ya existe en engineering.py)
```
- **Feature**: `home_cmt`, `away_cmt`, `cmt_diff`
- **Ventaja**: Ya implementado parcialmente en `compute_team_talent_delta()`
- **Limitación**: Depende de cobertura de `player_injuries` (desigual por liga)

### H3: Fatigue Exposure Index (FEI)
**Idea**: Equipos con copa/internacional tienen más carga. Minutos acumulados en últimos 14 días predicen fatiga.
```
FEI = mean(minutes_last_14d) para titulares esperados
```
- **Feature**: `home_fei`, `away_fei`, `fei_diff`
- **Ventaja**: 100% computable desde match_player_stats (no necesita injuries)
- **Relevancia LATAM**: Copa Libertadores/Sudamericana crea fatiga real
- **Fuente**: `SUM(minutes) WHERE match_date BETWEEN (asof-14, asof)` por jugador, luego promedio del XI esperado

### H4: Lineup Stability Score (LSS)
**Idea**: Equipos que rotan mucho tienen menor cohesión táctica.
```
LSS = jaccard(XI_last_match, XI_this_match) promedio últimos 5 partidos
```
- **Feature**: `home_lss`, `away_lss`, `lss_diff`
- **Ventaja**: 100% computable desde match_player_stats (quién empezó/suplente)
- **Señal**: Rotación forzada (lesiones) ≠ rotación táctica (entrenador). Ambas afectan cohesión.

### H5: Positional Dominance Score (PDS)
**Idea**: Ventaja posicional relativa en duelos directos.
```
PDS_attack = mean(FWD_rating_home) - mean(DEF_rating_away)
PDS_defense = mean(DEF_rating_home) - mean(FWD_rating_away)
```
- **Feature**: `pds_attack_diff`, `pds_defense_diff`
- **Ventaja**: Captura matchup asimétrico (ataque fuerte vs defensa débil)
- **Riesgo**: Misma debilidad que PTS Delta (ratings ruidosos). Mitigación: usar P75 en vez de mean.

---

## 4. Diseño del Experimento

### Filosofía: Alpha Lab LATAM Sprint 2
Mismo framework que Sprint 1 (`scripts/alpha_lab_latam.py`), con VORP features como tratamientos adicionales.

### Script: `scripts/alpha_lab_latam_s2.py`

### Tratamientos

| Treatment | Base | +Features | Propósito |
|-----------|------|-----------|-----------|
| Tcontrol | v1.3.0 baseline (18f) | — | Sprint 1 winner |
| V1 | 18f | +SDI (3f) | Squad depth signal |
| V2 | 18f | +CMT (3f) | Missing talent signal |
| V3 | 18f | +FEI (3f) | Fatigue signal |
| V4 | 18f | +LSS (3f) | Stability signal |
| V5 | 18f | +SDI+FEI (6f) | Depth + Fatigue combo |
| V6 | 18f | +ALL (12f) | Kitchen sink (overfit detector) |

### Split
- Mismo split temporal 70/10/20 que Sprint 1 (reutilizar)
- Test set IDÉNTICO a Sprint 1 para comparabilidad directa
- Diferencia: features VORP se computan dinámicamente desde match_player_stats

### Métricas
- Brier Score global y per-liga
- Skill Score vs market (consistencia con Sprint 1)
- Bootstrap paired delta vs Tcontrol
- Feature importance ranking (¿VORP features aparecen en top 10?)

### Criterio de Éxito
1. **Mínimo**: Al menos 1 treatment con Skill Score > Tcontrol en ≥ 3 ligas
2. **Target**: Bootstrap CI del delta no cruza 0 (estadísticamente significativo)
3. **Kill switch**: Si V6 (kitchen sink) es el mejor → sobreajuste, no hay señal real

---

## 5. Feature Engineering Pipeline

### Pre-computation (batch, offline)
```python
# Para cada match en lab CSVs:
# 1. Query match_player_stats WHERE match_date < asof (PIT safe)
# 2. Compute per-team aggregates (last 10 matches window)
# 3. Store as columns in augmented CSV

def compute_vorp_features(match_id, home_team_id, away_team_id, asof_date, session):
    """Compute all VORP features for a single match (PIT safe)."""

    # SDI: Squad Depth Index
    home_sdi = _squad_depth_index(home_team_id, asof_date, session)
    away_sdi = _squad_depth_index(away_team_id, asof_date, session)

    # FEI: Fatigue Exposure Index
    home_fei = _fatigue_index(home_team_id, asof_date, session)
    away_fei = _fatigue_index(away_team_id, asof_date, session)

    # LSS: Lineup Stability Score
    home_lss = _lineup_stability(home_team_id, asof_date, session)
    away_lss = _lineup_stability(away_team_id, asof_date, session)

    return {
        "home_sdi": home_sdi, "away_sdi": away_sdi,
        "sdi_diff": home_sdi - away_sdi,
        "home_fei": home_fei, "away_fei": away_fei,
        "fei_diff": home_fei - away_fei,
        "home_lss": home_lss, "away_lss": away_lss,
        "lss_diff": home_lss - away_lss,
    }
```

### SQL Patterns (PIT Safe)

```sql
-- SDI: Squad Depth Index (rolling 10 matches)
WITH recent_matches AS (
    SELECT DISTINCT match_id
    FROM match_player_stats
    WHERE team_id = :team_id AND match_date < :asof
    ORDER BY match_date DESC LIMIT 10
),
player_ratings AS (
    SELECT player_external_id, position,
           AVG(rating) AS avg_rating,
           SUM(minutes) AS total_minutes,
           BOOL_OR(NOT is_substitute) AS ever_started
    FROM match_player_stats
    WHERE match_id IN (SELECT match_id FROM recent_matches)
      AND team_id = :team_id AND rating IS NOT NULL
    GROUP BY player_external_id, position
)
SELECT
    AVG(avg_rating) FILTER (WHERE ever_started) AS mean_starter_rating,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY avg_rating) AS p25_squad_rating
FROM player_ratings;
-- SDI = mean_starter_rating - p25_squad_rating

-- FEI: Fatigue Exposure (14 días)
SELECT AVG(minutes) AS avg_minutes_14d
FROM match_player_stats
WHERE team_id = :team_id
  AND match_date BETWEEN :asof - INTERVAL '14 days' AND :asof
  AND NOT is_substitute;  -- solo titulares

-- LSS: Lineup Stability (Jaccard últimos 5 matches)
-- Requiere computación en Python (set intersection de player_ids por match)
```

---

## 6. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| VORP features sin señal (como PTS Delta) | ALTA | Sprint 2 = null result | Kill V6 test + early abort |
| Rating noise domina features | ALTA | Mediocre importance | Usar rankings/percentiles, no ratings absolutos |
| PIT leakage en feature computation | MEDIA | Resultados inválidos | Strict `match_date < asof` en toda query |
| Cobertura desigual por liga | MEDIA | Bias en resultados | Reportar N_coverage por liga |
| Compute time prohibitivo (1.1M rows × 11 ligas) | MEDIA | Sprint se alarga | Pre-materializar en batch CSV |

---

## 7. Entregables

1. **Script**: `scripts/alpha_lab_latam_s2.py` — Feature computation + lab experiment
2. **CSV**: `scripts/output/lab/vorp_features_{lid}.csv` — Features pre-computados (PIT safe)
3. **Resultado**: Skill Score matrix con treatments V1-V6 vs Tcontrol
4. **Decisión**: Promote/Kill cada feature family

---

## 8. Dependencias

- [x] Sprint 1 completado (v1.3.0-latam-first, snapshot_id=19)
- [x] match_player_stats con cobertura 3+ años
- [x] VORP infrastructure en `app/features/engineering.py`
- [ ] ABE/GDT approval de este spec
- [ ] Estimación de compute time para pre-materialización

---

## 9. Timeline Estimado

| Fase | Descripción | Estimación |
|------|-------------|------------|
| F1 | Pre-materializar VORP features en CSVs | 1-2 sesiones |
| F2 | Implementar alpha_lab_latam_s2.py | 1 sesión |
| F3 | Run experiment + análisis | 1 sesión |
| F4 | Decisión promote/kill | Inmediato post-F3 |

**Total**: ~3-4 sesiones de trabajo
