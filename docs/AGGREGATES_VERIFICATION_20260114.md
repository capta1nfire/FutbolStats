# Verificación Sistema de Aggregates - 2026-01-14

## Estado de Implementación

### Commits Desplegados
```
efb2815 feat(api): add manual aggregates refresh and status endpoints
7742347 fix(scheduler): add session to daily_refresh_aggregates job
e164b70 feat(scheduler): add daily aggregates refresh job at 06:30 UTC
9138a31 feat(aggregates): add league baselines and team profiles for narratives
```

---

## 1. Guardrails Verificados

### 1.1 Constantes de Umbral
```python
# app/aggregates/service.py:20-21
MIN_SAMPLE_MATCHES = 5   # Mínimo partidos para stats válidas
HIGH_CONFIDENCE_MATCHES = 10  # Partidos para rank_confidence="high"
```

### 1.2 Tabla `league_season_baselines`
| Campo | Propósito |
|-------|-----------|
| `sample_n_matches` | Número de partidos usados para calcular promedios |

**Guardrail:** Solo se crea baseline si `len(matches) >= MIN_SAMPLE_MATCHES` (service.py:68-71)

### 1.3 Tabla `league_team_profiles`
| Campo | Propósito | Cálculo |
|-------|-----------|---------|
| `matches_played` | Partidos del equipo en la muestra | Directo |
| `min_sample_ok` | Flag: suficientes partidos | `n >= 5` |
| `rank_confidence` | Confianza en rankings | `"high"` si `n >= 10`, else `"low"` |

**Guardrail:** Rankings solo se asignan si `min_sample_ok=True` (service.py:426-431)

### 1.4 `context_usable` en derived_facts
```python
# app/llm/derived_facts.py:927-932
context_usable = False
if home_team_context and away_team_context:
    home_ok = home_team_context.get("min_sample_ok", False)
    away_ok = away_team_context.get("min_sample_ok", False)
    context_usable = home_ok and away_ok
```

**Guardrail:** `relative_context` solo se construye si `context_usable=True` (derived_facts.py:936)

---

## 2. Flujo de Datos

```
[Daily 06:00 UTC] daily_sync_results
        ↓
[Daily 06:30 UTC] daily_refresh_aggregates
        ↓
     ┌──────────────────────────────────────┐
     │  Para cada liga con >= 5 partidos:   │
     │  1. Crear/actualizar baseline        │
     │  2. Para cada equipo con >= 5 partidos:│
     │     - Calcular rates y rankings      │
     │     - Guardar con min_sample_ok flag │
     └──────────────────────────────────────┘
        ↓
[Fast-path narratives] build_derived_facts()
        ↓
     ┌──────────────────────────────────────┐
     │  Si context_usable=True:             │
     │  - Incluir relative_context          │
     │  - LLM puede usar comparaciones      │
     │  Si context_usable=False:            │
     │  - relative_context=null             │
     │  - LLM no hace comparaciones team/league│
     └──────────────────────────────────────┘
```

---

## 3. Endpoints de Verificación

### 3.1 Estado de Aggregates
```bash
GET /aggregates/status?token=<API_KEY>
```
Response:
```json
{
  "status": "ok",
  "baselines_count": <int>,
  "profiles_count": <int>,
  "leagues_with_baselines": <int>,
  "latest_baseline_at": "<timestamp>",
  "latest_profile_at": "<timestamp>"
}
```

### 3.2 Trigger Manual de Refresh
```bash
POST /etl/refresh-aggregates?token=<API_KEY>
```
Response:
```json
{
  "status": "ok",
  "refresh_result": {
    "leagues_processed": <int>,
    "baselines_created": <int>,
    "profiles_created": <int>,
    "errors": []
  },
  "status_before": {...},
  "status_after": {...}
}
```

---

## 4. Checklist de Verificación Post-Deploy

- [ ] Servidor Railway disponible en `/health`
- [ ] `GET /aggregates/status` retorna `baselines_count=0` (pre-backfill)
- [ ] `POST /etl/refresh-aggregates` ejecuta sin errores
- [ ] `GET /aggregates/status` retorna `baselines_count > 0` (post-backfill)
- [ ] Verificar `scripts/verify_aggregates.py` (requiere DB local o conexión remota)
- [ ] Revisar logs del primer job diario (mañana 06:30 UTC)

---

## 5. Notas de Robustez

1. **Inicio de temporada:** Si un equipo tiene < 5 partidos, `min_sample_ok=False` y no se incluye en comparaciones.

2. **Rankings parciales:** Si una liga tiene solo 8 equipos con >= 5 partidos de 20 totales, solo esos 8 tendrán rankings.

3. **Fallback en narrativas:** El código de fastpath usa try/except para contexto - si aggregates falla, narrativas siguen funcionando sin contexto relativo.

4. **Idempotencia:** El refresh usa UPSERT, correrlo múltiples veces no crea duplicados.

---

**Estado:** Pendiente verificación en producción (Railway no disponible al momento de este informe)
**Próximo paso:** Ejecutar backfill manual via `/etl/refresh-aggregates` cuando Railway esté disponible
