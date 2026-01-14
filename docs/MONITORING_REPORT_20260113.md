# Informe de Monitoreo 24h - 13 Enero 2026

## Ventana de Observacion
- **Inicio**: 2026-01-13 02:38 UTC
- **Fin**: 2026-01-14 02:38 UTC
- **Generado**: 2026-01-14 02:38 UTC

---

## Resumen Ejecutivo

| Componente | Estado | Notas |
|------------|--------|-------|
| predictions_health | OK | 100% cobertura NS y FT |
| fastpath_health | OK | 29 ticks, 0 errores |
| LLM Narratives | OK | 6/6 ok, 0 rejected |
| Data Quality | OK | 0 quarantined, 0 tainted |
| API Budget | OK | 236/75000 requests (0.3%) |

**Alertas disparadas**: 0

---

## Detalle por Componente

### 1. Predictions Health

```json
{
  "status": "ok",
  "ns_matches_next_48h": 20,
  "ns_coverage_pct": 100.0,
  "ft_matches_last_48h": 10,
  "ft_coverage_pct": 100.0,
  "last_prediction_saved_at": "2026-01-12T07:00:15",
  "hours_since_last_prediction": 43.6
}
```

**Observaciones**:
- Cobertura de predicciones al 100% para partidos NS y FT
- 43.6h desde ultima prediccion guardada (normal, se guardan bajo demanda)

### 2. Fastpath Health

```json
{
  "status": "ok",
  "enabled": true,
  "last_tick_at": "2026-01-14T02:37:54",
  "minutes_since_tick": 0.9,
  "ticks_total": 29,
  "ticks_with_activity": 0,
  "last_60m": {
    "ok": 0,
    "error": 0,
    "error_rate_pct": 0.0
  },
  "pending_ready": 0
}
```

**Observaciones**:
- Scheduler corriendo normalmente (tick cada 2 min)
- 0 errores en ultimos 60 minutos
- Sin jobs pendientes

### 3. LLM Narratives (ultimas 24h)

| Fecha | Total | OK | Rejected | Error | Pending |
|-------|-------|-----|----------|-------|---------|
| 2026-01-13 | 6 | 6 | 0 | 0 | 0 |
| 2026-01-12 | 4 | 4 | 0 | 0 | 0 |

**Detalle de generaciones**:

| Match ID | Status | Exec (ms) | Tokens In | Tokens Out | Error Code |
|----------|--------|-----------|-----------|------------|------------|
| 70508 | ok | 21,066 | 2,618 | 423 | - |
| 70509 | ok | 22,817 | 2,837 | 454 | - |
| 70507 | ok | 27,553 | 2,702 | 437 | - |
| 6647 | ok | 28,226 | 2,687 | 446 | - |
| 6648 | ok | 24,766 | 2,699 | 500 | - |
| 6646 | ok | 55,807 | 2,660 | 1,200 | schema_invalid* |

*Nota: Match 6646 tuvo un intento con schema_invalid pero se recupero en retry, status final OK.

**Metricas agregadas**:
- Tiempo promedio ejecucion: ~30s
- Tokens promedio entrada: ~2,700
- Tokens promedio salida: ~480
- Tasa de exito: 100%
- Claim rejections: 0

### 4. Data Quality / Telemetry

```json
{
  "status": "OK",
  "summary": {
    "quarantined_odds_24h": 0,
    "tainted_matches_24h": 0,
    "unmapped_entities_24h": 0
  }
}
```

**Observaciones**:
- Sin odds en cuarentena
- Sin partidos marcados como tainted
- Sin entidades sin mapear

### 5. Stats Backfill

```json
{
  "finished_72h_with_stats": 36,
  "finished_72h_missing_stats": 0
}
```

**Observaciones**:
- 100% de partidos FT tienen stats completos

### 6. API Budget

```json
{
  "status": "ok",
  "plan": "Ultra",
  "requests_today": 236,
  "requests_limit": 75000,
  "requests_remaining": 74764
}
```

**Observaciones**:
- Uso del 0.3% del budget diario
- Sin riesgo de throttling

### 7. Model Performance

```json
{
  "status": "gray",
  "status_reason": "No report available yet"
}
```

**Accuracy ultimas 48h**:

| Fecha | Partidos | Correctos | Incorrectos | Accuracy |
|-------|----------|-----------|-------------|----------|
| 2026-01-13 | 6 | 3 | 3 | 50.0% |
| 2026-01-12 | 4 | 4 | 0 | 100.0% |

**Observaciones**:
- Performance mixto el 13 enero (50% accuracy)
- Muestra pequena (6 partidos) - no significativo estadisticamente

---

## PIT Snapshots (Point-In-Time)

```json
{
  "live_60m": 0,
  "live_24h": 9,
  "pit_snapshots_30d": 91,
  "target_pit_snapshots_30d": 100,
  "baseline_coverage_pct": 74.7
}
```

**Ultimos snapshots**:
- Mexico Liga MX (262): 3 snapshots
- Copa del Rey (143): 3 snapshots
- Bundesliga (78): 3 snapshots

---

## Incidentes

**Total incidentes**: 0

No se detectaron incidentes criticos durante la ventana de observacion.

---

## Acciones Recomendadas

1. **Sin cambios requeridos** - Sistema estable
2. **Monitorear**: Match 6646 tuvo retry por schema_invalid (55s exec time vs ~25s promedio). Revisar si patron se repite.
3. **Opcional**: Considerar alerta para exec_time > 45s como warning temprano

---

## Capturas de Evidencia

### OPS Dashboard JSON
Guardado en: `logs/ops_snapshot_20260114_0238.json` (ver comando curl en CLAUDE.md)

### Grafana Alerting
- Panel: Alerting > "Fired alerts" (24h)
- Resultado: 0 alertas disparadas

---

## Conclusiones

El sistema opero de forma estable durante las ultimas 24 horas:
- Cobertura de predicciones: 100%
- LLM narratives: 100% exito (6/6)
- Data quality: Sin anomalias
- Fastpath: Operando sin errores
- API budget: Uso minimo

**Estado**: CLOSED - Sin incidentes, sistema saludable.
