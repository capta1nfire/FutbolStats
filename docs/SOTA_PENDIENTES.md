# SOTA - Pendientes de Implementación

Estado actual de la arquitectura SOTA y tareas pendientes.

**Última actualización**: 2026-01-23

---

## Fase 1: ETL + Features (EN PROGRESO)

### Completado

| Tarea | Archivo(s) | Notas |
|-------|-----------|-------|
| Migración 029: tablas SOTA | `scripts/migrations/029_add_sota_core_tables.py` | 5 tablas creadas |
| ETL Understat provider (stub) | `app/etl/understat_provider.py` | Mock data, requiere scraper real |
| ETL Open-Meteo provider | `app/etl/open_meteo_provider.py` | Funcional con API real |
| Script: populate_venue_geo | `scripts/populate_venue_geo.py` | 242 venues geocodificados |
| Script: populate_team_home_city_profile | `scripts/populate_team_home_city_profile.py` | 164 perfiles con timezone |
| Script: capture_weather_prekickoff | `scripts/capture_weather_prekickoff.py` | Funcional |
| Script: backfill_understat_ft | `scripts/backfill_understat_ft.py` | Funcional (con mock data) |
| Script: populate_match_external_refs_understat | `scripts/populate_match_external_refs_understat.py` | Mock IDs |
| FeatureEngineer: Understat features | `app/features/engineering.py` | 12 features implementadas |
| FeatureEngineer: Weather/Bio features | `app/features/engineering.py` | 11 features implementadas |
| Tests PIT | `tests/test_feature_engineering_pit.py` | Point-in-time validation |

### Pendiente Fase 1

| Tarea | Prioridad | Descripción | Dependencias |
|-------|-----------|-------------|--------------|
| **Understat scraper real** | ALTA | Implementar scraping real de Understat (xG/xPTS por partido) | Requiere análisis de rate limits y ToS |
| **Climate normals reales** | MEDIA | Poblar `climate_normals_by_month` con Open-Meteo Historical API | Requiere ~10 API calls por ciudad |
| **Imputación weather por climatología** | BAJA | Cambiar default 15°C/60% a usar climatología del venue | Depende de climate normals |
| **Scheduler jobs SOTA** | MEDIA | Agregar jobs para ETL automático (venue_geo, weather, understat) | Definir frecuencia |

---

## Fase 2: Sofascore XI (NO INICIADO)

| Tarea | Prioridad | Descripción |
|-------|-----------|-------------|
| Sofascore provider | ALTA | Scraper para XI titulares y ratings pre-partido |
| Tabla `match_sofascore_xi` | ALTA | Migración para almacenar XI snapshot |
| Features XI en FeatureEngineer | ALTA | `xi_weighted_home/away`, `xi_p10/p50/p90`, `xi_weaklink`, `xi_std` |
| Flag `xi_missing` | MEDIA | Degradación controlada si no hay XI |

---

## Fase 3: Market Microstructure (NO INICIADO)

| Tarea | Prioridad | Descripción |
|-------|-----------|-------------|
| Tabla `odds_history_snapshot` | MEDIA | Snapshots de odds en momentos clave (open, lineup, close) |
| Features odds movement | MEDIA | `odds_log_move_open_to_close_*`, `steam_move_flag` |
| Flags `odds_*_missing` | BAJA | Degradación controlada |

---

## Fase 4: ML Integration (NO INICIADO)

| Tarea | Prioridad | Descripción |
|-------|-----------|-------------|
| Reentrenamiento con SOTA features | ALTA | Incluir xG, weather, bio en modelo |
| Evaluación PIT estricta | ALTA | Backtest con point-in-time enforcement |
| Feature importance analysis | MEDIA | Identificar features más predictivas |
| A/B test vs baseline | ALTA | Comparar accuracy con/sin SOTA features |

---

## Notas Técnicas

### Imputaciones actuales (Prompt 6)

| Feature | Default si falta | Flag |
|---------|------------------|------|
| Understat (xG, justice) | 0.0 | `understat_missing=1` |
| Weather temp | 15.0°C | `weather_missing=1` |
| Weather humidity | 60.0% | `weather_missing=1` |
| Weather wind | 3.0 m/s | `weather_missing=1` |
| Weather precip | 0.0 mm | `weather_missing=1` |
| Thermal shock | 0.0 | (sin flag adicional) |
| TZ shift | 0.0 | (sin flag adicional) |
| Circadian disruption | 0.0 | (sin flag adicional) |

### Constantes SOTA (engineering.py)

```python
JUSTICE_SHRINKAGE_K = 10        # k en rho = n/(n+k)
JUSTICE_EPSILON = 0.01          # epsilon en sqrt(XG + eps)
BIO_CIRCADIAN_WEIGHT = 0.6      # peso circadian en bio_disruption
BIO_TZ_WEIGHT = 0.4             # peso tz_shift en bio_disruption
CIRCADIAN_HISTORY_MATCHES = 20  # partidos para calcular hora típica
```

---

## Referencias

- Arquitectura: `docs/ARCHITECTURE_SOTA.md`
- Diccionario features: `docs/FEATURE_DICTIONARY_SOTA.md`
- Tests PIT: `tests/test_feature_engineering_pit.py`
