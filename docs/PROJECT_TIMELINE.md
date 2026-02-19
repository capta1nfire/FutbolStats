# Timeline del Proyecto Bon Jogo

**Período**: Enero 2 — Febrero 17, 2026

---

## Fase 0: MVP (Ene 2-4)
- **Ene 2**: Nace Bon Jogo MVP — sistema de predicción World Cup + ligas europeas. XGBoost con 14 features, iOS app, value betting engine.
- **Ene 3**: Optimización Optuna de hiperparámetros. Auto-train en startup. Caching.
- **Ene 4**: Post-match audit system. Prediction persistence. Recalibración automática. Odds history tracking. Freeze predictions pre-kickoff. Global Sync real-time. **Modelo 1.0.0 nace oficialmente**.

## Fase 1: Infraestructura Operacional (Ene 5-12)
- **Ene 5**: Narrative insights system (LLM post-match).
- **Ene 6-7**: Lineup-confirmed odds capture. PIT dashboard.
- **Ene 8**: Fixtures coverage audit + backfill (LATAM Pack1, MLS, Brasil). PIT integrity enforcement.
- **Ene 9**: FDUK odds ingestion pipeline. OPS Dashboard (`/dashboard/ops.json`). LATAM Pack2 backfill. PIT evaluation scripts v2. Market benchmark con Brier Score. EU Mid Pack2 + WC Qualifiers.
- **Ene 10**: RunPod/Qwen LLM para narrativas. DB-backed PIT reports.
- **Ene 11**: Fast-path narrativas (5-20 min post-match). Gemini provider con circuit breaker.
- **Ene 12**: Data Quality Telemetry P0. iOS UTC calendar fix. DB-first standings architecture. Team aliases para narrativas. Venues en matches.

## Fase 2: Shadow Mode & Sensor B (Ene 14-15) ⭐
- **Ene 14**: **Shadow Mode (Two-Stage model)** implementado — A/B comparison framework. OPS card para Shadow Mode. **Sensor B (LogReg L2)** — calibration diagnostics system. Predictions safety net job.
- **Ene 15**: **Shadow B y Sensor B oficialmente activos** (desplegados en prod). Team identity override system. Odds sync job dedicado. Rerun prediction system (DB-first gated serving).

> **NOTA CRÍTICA**: Shadow B empezó a emitir predicciones el 15 de enero con 100% de cobertura sobre TODOS los matches, incluyendo ligas LATAM sin temporada activa. Esto fue posible porque el startup catch-up NO aplicaba killswitch.

## Fase 3: Dashboard Next.js (Ene 16-24)
- **Ene 16-17**: Sentry integration. Copa del Rey + ML volume leagues + Belgian/Saudi/MLS. OPS Dashboard HTML Fase 1-2. Live_tick job (10s). **Modelo v1.0.0 commiteado** para Railway.
- **Ene 18**: Corrupted model fix (74→14 features).
- **Ene 19**: **Model Benchmark tile** — primer chart comparativo de modelos. Daily comparison tables. League stats page.
- **Ene 20-21**: **Dashboard Next.js completo** — Phase 0 a 11 en dos días. Overview, Jobs, Incidents, Data Quality, Analytics, Audit Trail, Settings, Predictions, URL State. UniFi design system.
- **Ene 22-24**: Wiring de TODOS los sections al backend real. SOTA hardening (geocoding, weather, Sofascore XI/refs). Calendar date filtering. Football navigation. Admin Panel P0-P2. Alerts bell Grafana. Understat internal API fix.

## Fase 4: TITAN OMNISCIENCE (Ene 25-27)
- **Ene 25**: TITAN Phase 1 infrastructure. Fase 2 Understat xG. Tier 1c Sofascore lineups. Tier 1d XI depth. Fase 3C devig/calibration. TITAN card en Overview. Gate N=500 formal.
- **Ene 26**: Admin panel P2A-P2C (admin_leagues, PATCH, rules_json). Football Navigation P3.1-P3.5.
- **Ene 27**: Sofascore refs backfill (96% coverage). Incidents DB persistence. Team home city cascade.

## Fase 5: Descubrimiento 14 Features & Model A v1.0.1 (Ene 29 — Feb 1) ⭐⭐
- **Ene 29**: Model Benchmark con 4 modelos (Market, Model A, Shadow, Sensor B).
- **Ene 31**: **Fase 0 — Liga-only stats + Kill-switch router**. Kill-switch: mínimo 5 partidos de liga en 90 días por equipo.
- **Feb 1**: **DESCUBRIMIENTO CRÍTICO**: De las 14 features del XGBoost, **solo 4 tenían datos reales**, el resto era NaN. Cobertura de muchas ligas era mediocre. **Caso Exeter** — equipos de ligas secundarias con datos ficticios.
  - **Model A v1.0.1-league-only** entrenado — entrena SOLO con matches de la misma liga.
  - Draw cap policy implementada.
  - Sensor B reentrenado de 66K a 22K matches (data limpia).
  - Shadow dependía del scope de Model A — su reentrenamiento también se vio afectado.
  - EXT-C shadow job creado para evaluación automática.

## Fase 6: Modelos Experimentales EXT (Feb 2-3)
- **Feb 2**: **EXT-A, EXT-B, EXT-C, EXT-D** — 4 variantes experimentales para evaluación en paralelo. EXT-D = league-only retrained. Dashboard + backend expone las 4 variantes.
- **Feb 3**: TITAN tier training scripts. Shadow gating metrics.

## Fase 7: Market Anchor & FotMob xG (Feb 8-9) ⭐
- **Feb 8**: **Market Anchor Policy** (ABE P0) — blending de model probs con de-vigged bookmaker odds para ligas low-signal. Configurable per-league via `LEAGUE_OVERRIDES`. Benchmark Matrix (league × source Brier Skill %). **FotMob xG provider** para 16 ligas non-Understat. FotMob team aliases + historical backfill 2023-2025.
- **Feb 9**: **Shadow automatic retraining pipeline** (ATI P0). Cohort-matched training dataset. FotMob xG para Colombia (2025 Clausura+).

> **⚠️ COMMIT CRÍTICO `bfa7a02` (Feb 9)**: "single-path startup catch-up with all guardrails (ATI P0)" — aplica killswitch a TODOS los paths de predicción, incluyendo startup catch-up que antes lo bypasseaba.
>
> **IMPACTO**: Shadow perdió cobertura de ligas LATAM en temporada temprana:
> - Perú (281): 18 predicciones antes → 0 después
> - Uruguay (268): 8 → 0
> - Venezuela (299): 14 → 0
> - CONMEBOL Libertadores, Brasil, Paraguay también afectados
> - Causa: Equipos con <5 partidos de liga en 90 días
> - Shadow pasó de 100% cobertura a ~80%, perdiendo ~33 matches del Feb 11 en adelante

## Fase 8: Shadow Rebaseline & Feature Lab (Feb 10-13)
- **Feb 10**: Shadow rebaseline — snapshot 5. Brier mejoró de 0.2094 a 0.2078. Feature coverage widget en Team Detail.
- **Feb 12**: Players table + squad sync + match lineups backfill. XI continuity computation + Feature Lab Section Q.
- **Feb 13**: **FotMob EUR xG expansion** — 10,238 matches nuevos (Eredivisie, Primeira Liga, Belgium, Süper Lig, Championship, Saudi). **Saudi OddsPortal backfill** (1,681 matches). **Feature Lab 23 leagues** completo. Feature Lab v3 — Brier decomposition, walk-forward, devig Shin. **Phase 2 Architecture** — Information Asymmetry & Market Microstructure. Phase 2 Sprints 1-4 completos.

## Fase 9: League Router & Family S (Feb 15-17) ⭐⭐
- **Feb 14**: Coverage Map (19 dimensiones por liga/country). Squad tab con player photos. Player stats sync.
- **Feb 15**: **League Router (M3)** — routing de modelo por tier. Photo pipeline Colombia pilot. PTS/VORP engine rewrite. **SSOT canonical cohort** + PIT horizon fix (GDT DEFCON 2).
- **Feb 16**: Photo review pipeline completo (MediaPipe face detection, crop, PhotoRoom, R2). TM injuries for MTV pipeline. Dockerfile para Railway (reemplaza Nixpacks). Team performance charts V1.
- **Feb 17**: **MANDATO D completo**:
  - **Family S V2 engine** entrenado (id=7, Brier 0.1934) para Tier 3 (5 ligas: Eredivisie, Primeira Liga, Belgian Pro, Süper Lig, Chile).
  - 24 features (17 core + 3 odds + 4 MTV de Transfermarkt).
  - **Serving layer**: DB-first overlay en `/predictions/upcoming`, `/predictions/match/{id}`, `/matches/{id}/details`.
  - Skip market-anchor para Family S (evitar doble anchor).
  - Cache bypass cuando Family S activo.
  - **Dashboard**: Family S en Model Benchmark chart (sky blue).
  - **P0 skew fix**: `league_only=True` en ambos training paths.

---

## Métricas Clave al Feb 17

| Modelo | Accuracy | N | Estado |
|--------|----------|---|--------|
| Market (Pinnacle) | ~52.5% | 495 | Referencia |
| Model A (v1.0.1) | ~47.8% | 450 | Producción (baseline) |
| Shadow B (Two-Stage) | ~49% | 462 | Evaluación (cobertura reducida desde Feb 9) |
| Sensor B (LogReg) | ~39.4% | 371 | Diagnóstico only |
| Family S (Tier 3) | TBD | Recién desplegado | Producción (5 ligas) |

## Lecciones Aprendidas

1. **Killswitch + single-path = riesgo oculto**: El commit `bfa7a02` fue correcto en principio (guardrails) pero eliminó predicciones de ligas LATAM en temporada temprana sin notificación.
2. **De 14 features, solo 4 tenían datos**: El modelo 1.0.0 operó semanas con 10 features NaN. XGBoost maneja NaN pero la señal era mínima.
3. **Shadow necesita scope independiente**: Shadow heredó las limitaciones de Model A. Su reentrenamiento post-Feb-1 también se vio afectado.
4. **Market > Model en todas las ligas LATAM**: Feature Lab v2 confirmó que el mercado supera al modelo en todas las ligas. Solo Argentina se acerca. Market Anchor es viable para todas.
5. **xG gap permanente**: Chile y Bolivia tienen CERO xG de cualquier fuente. Limitación permanente de Opta.
