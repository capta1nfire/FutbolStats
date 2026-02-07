# Ejemplos — kimi-auditor

Este archivo contiene ejemplos completos del formato esperado (bueno vs malo).

---

## Ejemplo completo: Bueno vs Malo (caso real estilo Genoa–Napoli)

### MAL (lo que queremos evitar)

```
El modelo tiene un feature gap grave. No usa posición en tabla ni odds
del mercado, lo que causa predicciones incorrectas cuando un equipo top
juega de visitante. Recomiendo agregar season_ppg y table_position como
features y reentrenar. Mientras tanto, hay que monitorear la situación.
```

Problemas: Sin etiquetas. Sin fuente. Causalidad sin datos. "Monitorear" sin métricas. Directiva de implementación.

### BIEN (lo que esperamos)

```
## Auditoría: Feature gap en XGBoost v1.0.0

### 1. PREMISAS VERIFICADAS
- [EVIDENCIA] FEATURE_COLUMNS en app/ml/engine.py (buscar "FEATURE_COLUMNS") define las features del modelo.
  Todas son promedios de stats de partidos recientes (goals, shots, corners,
  rest_days, matches_played) más 5 features diferenciales.
- [EVIDENCIA] models/*_metadata.json del modelo auditado confirma n_features y lista de features consistente con el código.
- [EVIDENCIA] Ningún feature incluye odds, xG, H2H, posición en tabla, ni PPG.

### 2. HALLAZGOS
- [EVIDENCIA] El modelo no consume señal de calidad relativa de equipo a nivel
  temporada. Solo ve promedios recientes.
- [HIPÓTESIS] Esto podría causar divergencias sistemáticas en partidos donde
  el mercado identifica un favorito claro (mkt_fav >= 0.45) y el modelo no.
  Requiere medición cuantitativa sobre tabla predictions para confirmar tasa
  base y severity.

### 3. EVIDENCIA EN CONTRA BUSCADA
- Busqué si abs_strength_gap (feature existente) captura indirectamente la
  diferencia de calidad → Solo mide diferencia de promedios recientes, no
  calidad acumulada en la temporada.
- No pude verificar la tasa de divergencia real ni su impacto en accuracy
  (requiere acceso a predictions + matches con frozen_odds).

### 4. LIMITACIONES
- No tengo acceso a DB para medir N, accuracy, ni Brier del subset divergente.
- No puedo confirmar si el patrón es sistemático o anecdótico con un solo caso.

### 5. RECOMENDACIÓN
- Medir tasa base: SELECT count(*) de predictions WHERE mkt_fav_prob >= 0.45
  AND model_fav != mkt_fav AND gap >= 0.20. Si N > 50 en 30 días, es sistémico.
- Comparar accuracy modelo vs mercado en ese subset.
- Si model_accuracy < 33% (random), confirma degradación y justifica
  exploración de features adicionales (season_ppg, implied_probs).
```

