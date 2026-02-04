# Prompt Definitivo: Implementación Model Benchmark Dinámico

## 🎯 Objetivo
Actualizar el endpoint `/dashboard/model-benchmark` y el componente frontend para soportar **fechas dinámicas según modelos seleccionados**, con reglas de negocio específicas.

---

## 📋 Contexto de Negocio

### Historial de Modelos
- **4 de enero 2026**: Primeras predicciones (Model A vs Market) - 4 partidos
- **10 de enero 2026**: Ajustes importantes a Model A - inicio de tracking "serio"
- **15 de enero 2026**: Shadow y Sensor B comienzan a emitir predicciones

### Reglas de Comparación
1. **Mínimo 2 modelos**: Un solo modelo es información, no comparación
2. **Fecha dinámica**: La fecha de inicio es la MÁS RECIENTE entre los modelos seleccionados
3. **Universo completo**: Cada modelo se evalúa en TODO su rango disponible desde la fecha calculada

---

## 🔧 Backend: `app/dashboard/model_benchmark.py`

### Cambio 1: Endpoint con Query Param

```python
@app.get("/model-benchmark")
async def get_model_benchmark(
    models: str = Query(..., description="Comma-separated model names"),
    db=Depends(get_db_session),
    _: str = Depends(require_dashboard_token)
):
    """
    Get benchmark data for selected models with dynamic date range.
    
    Examples:
        ?models=Market,Model%20A          -> desde 2026-01-10
        ?models=Market,Model%20A,Shadow   -> desde 2026-01-15
        ?models=Model%20A                 -> HTTP 400 (mínimo 2)
    """
```

### Cambio 2: Fechas de Disponibilidad

```python
MODEL_AVAILABILITY = {
    "Market": "2026-01-04",    # Odds disponibles desde siempre
    "Model A": "2026-01-10",   # Post-ajustes importantes
    "Shadow": "2026-01-15",    # Inicio Shadow
    "Sensor B": "2026-01-15",  # Inicio Sensor B
}

def calculate_start_date(selected: List[str]) -> str:
    """
    Retorna la fecha más reciente entre los modelos seleccionados.
    """
    if len(selected) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Selecciona al menos 2 modelos para comparar"
        )
    
    dates = [MODEL_AVAILABILITY[m] for m in selected]
    return max(dates)  # Fecha más reciente
```

### Cambio 3: Query SQL Dinámico

El query debe:
1. Usar la fecha calculada (`start_date`)
2. Solo requerir predicciones para los modelos seleccionados
3. Permitir NULLs en modelos NO seleccionados

```sql
WITH match_predictions AS (
  SELECT 
    m.id as match_id,
    DATE(m.date) as match_date,
    m.home_goals,
    m.away_goals,
    -- Market (siempre requerido si está seleccionado)
    CASE 
      WHEN m.odds_home < m.odds_draw AND m.odds_home < m.odds_away THEN 'H'
      WHEN m.odds_draw < m.odds_home AND m.odds_draw < m.odds_away THEN 'D'
      ELSE 'A'
    END as market_pred,
    -- Model A (solo si está seleccionado)
    CASE WHEN :include_model_a THEN
      (SELECT CASE 
         WHEN p.home_prob > p.draw_prob AND p.home_prob > p.away_prob THEN 'H'
         WHEN p.draw_prob > p.home_prob AND p.draw_prob > p.away_prob THEN 'D'
         ELSE 'A'
       END
       FROM predictions p 
       WHERE p.match_id = m.id AND p.model_version = 'v1.0.0'
       LIMIT 1)
    END as model_a_pred,
    -- Shadow (solo si está seleccionado)
    CASE WHEN :include_shadow THEN
      (SELECT CASE 
         WHEN p.home_prob > p.draw_prob AND p.home_prob > p.away_prob THEN 'H'
         WHEN p.draw_prob > p.home_prob AND p.draw_prob > p.away_prob THEN 'D'
         ELSE 'A'
       END
       FROM predictions p 
       WHERE p.match_id = m.id AND p.model_version = 'v1.1.0-two_stage'
       LIMIT 1)
    END as shadow_pred,
    -- Sensor B (solo si está seleccionado)
    CASE WHEN :include_sensor_b THEN
      (SELECT CASE 
         WHEN sp.b_pick = 'home' THEN 'H'
         WHEN sp.b_pick = 'draw' THEN 'D'
         WHEN sp.b_pick = 'away' THEN 'A'
       END
       FROM sensor_predictions sp 
       WHERE sp.match_id = m.id
       LIMIT 1)
    END as sensor_b_pred
  FROM matches m
  WHERE m.status = 'FT'
    AND m.date >= :start_date  -- Dinámico: 2026-01-10 o 2026-01-15
    AND m.odds_home IS NOT NULL 
    AND m.odds_draw IS NOT NULL 
    AND m.odds_away IS NOT NULL
),
complete_matches AS (
  SELECT *
  FROM match_predictions
  WHERE 
    -- Solo validar presencia de modelos SELECCIONADOS
    market_pred IS NOT NULL
    AND (model_a_pred IS NOT NULL OR NOT :include_model_a)
    AND (shadow_pred IS NOT NULL OR NOT :include_shadow)
    AND (sensor_b_pred IS NOT NULL OR NOT :include_sensor_b)
),
daily_results AS (
  SELECT 
    match_date,
    COUNT(*) as total_matches,
    SUM(CASE WHEN (home_goals > away_goals AND market_pred = 'H') OR (home_goals = away_goals AND market_pred = 'D') OR (home_goals < away_goals AND market_pred = 'A') THEN 1 ELSE 0 END) as market_correct,
    SUM(CASE WHEN (home_goals > away_goals AND model_a_pred = 'H') OR (home_goals = away_goals AND model_a_pred = 'D') OR (home_goals < away_goals AND model_a_pred = 'A') THEN 1 ELSE 0 END) as model_a_correct,
    SUM(CASE WHEN (home_goals > away_goals AND shadow_pred = 'H') OR (home_goals = away_goals AND shadow_pred = 'D') OR (home_goals < away_goals AND shadow_pred = 'A') THEN 1 ELSE 0 END) as shadow_correct,
    SUM(CASE WHEN (home_goals > away_goals AND sensor_b_pred = 'H') OR (home_goals = away_goals AND sensor_b_pred = 'D') OR (home_goals < away_goals AND sensor_b_pred = 'A') THEN 1 ELSE 0 END) as sensor_b_correct
  FROM complete_matches
  GROUP BY match_date
  ORDER BY match_date
)
SELECT * FROM daily_results;
```

### Cambio 4: Response Actualizado

```python
class ModelBenchmarkResponse(BaseModel):
    generated_at: str
    start_date: str           # ← NUEVO: fecha calculada
    selected_models: List[str] # ← NUEVO: modelos solicitados
    total_matches: int
    daily_data: List[DailyModelStats]
    models: List[ModelSummary]
```

---

## 🔧 Frontend: `dashboard/lib/hooks/use-model-benchmark.ts`

### Hook Actualizado

```typescript
export function useModelBenchmark(
  selectedModels: string[]
): UseModelBenchmarkResult {
  
  const enabled = selectedModels.length >= 2;
  
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["model-benchmark", selectedModels.sort().join(",")],
    queryFn: () => fetchModelBenchmark(selectedModels),
    enabled,
    retry: 1,
    staleTime: 5 * 60 * 1000,
  });

  return {
    data,
    isLoading: isLoading || !enabled,
    isDegraded: !enabled || !!error || !data,
    error: error as Error | null,
    refetch,
    // Mensaje de error para UI
    validationError: !enabled && selectedModels.length > 0 
      ? "Selecciona al menos 2 modelos para comparar" 
      : null,
  };
}

async function fetchModelBenchmark(
  models: string[]
): Promise<ModelBenchmarkResponse> {
  const response = await fetch(
    `/api/model-benchmark?models=${encodeURIComponent(models.join(","))}`,
    {
      headers: {
        "X-Dashboard-Token": process.env.NEXT_PUBLIC_DASHBOARD_TOKEN || "",
      },
    }
  );

  if (!response.ok) {
    throw new Error("Failed to fetch benchmark");
  }

  return response.json();
}
```

---

## 🔧 Frontend: `ModelBenchmarkTile.tsx`

### Cambio 1: Tipos Actualizados

```typescript
interface ModelBenchmarkResponse {
  generated_at: string;
  start_date: string;        // ← NUEVO
  selected_models: string[]; // ← NUEVO
  total_matches: number;
  daily_data: DailyModelStats[];
  models: ModelSummary[];
}
```

### Cambio 2: UI - Mostrar Rango Dinámico

```tsx
{/* Header con info de rango */}
<div className="flex items-start justify-between mb-4">
  <div className="flex items-center gap-2">
    <TrendingUp className="h-4 w-4 text-primary" />
    <h3 className="text-sm font-semibold text-foreground">
      Model Benchmark
    </h3>
    {data && (
      <span className="text-xs text-muted-foreground">
        desde {new Date(data.start_date).toLocaleDateString("es-ES", {
          day: "numeric",
          month: "short"
        })} 
        (n={data.total_matches})
      </span>
    )}
  </div>
  
  {/* Leader */}
  {leader && (
    <div className="flex items-center gap-1.5 text-xs">
      <Trophy className="h-3 w-3 text-yellow-500" />
      <span className="text-muted-foreground">Líder:</span>
      <span className="font-medium">{leader.name}</span>
      <span className="text-muted-foreground">({leader.accuracy}%)</span>
    </div>
  )}
</div>

{/* Alerta si menos de 2 modelos seleccionados */}
{selectedModels.length < 2 && (
  <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-md">
    <p className="text-xs text-yellow-500">
      Selecciona al menos 2 modelos para ver la comparación
    </p>
  </div>
)}
```

### Cambio 3: Toggle con Validación

```tsx
{/* Deshabilitar si quedaría < 2 modelos */}
<button
  key={model.name}
  onClick={() => toggleModel(model.name)}
  disabled={
    selectedModels.includes(model.name) && 
    selectedModels.length <= 2
  }
  className={cn(
    "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all border",
    selectedModels.includes(model.name)
      ? "bg-surface border-border hover:border-primary/50"
      : "bg-transparent border-transparent opacity-50 hover:opacity-75",
    selectedModels.includes(model.name) && 
    selectedModels.length <= 2 && 
    "cursor-not-allowed opacity-75"
  )}
>
  {/* ... contenido ... */}
</button>
```

---

## 🧪 Casos de Prueba Esperados

### Caso 1: Market + Model A
```
GET /model-benchmark?models=Market,Model%20A
→ start_date: 2026-01-10
→ n: ~800+ partidos
```

### Caso 2: Todos los modelos
```
GET /model-benchmark?models=Market,Model%20A,Shadow,Sensor%20B
→ start_date: 2026-01-15
→ n: ~392 partidos
```

### Caso 3: Market + Shadow
```
GET /model-benchmark?models=Market,Shadow
→ start_date: 2026-01-15
→ n: ~392 partidos (desde que Shadow existe)
```

### Caso 4: Error - Solo 1 modelo
```
GET /model-benchmark?models=Model%20A
→ HTTP 400: "Selecciona al menos 2 modelos para comparar"
```

---

## 📊 Validación de Resultados

Después de implementar, los números deben coincidir con el dashboard sencillo:

| Escenario | n esperado | Market | Model A | Shadow | Sensor B |
|-----------|------------|--------|---------|--------|----------|
| 4 modelos | ~392 | 49.7% | 45.9% | 46.7% | 38.0% |
| Market + Model A | ~800 | ~48% | ~45% | - | - |

Si Shadow muestra ~15% en lugar de ~46%, el filtro de `complete_matches` está incorrecto.

---

## ✅ Checklist de Implementación

- [ ] Backend: Endpoint acepta query param `models`
- [ ] Backend: Valida mínimo 2 modelos
- [ ] Backend: Calcula `start_date` como max(fechas de modelos)
- [ ] Backend: Query SQL dinámico con CASE para modelos opcionales
- [ ] Backend: Response incluye `start_date` y `selected_models`
- [ ] Frontend: Hook envía modelos seleccionados como query param
- [ ] Frontend: Deshabilita query si < 2 modelos
- [ ] Frontend: Muestra `start_date` en UI
- [ ] Frontend: Prevenir deseleccionar si quedaría < 2 modelos
- [ ] Testing: Validar números contra dashboard sencillo

---

## 📝 Notas Importantes

1. **Sensor B mapping**: `b_pick` tiene valores `'home'/'draw'/'away'` que deben convertirse a `'H'/'D'/'A'`

2. **Shadow version**: `model_version = 'v1.1.0-two_stage'` (no 'shadow')

3. **Model A version**: `model_version = 'v1.0.0'`

4. **Market**: Calculado desde `matches.odds_home/draw/away` (cuota más baja)

5. **Timezone**: Las fechas están en hora local Los Angeles (PST/PDT), pero la DB probablemente usa UTC. Verificar que el query use la fecha correcta.
