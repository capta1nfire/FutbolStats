# Logos 3D - Especificación v3

> **Status**: Aprobado por Kimi - Listo para implementación
> **Última actualización**: 2026-01-28
> **Validado por**: Kimi (ADB)

## Objetivo

Sistema de logos 3D para:
1. **Equipos/Selecciones**: Perspectiva 3D estilo "cartel de boxeo" para matchups
2. **Competiciones (Ligas/Torneos)**: Logo 3D frontal para headers y badges

```
    HOME                              AWAY
┌─────────────┐                  ┌─────────────┐
│   ╲╲╲╲╲╲╲   │                  │   ╱╱╱╱╱╱╱   │
│  ╲ AMÉRICA ╲│  ──────────────► │╱  SANTA FE ╱│
│ ╲   🔱    ╲ │   se miran       │ ╱    ●    ╱ │
│╲__________╲ │                  │╱__________╱ │
└─────────────┘                  └─────────────┘
  facing_right                     facing_left
  (mira →)                         (mira ←)
```

---

## Variantes de Logo

| Variante | Descripción | Generado por |
|----------|-------------|--------------|
| `original` | Logo base subido por usuario | Upload manual |
| `front_3d` | Escudo 3D frontal con efecto metálico | IA (opcional) |
| `facing_right` | Escudo 3D mirando → (HOME en matchup) | IA (opcional) |
| `facing_left` | Escudo 3D mirando ← (AWAY en matchup) | IA (opcional) |
| `logo_url` (existente) | API-Football flat | Fallback |

---

## Modos de Generación (Flexibilidad)

El sistema permite elegir qué generar según presupuesto/necesidad:

| Modo | Imágenes IA | Front usado | Costo/equipo (DALL-E) |
|------|-------------|-------------|----------------------|
| **Full 3D** | 3 (front + right + left) | `front_3d` generado | $0.12 |
| **Facing Only** | 2 (right + left) | `original` subido | $0.08 |
| **Front Only** | 1 (front) | `front_3d` generado | $0.04 |
| **Manual** | 0 | `original` subido | $0.00 |

```
┌─────────────────────────────────────────────────────────────────┐
│  Modo de Generación:                                            │
│                                                                  │
│  (•) Full 3D        - Generar front + facing (3 imágenes IA)   │
│  ( ) Facing Only    - Usar original como front, generar facing │
│  ( ) Front Only     - Solo generar front 3D                    │
│  ( ) Manual         - Usar original como front, sin facing     │
└─────────────────────────────────────────────────────────────────┘
```

**Casos de uso:**
- **Full 3D**: Máxima calidad visual, presupuesto completo
- **Facing Only**: El logo original ya es bueno, solo necesita perspectiva
- **Front Only**: Solo se usa en drawer/perfil, no en matchups
- **Manual**: Logo ya tiene calidad suficiente, no requiere IA

---

## Logos de Competiciones (Ligas/Torneos)

Las competiciones **solo necesitan 1 variante**: `main` (front 3D).

| Variante | Descripción | Uso |
|----------|-------------|-----|
| `original` | Logo base subido | Input para IA |
| `main` | Logo 3D frontal | Headers, badges, filtros |

**¿Por qué no facing?**
- Las competiciones no se "enfrentan" entre sí
- Se usan como badges/headers, no en matchups

### Números de Competiciones

| Tipo | Activas |
|------|---------|
| Ligas | 25 |
| Internacionales | 19 |
| Copas | 2 |
| Friendly | 1 |
| **Total** | **47** |

### Costo adicional (insignificante)

| Modelo | Costo 47 imgs |
|--------|---------------|
| SDXL | ~$0.38 |
| DALL-E | ~$1.88 |

---

## Flujo de Procesamiento

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUJO DE IMÁGENES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 1: Upload Masivo (Dashboard Settings)                                  │
│  ══════════════════════════════════════════                                  │
│                                                                              │
│  Usuario sube logos originales (1600+ clubes + selecciones)                  │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────┐                                                     │
│  │  Validación         │  - Formato: PNG, SVG, WebP                          │
│  │  - Min 512x512px    │  - Max 5MB                                          │
│  │  - Aspect ~1:1      │  - Fondo transparente preferido                     │
│  └─────────────────────┘                                                     │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────┐                                                     │
│  │  R2: original       │  logos/{team_id}/original.png                       │
│  │  (solo referencia)  │  ⚠️ NO ES EL FRONT - Solo input para IA            │
│  └─────────────────────┘                                                     │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────┐                                                     │
│  │  DB: team_logos     │  status = 'pending'                                 │
│  │                     │  r2_key_original = "logos/1234/original.png"        │
│  └─────────────────────┘                                                     │
│                                                                              │
│                                                                              │
│  FASE 2: Configuración IA (Dashboard Settings)                               │
│  ═════════════════════════════════════════════                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  🎨 Generador de Escudos IA                                             │  │
│  │                                                                         │  │
│  │  Modelo: [DALL-E 3 ▼] [Gemini] [SDXL/Replicate] [Midjourney]           │  │
│  │                                                                         │  │
│  │  Prompts:                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │ Front 3D:                                                        │   │  │
│  │  │ [Transform this 2D football shield into a 3D metallic badge...] │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │ Facing Right:                                                    │   │  │
│  │  │ [...facing 45 degrees to the right with left-to-right lighting] │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │  │ Facing Left:                                                     │   │  │
│  │  │ [...facing 45 degrees to the left with right-to-left lighting]  │   │  │
│  │  └─────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                         │  │
│  │  Preview (con 1 equipo de prueba):                                      │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                                 │  │
│  │  │ front   │  │  right  │  │  left   │                                 │  │
│  │  │  3D     │  │   →     │  │   ←     │                                 │  │
│  │  └─────────┘  └─────────┘  └─────────┘                                 │  │
│  │                                                                         │  │
│  │  Equipos seleccionados: [✓] Todos (1620)  [ ] Solo sin procesar (1450) │  │
│  │                                                                         │  │
│  │  Estimado: 4860 imágenes · ~$39 (SDXL) / ~$195 (DALL-E) · ~2-4 horas   │  │
│  │                                                                         │  │
│  │                              [🚀 Generar Escudos IA]                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                                                                              │
│  FASE 3: Batch IA (Asíncrono con Progress Bar)                               │
│  ═════════════════════════════════════════════                               │
│                                                                              │
│  Al hacer clic en "Generar", se crea un batch job y aparece en Overview:     │
│                                                                              │
│  Dashboard Overview:                                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 🎨 Generación Escudos IA                              [En progreso]    │  │
│  │ ████████████░░░░░░░░░░░░░░░░░░░░ 35%                                   │  │
│  │                                                                         │  │
│  │ Equipos:   567 / 1,620 completados                                      │  │
│  │ Imágenes:  1,701 / 4,860 generadas                                      │  │
│  │                                                                         │  │
│  │ Modelo:    DALL-E 3                                                     │  │
│  │ Costo:     $68.04 / $194.40 estimado                                    │  │
│  │ Tiempo:    1h 23m transcurrido · ETA ~2h 30m                            │  │
│  │ Errores:   12 (0.7%) - [Ver detalles]                                   │  │
│  │                                                                         │  │
│  │ [⏸ Pausar]  [⏹ Cancelar]  [📋 Ver Log]                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Procesamiento por equipo:                                                   │
│  1. Descargar original.png de R2                                             │
│  2. Llamar IA con prompt_front → front_3d.png                                │
│  3. Llamar IA con prompt_right → facing_right.png                            │
│  4. Llamar IA con prompt_left → facing_left.png                              │
│  5. Validar resultados (tamaño, transparencia)                               │
│  6. Subir 3 imágenes a R2                                                    │
│  7. Actualizar DB: status = 'pending_resize'                                 │
│                                                                              │
│  ⚠️ 3 prompts separados por equipo (no 1 prompt → 3 imágenes)               │
│  Razón: Control fino por perspectiva, retry selectivo, lighting diferente    │
│                                                                              │
│                                                                              │
│  FASE 4: Thumbnails (Automático Post-IA)                                     │
│  ═══════════════════════════════════════                                     │
│                                                                              │
│  Job automático procesa equipos con status = 'pending_resize':               │
│                                                                              │
│  Para cada variante (front_3d, facing_right, facing_left):                   │
│  - 64px  (tiny, tables)                                                      │
│  - 128px (small, cards)                                                      │
│  - 256px (medium, drawer)                                                    │
│  - 512px (large, hero)                                                       │
│  Formato: WebP (mejor compresión)                                            │
│                                                                              │
│  Total: 3 variantes × 4 tamaños = 12 thumbnails por equipo                   │
│                                                                              │
│  Al completar: status = 'ready'                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ¿Por qué 3 prompts separados?

| Aspecto | 1 prompt → 3 imgs | 3 prompts separados |
|---------|-------------------|---------------------|
| Control | Bajo | Alto (lighting, ángulo específico) |
| Retry | Todo o nada | Selectivo por imagen |
| Costo | Menor | 3x (pero necesario) |
| Soporte | Pocos modelos | Todos |
| Calidad | Variable | Consistente |

**Decisión**: 3 prompts separados porque cada perspectiva tiene lighting diferente y necesitamos control fino.

---

## Procesamiento Controlado por Liga

Para **control de calidad**, el procesamiento se hace **liga por liga** con supervisión manual:

### ¿Por qué no batch masivo?

| Aspecto | Batch Masivo | Liga por Liga |
|---------|--------------|---------------|
| Riesgo | Alto ($200 de golpe) | Bajo (~$2-5/liga) |
| Control de calidad | Post-mortem | En tiempo real |
| Ajuste de prompts | Difícil (ya gastaste) | Fácil (entre ligas) |
| Rollback | Costoso | Barato |
| Supervisión | Imposible (4,900 imgs) | Manejable (~40-80 imgs/liga) |

### Flujo de Aprobación por Liga

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GENERACIÓN CONTROLADA POR LIGA                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Seleccionar Liga: [Liga Colombiana        ▼]                            │
│     Equipos: 20 · Imágenes: 60 · Costo estimado: ~$2.40 (DALL-E)           │
│                                                                              │
│  2. [🚀 Generar Liga]                                                       │
│                                                                              │
│  3. Revisión Visual (post-generación):                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ ✅ América de Cali                                                      │ │
│  │ ┌─────────┐  ┌─────────┐  ┌─────────┐                                  │ │
│  │ │ front   │  │  right  │  │  left   │     [✓ OK]  [🔄 Regenerar]       │ │
│  │ └─────────┘  └─────────┘  └─────────┘                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ ⚠️ Deportivo Cali (marcado para revisión)                              │ │
│  │ ┌─────────┐  ┌─────────┐  ┌─────────┐                                  │ │
│  │ │ front   │  │  right  │  │  left   │     [✓ OK]  [🔄 Regenerar]       │ │
│  │ └─────────┘  └─────────┘  └─────────┘                                  │ │
│  │ Problema: "Colores distorsionados en facing_left"                       │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ ✅ Millonarios FC                                                       │ │
│  │ ...                                                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  4. Resumen Liga Colombiana:                                                │
│     ├── 18/20 aprobados ✓                                                   │
│     ├── 2 marcados para regenerar                                           │
│     └── Costo real: $2.28                                                   │
│                                                                              │
│  5. Acciones:                                                               │
│     [✓ Aprobar Liga y Continuar]  [🔄 Regenerar Marcados]  [⏸ Pausar]      │
│                                                                              │
│  6. Siguiente liga: [Liga Argentina        ▼]                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Orden Sugerido de Procesamiento

| Fase | Entidad | Equipos | Costo (DALL-E) | Propósito |
|------|---------|---------|----------------|-----------|
| 1 | 🧪 Liga Colombia (prueba) | ~20 | ~$2.40 | Validar prompts |
| 2 | ⭐ Top 5 ligas (Premier, LaLiga, Serie A, Bundesliga, Ligue 1) | ~100 | ~$12 | Alta visibilidad |
| 3 | 🇪🇺 Resto Europa | ~300 | ~$36 | Completar Europa |
| 4 | 🌎 LATAM | ~200 | ~$24 | Mercado importante |
| 5 | 🌍 Otros (Asia, África, etc.) | ~200 | ~$24 | Cobertura global |
| 6 | 🏳️ Selecciones nacionales | ~200 | ~$24 | Internacionales |
| 7 | 🏆 Competiciones (logos de ligas/torneos) | ~47 | ~$1.88 | Badges/headers |
| **Total** | - | **~1,067** | **~$124** | - |

### Schema Adicional para Control por Liga

```sql
-- Agregar a logo_batch_jobs
ALTER TABLE logo_batch_jobs ADD COLUMN IF NOT EXISTS
  -- Scope por liga
  entity_type VARCHAR(20) NOT NULL DEFAULT 'league',
  -- Valores: 'league', 'national_teams', 'competitions', 'custom'
  league_id INTEGER NULL REFERENCES admin_leagues(league_id),

  -- Aprobación manual
  approval_status VARCHAR(20) DEFAULT 'pending_review',
  -- Valores: 'pending_review', 'approved', 'partially_approved', 'rejected'
  approved_count INTEGER DEFAULT 0,
  rejected_count INTEGER DEFAULT 0,
  approved_by VARCHAR(100),
  approved_at TIMESTAMP,

  -- Para re-runs
  parent_batch_id UUID NULL REFERENCES logo_batch_jobs(id),
  is_rerun BOOLEAN DEFAULT FALSE,
  rerun_reason VARCHAR(100);  -- 'bad_quality', 'prompt_updated', 'partial_failures'
```

### Estados de Aprobación por Equipo

```sql
-- Agregar a team_logos
ALTER TABLE team_logos ADD COLUMN IF NOT EXISTS
  review_status VARCHAR(20) DEFAULT 'pending',
  -- Valores: 'pending', 'approved', 'rejected', 'needs_regeneration'
  review_notes TEXT,
  reviewed_by VARCHAR(100),
  reviewed_at TIMESTAMP;
```

### API Endpoints para Control por Liga

```
# Obtener ligas disponibles para procesar
GET /dashboard/logos/leagues
Response:
{
  "leagues": [
    {
      "league_id": 239,
      "name": "Liga Colombiana",
      "country": "Colombia",
      "teams_count": 20,
      "teams_with_original": 20,
      "teams_processed": 0,
      "estimated_cost_usd": 2.40,
      "status": "ready"  // ready | in_progress | completed | needs_review
    },
    ...
  ]
}

# Iniciar generación por liga
POST /dashboard/logos/generate/league/{league_id}
Body:
{
  "generation_mode": "full_3d",
  "ia_model": "dall-e-3",
  "prompt_front": "...",
  "prompt_right": "...",
  "prompt_left": "...",
  "prompt_version": "v1"
}

# Obtener resultados de liga para revisión
GET /dashboard/logos/review/league/{league_id}
Response:
{
  "league_id": 239,
  "batch_job_id": "...",
  "teams": [
    {
      "team_id": 1234,
      "name": "América de Cali",
      "status": "ready",
      "review_status": "pending",
      "urls": {
        "front": "https://...",
        "right": "https://...",
        "left": "https://..."
      }
    },
    ...
  ],
  "summary": {
    "total": 20,
    "approved": 0,
    "rejected": 0,
    "pending": 20
  }
}

# Aprobar/Rechazar equipo individual
POST /dashboard/logos/review/team/{team_id}
Body:
{
  "action": "approve" | "reject" | "regenerate",
  "notes": "Colores distorsionados en facing_left",
  "regenerate_variants": ["facing_left"]  // Solo si action = regenerate
}

# Aprobar liga completa
POST /dashboard/logos/review/league/{league_id}/approve
Body:
{
  "action": "approve_all" | "approve_reviewed" | "reject_all"
}
```

### UI de Revisión por Liga

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📋 Revisión: Liga Colombiana                    [Batch #a1b2c3] [En revisión]│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Filtros: [Todos ▼]  [Pendientes]  [Aprobados]  [Rechazados]               │
│                                                                              │
│  ┌─ Grid de Equipos ─────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │ América Cali │  │ Dep. Cali    │  │ Millonarios  │                 │  │
│  │  │ [F] [R] [L]  │  │ [F] [R] [L]  │  │ [F] [R] [L]  │                 │  │
│  │  │ ○ Pendiente  │  │ ⚠ Rechazado  │  │ ✓ Aprobado   │                 │  │
│  │  │ [✓] [✗] [🔄]│  │ [✓] [✗] [🔄]│  │ [✓] [✗] [🔄]│                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  │                                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │ Junior       │  │ Nacional     │  │ Santa Fe     │                 │  │
│  │  │ ...          │  │ ...          │  │ ...          │                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  Resumen:  ✓ 15 Aprobados  ⚠ 3 Rechazados  ○ 2 Pendientes                  │
│  Costo:    $2.28 gastado                                                    │
│                                                                              │
│  [Aprobar Todos Pendientes]  [Regenerar Rechazados ($0.36)]  [Siguiente Liga →]│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Números Totales

| Entidad | Cantidad | Imgs IA (Full 3D) |
|---------|----------|-------------------|
| Equipos + selecciones | ~1,620 | 3 c/u = 4,860 |
| Competiciones | ~47 | 1 c/u = 47 |
| **Total** | **~1,667** | **~4,907** |

### Imágenes IA por modo (equipos)

| Modo | Imgs/equipo | Total imgs | Thumbnails/equipo |
|------|-------------|------------|-------------------|
| Full 3D | 3 | ~4,860 | 12 (3×4) |
| Facing Only | 2 | ~3,240 | 12 (3×4)* |
| Front Only | 1 | ~1,620 | 4 (1×4) |
| Manual | 0 | 0 | 4 (1×4) |

*Facing Only: original se usa como front, pero igual genera 3 variantes de thumbnails

### Costos estimados por modo (~1,620 equipos + 47 competiciones)

| Modelo | $/img | Full 3D (equipos) | + Competiciones | **Total** |
|--------|-------|-------------------|-----------------|-----------|
| SDXL (Replicate) | $0.008 | ~$39 | +$0.38 | **~$39** |
| Stable Diffusion 3 | $0.035 | ~$170 | +$1.65 | **~$172** |
| DALL-E 3 | $0.040 | ~$195 | +$1.88 | **~$197** |
| Midjourney | $0.050 | ~$243 | +$2.35 | **~$245** |

*Competiciones son solo ~1% del costo total*

---

## Schema de Base de Datos

```sql
-- ============================================================================
-- Tabla: team_logos
-- ============================================================================

CREATE TABLE team_logos (
  team_id INTEGER PRIMARY KEY REFERENCES teams(id) ON DELETE CASCADE,

  -- ══════════════════════════════════════════════════════════════════════════
  -- Referencias R2 (solo originales, thumbnails derivados en urls JSONB)
  -- ══════════════════════════════════════════════════════════════════════════
  r2_key_original VARCHAR(255),      -- logos/{team_id}/original.png (input)
  r2_key_front VARCHAR(255),         -- logos/{team_id}/front_3d.png
  r2_key_right VARCHAR(255),         -- logos/{team_id}/facing_right.png
  r2_key_left VARCHAR(255),          -- logos/{team_id}/facing_left.png

  -- URLs de thumbnails (generadas post-resize)
  urls JSONB DEFAULT '{}',
  -- Estructura:
  -- {
  --   "front": {"64": "https://...", "128": "...", "256": "...", "512": "..."},
  --   "right": {"64": "...", "128": "...", "256": "...", "512": "..."},
  --   "left":  {"64": "...", "128": "...", "256": "...", "512": "..."}
  -- }

  -- Fallback (API-Football URL original)
  fallback_url VARCHAR(500),

  -- ══════════════════════════════════════════════════════════════════════════
  -- Estado del pipeline
  -- ══════════════════════════════════════════════════════════════════════════
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  -- Valores:
  --   'pending'          = Original subido, esperando generación IA
  --   'queued'           = En cola para batch IA
  --   'processing'       = IA generando imágenes
  --   'pending_resize'   = IA completó, esperando thumbnails
  --   'ready'            = Todo listo
  --   'error'            = Falló (ver error_message)
  --   'paused'           = Pausado por usuario

  -- ══════════════════════════════════════════════════════════════════════════
  -- Metadata del procesamiento
  -- ══════════════════════════════════════════════════════════════════════════
  batch_job_id UUID,                  -- Referencia al batch job que lo procesó
  generation_mode VARCHAR(20),        -- 'full_3d', 'facing_only', 'front_only', 'manual'
  ia_model VARCHAR(50),               -- 'dall-e-3', 'sdxl', 'gemini', etc. (NULL si manual)
  ia_prompt_version VARCHAR(20),      -- 'v1', 'v2', etc. (para tracking)
  use_original_as_front BOOLEAN DEFAULT FALSE,  -- TRUE si facing_only o manual

  -- Timestamps
  uploaded_at TIMESTAMP,
  processing_started_at TIMESTAMP,
  processing_completed_at TIMESTAMP,
  resize_completed_at TIMESTAMP,

  -- Costos
  ia_cost_usd DECIMAL(10,4),          -- Costo total IA (3 imágenes)

  -- ══════════════════════════════════════════════════════════════════════════
  -- Error handling
  -- ══════════════════════════════════════════════════════════════════════════
  error_message TEXT,
  error_phase VARCHAR(20),            -- 'upload', 'ia_front', 'ia_right', 'ia_left', 'resize'
  retry_count INTEGER DEFAULT 0,
  last_retry_at TIMESTAMP,

  -- ══════════════════════════════════════════════════════════════════════════
  -- Audit
  -- ══════════════════════════════════════════════════════════════════════════
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_team_logos_status ON team_logos(status)
  WHERE status NOT IN ('ready', 'error');
CREATE INDEX idx_team_logos_batch ON team_logos(batch_job_id)
  WHERE batch_job_id IS NOT NULL;

-- ============================================================================
-- Tabla: competition_logos (para ligas/torneos - solo main, sin facing)
-- ============================================================================

CREATE TABLE competition_logos (
  league_id INTEGER PRIMARY KEY REFERENCES admin_leagues(league_id) ON DELETE CASCADE,

  -- Referencias R2
  r2_key_original VARCHAR(255),      -- logos/competitions/{league_id}/original.png
  r2_key_main VARCHAR(255),          -- logos/competitions/{league_id}/main.png

  -- URLs de thumbnails
  urls JSONB DEFAULT '{}',
  -- Estructura: { "64": "https://...", "128": "...", "256": "...", "512": "..." }

  -- Fallback
  fallback_url VARCHAR(500),         -- URL de API-Football

  -- Estado (simplificado - no hay facing)
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  -- Valores: 'pending', 'queued', 'processing', 'pending_resize', 'ready', 'error'

  -- Metadata
  batch_job_id UUID,
  ia_model VARCHAR(50),
  ia_cost_usd DECIMAL(10,4),

  -- Error handling
  error_message TEXT,
  retry_count INTEGER DEFAULT 0,

  -- Timestamps
  uploaded_at TIMESTAMP,
  processing_completed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_competition_logos_status ON competition_logos(status)
  WHERE status NOT IN ('ready', 'error');

-- ============================================================================
-- Tabla: logo_batch_jobs (para tracking de generación masiva)
-- ============================================================================

CREATE TABLE logo_batch_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Configuración
  ia_model VARCHAR(50) NOT NULL,
  generation_mode VARCHAR(20) NOT NULL DEFAULT 'full_3d',
  -- Valores: 'full_3d', 'facing_only', 'front_only', 'manual'
  prompt_front TEXT,              -- NULL si mode = facing_only o manual
  prompt_right TEXT,              -- NULL si mode = front_only o manual
  prompt_left TEXT,               -- NULL si mode = front_only o manual
  prompt_version VARCHAR(20) NOT NULL,

  -- Scope
  total_teams INTEGER NOT NULL,
  team_ids INTEGER[],                 -- NULL = todos con status 'pending'

  -- Estado
  status VARCHAR(20) NOT NULL DEFAULT 'running',
  -- Valores: 'running', 'paused', 'completed', 'cancelled', 'error'

  -- Progreso
  processed_teams INTEGER DEFAULT 0,
  processed_images INTEGER DEFAULT 0,
  failed_teams INTEGER DEFAULT 0,

  -- Costos
  estimated_cost_usd DECIMAL(10,2),
  actual_cost_usd DECIMAL(10,2) DEFAULT 0,

  -- Timestamps
  started_at TIMESTAMP DEFAULT NOW(),
  paused_at TIMESTAMP,
  completed_at TIMESTAMP,

  -- Metadata
  started_by VARCHAR(100),            -- Usuario que inició

  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## Estructura R2

```
bucket: futbolstats-logos

logos/
├── teams/
│   └── {team_id}/
│       ├── original.png           # Subido por usuario (input para IA)
│       ├── front_3d.png           # Generado por IA
│       ├── front_3d_64.webp       # Thumbnails
│       ├── front_3d_128.webp
│       ├── front_3d_256.webp
│       ├── front_3d_512.webp
│       ├── facing_right.png       # Generado por IA
│       ├── facing_right_64.webp
│       ├── facing_right_128.webp
│       ├── facing_right_256.webp
│       ├── facing_right_512.webp
│       ├── facing_left.png        # Generado por IA
│       ├── facing_left_64.webp
│       ├── facing_left_128.webp
│       ├── facing_left_256.webp
│       └── facing_left_512.webp
│
└── competitions/
    └── {league_id}/
        ├── original.png           # Subido por usuario
        ├── main.png               # Generado por IA (solo 1 variante)
        ├── main_64.webp           # Thumbnails
        ├── main_128.webp
        ├── main_256.webp
        └── main_512.webp
```

**CDN URL patterns**:
- Equipos: `https://logos.futbolstats.app/teams/{team_id}/{variante}_{size}.webp`
- Competiciones: `https://logos.futbolstats.app/competitions/{league_id}/main_{size}.webp`

---

## API Endpoints

### Upload Logo Original

```
POST /dashboard/teams/{team_id}/logo
Content-Type: multipart/form-data
Body: file (PNG/SVG/WebP, max 5MB, min 512x512)

Response 201:
{
  "team_id": 1234,
  "status": "pending",
  "r2_key_original": "logos/1234/original.png",
  "message": "Logo uploaded. Ready for IA generation."
}
```

### Iniciar Batch IA

```
POST /dashboard/logos/generate
Content-Type: application/json
Body:
{
  "generation_mode": "full_3d",  // full_3d | facing_only | front_only | manual
  "ia_model": "dall-e-3",
  "prompt_front": "Transform this 2D football shield into a 3D metallic badge...",  // null si facing_only
  "prompt_right": "...facing 45 degrees to the right...",  // null si front_only
  "prompt_left": "...facing 45 degrees to the left...",    // null si front_only
  "prompt_version": "v1",
  "team_ids": null,  // null = todos con status 'pending'
  "dry_run": false   // true = solo calcular estimados
}

Response 202:
{
  "batch_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "total_teams": 1620,
  "total_images": 4860,
  "estimated_cost_usd": 194.40,
  "estimated_time_minutes": 180
}
```

### Status de Batch Job

```
GET /dashboard/logos/batch/{job_id}

Response:
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": {
    "teams": { "processed": 567, "total": 1620, "failed": 12 },
    "images": { "processed": 1701, "total": 4860 }
  },
  "cost": {
    "actual_usd": 68.04,
    "estimated_usd": 194.40
  },
  "time": {
    "started_at": "2026-01-28T10:00:00Z",
    "elapsed_minutes": 83,
    "eta_minutes": 150
  },
  "errors": [
    { "team_id": 1234, "team_name": "River Plate", "phase": "ia_right", "message": "Rate limit" }
  ]
}
```

### Pausar/Reanudar/Cancelar Batch

```
POST /dashboard/logos/batch/{job_id}/pause
POST /dashboard/logos/batch/{job_id}/resume
POST /dashboard/logos/batch/{job_id}/cancel
```

### Status de Logo Individual

```
GET /dashboard/teams/{team_id}/logo/status

Response:
{
  "team_id": 1234,
  "status": "ready",
  "urls": {
    "front": { "64": "https://...", "128": "...", "256": "...", "512": "..." },
    "right": { "64": "...", "128": "...", "256": "...", "512": "..." },
    "left": { "64": "...", "128": "...", "256": "...", "512": "..." }
  },
  "fallback_url": "https://media.api-sports.io/football/teams/1234.png",
  "ia_model": "dall-e-3",
  "processed_at": "2026-01-28T12:30:00Z"
}
```

---

## UI Components

### Settings > Logo Generator

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🎨 Generador de Escudos 3D                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PASO 1: Subir Logos Originales                                              │
│  ─────────────────────────────────                                           │
│  [📁 Subir múltiples] o [Buscar equipo: ___________]                        │
│                                                                              │
│  Equipos con logo original: 1,620 / 1,620 ✓                                  │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  PASO 2: Configurar Generación IA                                            │
│  ─────────────────────────────────                                           │
│                                                                              │
│  Modo:                                                                       │
│  (•) Full 3D      - Front + Facing Right + Facing Left (3 imgs/equipo)      │
│  ( ) Facing Only  - Usar original como front, generar facing (2 imgs)       │
│  ( ) Front Only   - Solo front 3D, sin facing (1 img)                       │
│  ( ) Manual       - Usar original, sin IA ($0)                              │
│                                                                              │
│  Modelo IA: [DALL-E 3        ▼]                                             │
│                                                                              │
│  Prompt Front 3D: [deshabilitado si Facing Only o Manual]                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Transform this 2D football team shield into a 3D metallic badge.       ││
│  │ Style: glossy chrome rim, professional sports badge, transparent bg.   ││
│  │ Lighting: frontal, even illumination.                                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Prompt Facing Right (HOME):                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ...same but facing 45 degrees to the right, left-to-right lighting...  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  Prompt Facing Left (AWAY):                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ...same but facing 45 degrees to the left, right-to-left lighting...   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  PASO 3: Preview                                                             │
│  ───────────────                                                             │
│                                                                              │
│  Equipo de prueba: [América de Cali    ▼]  [🔄 Generar Preview]             │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │             │  │             │  │             │                          │
│  │   FRONT     │  │   RIGHT     │  │   LEFT      │                          │
│  │    3D       │  │     →       │  │     ←       │                          │
│  │             │  │             │  │             │                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
│  [✓ Aprobado]     [✓ Aprobado]     [✓ Aprobado]                             │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                              │
│  PASO 4: Generar Masivo                                                      │
│  ──────────────────────                                                      │
│                                                                              │
│  Equipos: (•) Todos pendientes (1,450)  ( ) Selección manual                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ ⚠️ Resumen:                                                             ││
│  │ • 1,450 equipos × 3 imágenes = 4,350 generaciones IA                   ││
│  │ • Costo estimado: $174.00 (DALL-E 3 @ $0.04/img)                       ││
│  │ • Tiempo estimado: ~3-4 horas                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│                                        [🚀 Iniciar Generación Masiva]       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Overview > Progress Bar

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🎨 Generación Escudos IA                                      [En progreso] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 42%          │
│                                                                              │
│ ┌────────────────────┬────────────────────┬────────────────────┐            │
│ │ Equipos            │ Imágenes           │ Costo              │            │
│ │ 609 / 1,450        │ 1,827 / 4,350      │ $73.08 / $174.00   │            │
│ └────────────────────┴────────────────────┴────────────────────┘            │
│                                                                              │
│ Modelo: DALL-E 3 · Iniciado: 10:00 AM · ETA: 12:45 PM                       │
│ Errores: 8 (0.4%)                                                            │
│                                                                              │
│ [⏸ Pausar]  [⏹ Cancelar]  [📋 Ver Log]                      [Ver detalles →]│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Consideraciones de Auditoría (Kimi - ADB)

Las siguientes consideraciones fueron agregadas tras validación de Kimi para garantizar robustez operacional.

### 1. Validación Automática Post-IA

Toda imagen generada por IA debe pasar validación antes de guardarse en R2:

```python
from dataclasses import dataclass
from PIL import Image
import io

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    dimensions: tuple[int, int] | None = None
    has_alpha: bool = False

def validate_ia_output(image_bytes: bytes, variant: str) -> ValidationResult:
    """
    Validar imagen generada por IA antes de guardar.

    Checks:
    - Size mínimo: 512x512px
    - Aspect ratio: ~1:1 (tolerancia 5%)
    - Formato válido: PNG con alpha channel
    - Transparencia: background debe ser transparente
    - Corrupción: imagen carga correctamente
    - File size: < 5MB
    """
    errors = []

    # Verificar tamaño de archivo
    if len(image_bytes) > 5 * 1024 * 1024:
        errors.append(f"Archivo muy grande: {len(image_bytes) / 1024 / 1024:.1f}MB")

    # Cargar y validar
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return ValidationResult(valid=False, errors=["Imagen corrupta o formato inválido"])

    # Dimensiones mínimas
    if img.width < 512 or img.height < 512:
        errors.append(f"Tamaño insuficiente: {img.width}x{img.height} (mínimo 512x512)")

    # Aspect ratio (~1:1)
    ratio = img.width / img.height
    if not (0.95 <= ratio <= 1.05):
        errors.append(f"Aspect ratio incorrecto: {ratio:.2f} (esperado ~1:1)")

    # Transparencia (canal alpha)
    if img.mode != 'RGBA':
        errors.append(f"Sin canal alpha: modo={img.mode} (esperado RGBA)")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        dimensions=(img.width, img.height),
        has_alpha=img.mode == 'RGBA'
    )
```

**Política de retry**:
| Condición | Acción |
|-----------|--------|
| `retry_count < 3` | Re-intentar con mismo prompt |
| `retry_count >= 3` | Marcar `status='error'`, agregar a DLQ manual |

**Columnas en team_logos para tracking**:
```sql
ALTER TABLE team_logos ADD COLUMN IF NOT EXISTS
  validation_errors JSONB,          -- Errores de última validación
  last_validation_at TIMESTAMP;     -- Cuándo se validó
```

### 2. Sistema de Plantillas de Prompts

Para versionado y A/B testing de prompts:

```sql
CREATE TABLE logo_prompt_templates (
    id SERIAL PRIMARY KEY,
    version VARCHAR(10) NOT NULL,           -- 'v1', 'v2', etc.
    variant VARCHAR(20) NOT NULL,           -- 'front', 'right', 'left', 'main'
    prompt_template TEXT NOT NULL,
    ia_model VARCHAR(50),                   -- NULL = todos los modelos
    is_active BOOLEAN DEFAULT FALSE,
    success_rate DECIMAL(5,2),              -- % éxito histórico
    avg_quality_score DECIMAL(3,2),         -- Rating manual promedio
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),

    UNIQUE(version, variant, ia_model)
);

CREATE INDEX idx_prompt_templates_active ON logo_prompt_templates(is_active, variant)
  WHERE is_active = TRUE;
```

**Ejemplo de datos iniciales**:
```sql
INSERT INTO logo_prompt_templates (version, variant, prompt_template, is_active, notes) VALUES
('v1', 'front',
 'Transform this 2D football team shield into a photorealistic 3D metallic badge. Style: glossy chrome rim, brushed metal center, professional sports badge aesthetic. Lighting: frontal, even illumination, subtle reflections. Background: completely transparent (alpha channel). Preserve all original colors, symbols, and design elements exactly. Output: 1024x1024 PNG.',
 TRUE, 'Prompt inicial para front 3D'),

('v1', 'right',
 'Transform this 2D football team shield into a photorealistic 3D metallic badge rotated 45 degrees to face RIGHT (as if looking at an opponent on the right). Style: glossy chrome rim, brushed metal center. Lighting: left-to-right directional lighting with shadows on the left side. Background: completely transparent. Preserve all original design elements. Output: 1024x1024 PNG.',
 TRUE, 'Prompt inicial para HOME (facing right)'),

('v1', 'left',
 'Transform this 2D football team shield into a photorealistic 3D metallic badge rotated 45 degrees to face LEFT (as if looking at an opponent on the left). Style: glossy chrome rim, brushed metal center. Lighting: right-to-left directional lighting with shadows on the right side. Background: completely transparent. Preserve all original design elements. Output: 1024x1024 PNG.',
 TRUE, 'Prompt inicial para AWAY (facing left)'),

('v1', 'main',
 'Transform this 2D football league/tournament logo into a photorealistic 3D badge. Style: glossy metallic finish, professional sports aesthetic. Lighting: frontal, even illumination. Background: completely transparent. Preserve all original design elements. Output: 1024x1024 PNG.',
 TRUE, 'Prompt para logos de competiciones');
```

**Beneficios**:
- Versionado para rollback si v2 produce peores resultados
- Métricas de éxito por versión de prompt
- A/B testing entre versiones

### 3. CDN Invalidation para Regeneraciones

Cuando se regenera un logo, el cache de Cloudflare debe invalidarse:

```python
import httpx
import logging

logger = logging.getLogger(__name__)

CLOUDFLARE_ZONE_ID = "..."  # Desde config
CLOUDFLARE_API_TOKEN = "..."  # Desde env

async def invalidate_team_logo_cdn(team_id: int, variants: list[str] | None = None):
    """
    Invalidar cache CDN cuando se regenera un logo.

    Args:
        team_id: ID del equipo
        variants: Lista de variantes a invalidar (None = todas)
    """
    if variants is None:
        variants = ['front_3d', 'facing_right', 'facing_left']

    paths_to_purge = []
    sizes = [64, 128, 256, 512]

    for variant in variants:
        # PNG original
        paths_to_purge.append(f"/teams/{team_id}/{variant}.png")
        # WebP thumbnails
        for size in sizes:
            paths_to_purge.append(f"/teams/{team_id}/{variant}_{size}.webp")

    # Cloudflare API - Purge by URL
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.cloudflare.com/client/v4/zones/{CLOUDFLARE_ZONE_ID}/purge_cache",
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "files": [f"https://logos.futbolstats.app{path}" for path in paths_to_purge]
            }
        )

        if response.status_code != 200:
            logger.error(f"CDN purge failed for team {team_id}: {response.text}")
            return False

    logger.info(f"CDN invalidated for team {team_id}: {len(paths_to_purge)} paths")
    return True


async def invalidate_competition_logo_cdn(league_id: int):
    """Invalidar cache CDN para logo de competición."""
    paths_to_purge = [f"/competitions/{league_id}/main.png"]
    for size in [64, 128, 256, 512]:
        paths_to_purge.append(f"/competitions/{league_id}/main_{size}.webp")

    # Similar al anterior...
```

**Trigger**: Automático al completar regeneración exitosa (en `batch_worker.py`).

### 4. Alerting Básico (Prometheus + Grafana)

```python
from prometheus_client import Counter, Gauge, Histogram

# Contadores de errores
logo_batch_errors_total = Counter(
    'futbolstats_logo_batch_errors_total',
    'Total de errores en batch de logos',
    ['phase', 'ia_model', 'error_type']
)

# Gauge para jobs stuck
logo_batch_stuck = Gauge(
    'futbolstats_logo_batch_stuck',
    'Indica si hay batch jobs sin progreso (1=stuck, 0=ok)',
    ['batch_id']
)

# Tasa de error actual
logo_batch_error_rate = Gauge(
    'futbolstats_logo_batch_error_rate',
    'Tasa de error actual del batch (%)',
    ['batch_id']
)

# Histograma de tiempos de generación
logo_generation_duration = Histogram(
    'futbolstats_logo_generation_duration_seconds',
    'Tiempo de generación por imagen',
    ['ia_model', 'variant'],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120]
)

# Costo acumulado
logo_batch_cost_usd = Gauge(
    'futbolstats_logo_batch_cost_usd',
    'Costo acumulado del batch en USD',
    ['batch_id']
)
```

**Alertas Grafana** (agregar a `grafana/alerts/`):

```yaml
# logo_alerts.yaml
groups:
  - name: logo_generation
    rules:
      - alert: LogoBatchStuck
        expr: |
          (time() - futbolstats_logo_batch_last_progress_timestamp) > 1800
          AND futbolstats_logo_batch_status == 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Logo batch {{ $labels.batch_id }} sin progreso"
          description: "El batch lleva más de 30 minutos sin procesar nuevas imágenes"

      - alert: LogoBatchHighErrorRate
        expr: futbolstats_logo_batch_error_rate > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Logo batch {{ $labels.batch_id }} con alta tasa de error"
          description: "Tasa de error: {{ $value }}% (umbral: 10%)"

      - alert: LogoBatchFailed
        expr: futbolstats_logo_batch_status == 3  # 3 = error
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Logo batch {{ $labels.batch_id }} falló"
          description: "El batch terminó en estado de error"
```

### 5. Backup de Originales (R2 Versioning)

Configurar versionado en el bucket R2 para proteger originales:

```python
# En app/logos/config.py

from pydantic_settings import BaseSettings

class LogosR2Settings(BaseSettings):
    """Configuración de R2 para logos."""

    R2_LOGOS_ENABLED: bool = False
    R2_LOGOS_ENDPOINT_URL: str = ""  # https://<account_id>.r2.cloudflarestorage.com
    R2_LOGOS_ACCESS_KEY_ID: str = ""
    R2_LOGOS_SECRET_ACCESS_KEY: str = ""
    R2_LOGOS_BUCKET: str = "futbolstats-logos"

    # CDN
    R2_LOGOS_CDN_URL: str = "https://logos.futbolstats.app"

    # Cloudflare API (para purge)
    CLOUDFLARE_ZONE_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""

    class Config:
        env_prefix = "LOGOS_"
        env_file = ".env"
```

**Lifecycle Rules** (configurar en Cloudflare Dashboard o via API):

```json
{
  "rules": [
    {
      "id": "keep-versions-30d",
      "status": "Enabled",
      "filter": {
        "prefix": "teams/"
      },
      "noncurrent_version_expiration": {
        "noncurrent_days": 30
      }
    },
    {
      "id": "keep-originals-forever",
      "status": "Enabled",
      "filter": {
        "prefix": "teams/",
        "suffix": "/original.png"
      },
      "noncurrent_version_expiration": null
    },
    {
      "id": "abort-multipart-7d",
      "status": "Enabled",
      "abort_incomplete_multipart_upload": {
        "days_after_initiation": 7
      }
    }
  ]
}
```

**Beneficios**:
- Originales nunca se pierden (versiones anteriores preservadas)
- Rollback a versión anterior si regeneración produce peor resultado
- Auditoría de cambios
- Protección contra borrado accidental

---

## Fallback Strategy

```typescript
function getTeamLogoUrl(
  teamId: number,
  variant: 'front' | 'right' | 'left',
  size: 64 | 128 | 256 | 512
): string {
  const logo = teamLogosCache.get(teamId);

  // 1. Si tiene logo 3D ready, usar R2
  if (logo?.status === 'ready' && logo.urls?.[variant]?.[size]) {
    return logo.urls[variant][size];
  }

  // 2. Fallback a API-Football (solo para front, no para facing)
  if (variant === 'front' && logo?.fallback_url) {
    return logo.fallback_url;
  }

  // 3. Para facing sin logo 3D: retornar null (UI muestra placeholder o flat)
  return null;
}
```

---

## Validación de Kimi (Aprobado)

| Pregunta | Respuesta | Status |
|----------|-----------|--------|
| ¿Schema OK? | Sí - `team_logos` + `logo_batch_jobs` + `logo_prompt_templates` | ✅ |
| ¿3 prompts separados? | Sí - Control fino por perspectiva, retry selectivo | ✅ |
| ¿Progress bar en Overview? | Sí - Componente visible durante batch | ✅ |
| ¿Modelo default? | Owner decide - SDXL (económico) o DALL-E (calidad) | ✅ |
| ¿R2 bucket nuevo? | Sí - `futbolstats-logos` (separado de TITAN) | ✅ |
| ¿Resize con Pillow? | Sí - Python backend (consistente con stack) | ✅ |

### Consideraciones Adicionales Aprobadas

- ✅ Validación automática post-IA (size, transparency, corruption)
- ✅ Sistema de plantillas de prompts (versionado)
- ✅ CDN invalidation para regeneraciones
- ✅ Alerting básico (Prometheus + Grafana)
- ✅ Backup de originales (R2 versioning)

---

## Archivos a Crear/Modificar

### Backend

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `migrations/XXX_team_logos.sql` | CREATE | Tablas team_logos, competition_logos, logo_batch_jobs, logo_prompt_templates |
| `app/models.py` | MODIFY | Modelos SQLAlchemy: TeamLogo, CompetitionLogo, LogoBatchJob, LogoPromptTemplate |
| `app/logos/__init__.py` | CREATE | Módulo de logos |
| `app/logos/config.py` | CREATE | LogosR2Settings (configuración R2 + CDN) |
| `app/logos/r2_client.py` | CREATE | Cliente R2 para logos (upload/download/delete) |
| `app/logos/cdn.py` | CREATE | Invalidación CDN Cloudflare |
| `app/logos/ia_generator.py` | CREATE | Integración con DALL-E/SDXL |
| `app/logos/validator.py` | CREATE | Validación post-IA (consideración Kimi) |
| `app/logos/processor.py` | CREATE | Resize/thumbnails con Pillow |
| `app/logos/batch_worker.py` | CREATE | Worker para batch processing |
| `app/logos/prompt_templates.py` | CREATE | Gestión de plantillas de prompts |
| `app/scheduler.py` | MODIFY | Job `logo_resize_pending` |
| `app/main.py` | MODIFY | Endpoints upload/batch/status/review |
| `app/telemetry/metrics.py` | MODIFY | Métricas Prometheus para logos |

### Dashboard

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `dashboard/lib/types/logos.ts` | CREATE | Interfaces TypeScript |
| `dashboard/lib/api/logos.ts` | CREATE | API client |
| `dashboard/lib/hooks/use-logo-batch.ts` | CREATE | Hook para batch status |
| `dashboard/components/settings/LogoGenerator.tsx` | CREATE | UI configuración IA |
| `dashboard/components/settings/LogoUploader.tsx` | CREATE | UI upload masivo |
| `dashboard/components/settings/LeagueSelector.tsx` | CREATE | Selector de liga |
| `dashboard/components/settings/LogoReviewGrid.tsx` | CREATE | Grid de revisión |
| `dashboard/components/overview/LogoBatchProgress.tsx` | CREATE | Progress bar |
| `dashboard/app/api/logos/upload/route.ts` | CREATE | Proxy upload |
| `dashboard/app/api/logos/generate/[leagueId]/route.ts` | CREATE | Proxy generación |
| `dashboard/app/api/logos/review/[leagueId]/route.ts` | CREATE | Proxy revisión |
| `dashboard/app/api/logos/batch/[jobId]/route.ts` | CREATE | Proxy batch status |

### Infraestructura

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `grafana/alerts/logo_alerts.yaml` | CREATE | Alertas de batch |
| `.env.example` | MODIFY | Variables R2/CDN |

---

## Timeline Estimado

| Fase | Scope | Estimado |
|------|-------|----------|
| 0 | Preparación (bucket R2, API keys) | 1h |
| 1 | Schema DB + R2 bucket | 2h |
| 2 | Upload endpoint + validación | 2h |
| 3 | Integración IA (DALL-E/SDXL) | 4h |
| 4 | Batch worker + progress tracking | 4h |
| 5 | Job resize (thumbnails) | 2h |
| 6 | Dashboard UI (upload + config + progress) | 6h |
| 7 | Testing + ajustes | 4h |

**Total**: ~25 horas de trabajo

---

## Plan de Implementación

Plan detallado disponible en: `.claude/plans/partitioned-tickling-curry.md`

### Checklist de Pre-requisitos

- [ ] Bucket R2 `futbolstats-logos` creado
- [ ] Versionado R2 habilitado
- [ ] Lifecycle rules configuradas
- [ ] API keys IA obtenidas (DALL-E/SDXL)
- [ ] Budget inicial aprobado (~$5 para prueba Colombia)

### Orden de Implementación

1. **[Claude]** Actualizar spec con consideraciones Kimi (este commit)
2. **[Master]** Crear bucket R2 + configurar versionado
3. **[Master]** Migración SQL (tables)
4. **[Master]** Cliente R2 + CDN invalidation
5. **[Master]** Generadores IA + validador
6. **[Master]** Batch worker
7. **[Master]** API endpoints
8. **[Claude]** Dashboard UI
9. **[Master]** Scheduler job + alertas Grafana
10. **Testing** Liga Colombia como piloto
