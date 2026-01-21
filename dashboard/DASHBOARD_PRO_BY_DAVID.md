# FutbolStats Ops Dashboard (Master Spec Unificado)

## 0) Resumen ejecutivo (qué estamos construyendo)
Vamos a construir un **Dashboard Ops profesional**, standalone, orientado a operación/observabilidad del sistema FutbolStats.  
**Fase inicial (Fase 0): UI-first**: estructura, navegación, sistema visual, componentes, patrones UX, y mock data.  
**No** se construyen endpoints ni integración real en esta fase; solo se define el **contrato** para migrar a data real luego.

---

## 1) Referencia visual (UI a imitar)
Referencia: **UniFi Network Dashboard**.

### 1.1 Layout confirmado (4 zonas)
- **Top Bar global** (~48–56px): logo + site switcher + app tabs + notificaciones + avatar.
- **Icon Sidebar** (~48–56px): navegación primaria por íconos monocromáticos.
- **Filter Panel** (~200–220px): **colapsable**; filtros con accordions, checkboxes y contadores.
- **Main Content**: tabla (flexible) + paginación.
- **Detail Drawer (inline, no modal)** (~300–320px): panel a la derecha que **empuja** la tabla (sin overlay), con tabs internos.

### 1.2 Patrones UX imprescindibles
- Patrón **Master-Detail**: click en row → abre drawer inline.
- Tabla: **header sticky**, columnas sortables, row hover + selected (tinte azul + borde izq accent).
- Filtros: accordions colapsables, checkboxes con contadores, search interno.
- Tabs: estilo **pill/segmented** (rounded-full).
- Diseño flat: **sin sombras** relevantes; separación por fondos y borders.

---

## 2) Stack tecnológico (decisión cerrada)
**Stack principal (obligatorio):**
- **Next.js 14+ (App Router)**
- **TypeScript**
- **Tailwind CSS**
- **shadcn/ui** (Radix primitives, accesibilidad)
- **TanStack Table v8**
- **TanStack Query (React Query)**
- **Icons**: **Lucide** (outline)

**Notas**
- El sistema debe ser **token-driven** (CSS variables + Tailwind config).
- Preparado para virtualización futura (`@tanstack/react-virtual`) si tablas crecen.

---

## 3) Fases del proyecto (evitar scope creep)
### Fase 0 — UI-first (objetivo actual)
- Layout global completo
- Sistema visual y componentes base
- Secciones navegables (pantallas) con mock data
- Contratos TypeScript estables
- Deep-linking básico (ej: `/matches/123` abre detalle)

**Prohibido en Fase 0**
- Crear/modificar endpoints del backend
- Persistencia real de incidents/ack/resolve
- Settings mutables (keys/feature flags/users) — solo UI placeholder
- Integración con Prometheus/DB real

### Fase 1+ (futuro)
- Conectar endpoints reales / DB views / incidents persistidos / audit / analytics / settings

---

## 4) Diseño: tokens (para clonar estética UniFi)
### 4.1 Paleta (aprox)
- Background base: `#0d0d0d` / `#111111`
- Surface: `#1a1a1a`
- Elevated surface: `#252525` / `#2a2a2a`
- Border: `#2d2d2d` / `#333333`
- Text primary: `#f5f5f5`
- Text secondary: `#9ca3af`
- Text muted: `#6b7280`
- Accent blue: `#3b82f6` / `#006fff`
- Success: `#4ade80`
- Warning: `#eab308`
- Error: `#ef4444`
- Info/cyan: `#06b6d4`

### 4.2 Estilo
- **Sin sombras** (o casi ninguna).
- Bordes sutiles.
- `rounded-full` en tabs/pills.
- Densidad alta: padding 8–12px en celdas y panels.
- Transiciones suaves 150–200ms.

### 4.3 Tipografía
- `Inter` (fallback system-ui).
- Headers tabla 12–13px, cuerpo 13–14px.

---

## 5) Arquitectura UI (estructura de carpetas recomendada)
- `app/` (Next App Router)
  - `(shell)/layout.tsx` → TopBar + IconSidebar + FilterPanel + Content + DrawerSlot
  - `overview/page.tsx`
  - `matches/page.tsx` + `matches/[matchId]/page.tsx` (deep-link)
  - `predictions/page.tsx`
  - `jobs/page.tsx`
  - `incidents/page.tsx`
  - `analytics/page.tsx`
  - `data-quality/page.tsx`
  - `audit/page.tsx`
  - `settings/page.tsx`
- `components/`
  - `shell/TopBar.tsx`, `shell/IconSidebar.tsx`, `shell/FilterPanel.tsx`, `shell/DetailDrawer.tsx`
  - `ui/` (shadcn base + wrappers): Button, Badge, Tabs, Accordion, Input, Select, Tooltip, etc.
  - `tables/DataTable.tsx` (wrapper TanStack)
- `lib/`
  - `types/` (contratos TS)
  - `mocks/` (data mock por sección)
  - `state/` (URL state para filtros, selección)
  - `theme/` (tokens, helpers)

---

## 6) Componentes obligatorios (prioridad de implementación)
### 6.1 Shell global
1. **TopBar / GlobalHeader**
   - logo, site switcher (mock), app tabs (mock), notification bell con badge, avatar/menu (mock)
2. **IconSidebar**
   - icon-only nav + tooltips, active state
3. **FilterPanel**
   - scroll interno, accordions, search interno, checkbox groups
4. **Main content container**
   - tabla + paginación
5. **Detail Drawer inline (no modal)**
   - panel derecho que empuja el contenido (sin overlay)
   - tabs internos (Info/Charts/Settings)

### 6.2 Tabla / filtros
- **DataTable** (TanStack wrapper)
  - sticky header, sorting, row hover/selected, column width, empty/loading
- **Pagination** minimalista (“1–100 of 115”, rows per page, prev/next)
- **StatusBadge / Chips** (Excellent/Good/etc)
- **SegmentedTabs** (pill)

---

## 7) Secciones del dashboard (Fase 0: UI + mocks)
Cada sección debe existir con:
- Subpanel de filtros (aunque sea mock)
- Tabla principal
- Drawer/detail panel (o placeholder si no aplica)

### 7.1 Overview
- Cards de health (System, Preds, Jobs, Live)
- Barra de coverage (mock)
- Upcoming list (mock)
- Active incidents list (mock)
- Timeseries charts: **mock** (placeholder)

### 7.2 Matches
- Tabla: status dot, match, league, kickoff, score, elapsed, prediction badge, model
- Drawer: tabs Overview / Predictions / Live Data (mock)

### 7.3 Predictions
- Coverage card + tabla
- “Missing predictions” destacado (mock)

### 7.4 Jobs
- Scheduler status (mock)
- Job runs table (mock)
- Drawer con Run Details / Logs (logs = mock o “link placeholder”)

### 7.5 Incidents
- Lista + drawer
- Acciones acknowledge/resolve: **UI mock** (sin persistir)

### 7.6 Analytics / Data Quality / Audit / Settings
- UI completa pero **mock/coming soon** según aplique.
- Settings mutables: deshabilitado (read-only UI).

---

## 8) Contratos TypeScript (mínimos, estables)
> Estos tipos deben existir desde Fase 0 para que la migración a data real sea plug-and-play.
> Los contratos marcados con ✅ están implementados; los marcados con 📋 son placeholder para fases futuras.

### ✅ MatchSummary (implementado)
```typescript
interface MatchSummary {
  id: number;
  status: "scheduled" | "live" | "ht" | "ft" | "postponed" | "cancelled";
  leagueName: string;
  leagueCountry: string;
  home: string;
  away: string;
  kickoffISO: string;
  score?: { home: number; away: number };
  elapsed?: { min: number; extra?: number };
  prediction?: {
    model: "A" | "Shadow";
    pick: "home" | "draw" | "away";
    probs?: { home: number; draw: number; away: number };
  };
}
```

### ✅ JobRun (implementado)
```typescript
type JobStatus = "running" | "success" | "failed" | "pending";

interface JobRun {
  id: number;
  jobName: string;                           // e.g. "global_sync", "live_tick"
  status: JobStatus;
  startedAt: string;                         // ISO timestamp
  finishedAt?: string;                       // ISO timestamp (undefined if running/pending)
  durationMs?: number;                       // Runtime in milliseconds
  triggeredBy: "scheduler" | "manual" | "retry";
  error?: string;                            // Error message if failed
  metadata?: Record<string, unknown>;        // Optional extra data
}
```

### ✅ JobDefinition (implementado)
```typescript
interface JobDefinition {
  name: string;
  description: string;
  schedule: string;                          // Cron or human-readable (e.g. "Every 1 minute")
  lastRun?: JobRun;
  nextRunAt?: string;                        // ISO timestamp
  enabled: boolean;
}
```

### ✅ JobFilters (implementado)
```typescript
interface JobFilters {
  status?: JobStatus[];
  jobName?: string[];
  search?: string;
  dateRange?: { start: string; end: string };
}
```

### ✅ Incident (implementado)
```typescript
type IncidentSeverity = "critical" | "warning" | "info";
type IncidentStatus = "active" | "acknowledged" | "resolved";
type IncidentType =
  | "missing_prediction"
  | "job_failure"
  | "api_error"
  | "data_inconsistency"
  | "high_latency"
  | "other";

interface Incident {
  id: number;
  type: IncidentType;
  severity: IncidentSeverity;
  status: IncidentStatus;
  createdAt: string;                         // ISO timestamp
  title: string;
  description?: string;
  entity?: { kind: "match" | "job" | "prediction"; id: number };
  runbook?: { steps: { id: string; text: string; done?: boolean }[] };
  timeline?: { ts: string; message: string }[];
  acknowledgedAt?: string;                   // ISO timestamp
  resolvedAt?: string;                       // ISO timestamp
}
```

### ✅ IncidentFilters (implementado)
```typescript
interface IncidentFilters {
  status?: IncidentStatus[];
  severity?: IncidentSeverity[];
  type?: IncidentType[];
  search?: string;
}
```

### 📋 HealthSummary (placeholder - Phase 1+)
```typescript
interface HealthSummary {
  cards: HealthCard[];
  counts: Record<string, number>;
  coveragePct: number;
}
```

---

## 9) Mock strategy (regla de oro)
- Todos los datos provienen de `lib/mocks/*.json` o factories TS.
- `useMockData = true` global.
- Estructura de mocks por sección: `overview.json`, `matches.json`, `predictions.json`, `jobs.json`, etc.
- Los mocks deben incluir casos: empty, loading, error, long lists (115+).

---

## 10) Criterios de aceptación (Fase 0)
1. **Layout**: TopBar + IconSidebar + FilterPanel + Table + Drawer inline funcional.
2. **Drawer inline**: abre/cierra sin overlay; empuja contenido; mantiene interacción con tabla.
3. **Tabla**: sorting visual, sticky header, row selected persistente mientras drawer abierto.
4. **Filtros**: FilterPanel colapsable (sin perder estado) + accordions colapsables + checkbox counts + search input (aunque sin lógica real).
5. **Navegación**: rutas de todas las secciones existen; estados activos en sidebar.
6. **Deep links**: `/matches/[id]` abre el drawer (mock) del match.
7. **Consistencia visual**: tokens aplicados, sin estilos “default” visibles, dark theme pulido.
8. **A11y base**: focus states visibles, navegación con teclado en tabs/accordion/drawer.

---

## 11) Riesgos y mitigaciones (UI)
- Iconografía específica (UniFi) → mapping con fallback Lucide.
- Tablas anchas → scroll horizontal + column visibility (fase posterior).
- Responsive: en <1280px, drawer puede pasar a modo overlay (pero **mantener inline en desktop**).
- Performance: preparar virtualización para listas grandes (posterior).

---

## 12) Apéndice (Fase 1+): Endpoints sugeridos (NO implementar en Fase 0)
> Estos endpoints son **orientativos** para facilitar la migración a data real. En Fase 0 todo consume **mock data**.

### 12.1 Overview
```
GET /ops/health/summary
GET /ops/predictions/coverage?hours=24
GET /ops/matches/upcoming?hours=6
GET /ops/incidents/active
GET /ops/metrics/timeseries?period=24h
```

### 12.2 Matches
```
GET /ops/matches?status=live&league=X&date_from=X&date_to=X&page=1&limit=100
GET /ops/matches/{match_id}
GET /ops/matches/{match_id}/predictions
GET /ops/matches/{match_id}/events
GET /ops/matches/{match_id}/stats
```

### 12.3 Predictions
```
GET /ops/predictions?status=X&model=X&date_from=X&date_to=X
GET /ops/predictions/coverage?hours=24
GET /ops/predictions/missing
GET /ops/predictions/{prediction_id}
GET /ops/predictions/{prediction_id}/evaluation
```

### 12.4 Jobs
```
GET /ops/scheduler/status
GET /ops/jobs/types
GET /ops/jobs/runs?type=X&status=X&date_from=X&date_to=X
GET /ops/jobs/runs/{run_id}
GET /ops/jobs/runs/{run_id}/logs
GET /ops/jobs/{job_type}/history?limit=50
```

### 12.5 Incidents
```
GET /ops/incidents?status=active&severity=X&type=X
GET /ops/incidents/{incident_id}
POST /ops/incidents/{incident_id}/acknowledge
POST /ops/incidents/{incident_id}/resolve
GET /ops/incidents/{incident_id}/runbook
GET /ops/incidents/{incident_id}/history
```

### 12.6 Analytics
```
GET /ops/analytics/model-performance?model=X&date_from=X&date_to=X
GET /ops/analytics/accuracy?league=X&date_from=X&date_to=X
GET /ops/analytics/system-metrics?period=7d
GET /ops/analytics/api-usage?period=30d
```

### 12.7 Data Quality
```
GET /ops/data-quality/checks
GET /ops/data-quality/checks/{check_id}
GET /ops/data-quality/checks/{check_id}/affected-items
GET /ops/data-quality/checks/{check_id}/history
```

### 12.8 Audit
```
GET /ops/audit/events?type=X&date_from=X&date_to=X&user=X
GET /ops/audit/events/{event_id}
GET /ops/audit/events/{event_id}/context
```

### 12.9 Settings
```
GET /ops/settings
PUT /ops/settings
GET /ops/settings/model-versions
GET /ops/settings/feature-flags
PUT /ops/settings/feature-flags/{flag_id}
GET /ops/settings/users
POST /ops/settings/users/invite
```