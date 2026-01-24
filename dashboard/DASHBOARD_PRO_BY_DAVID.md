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
- **Detail Drawer (overlay docked, NO modal)** (~400px): panel a la derecha que se **superpone** sobre columnas (sin reflow de tabla), sin backdrop, sin bloqueo de interacción. Tabs internos.

### 1.2 Patrones UX imprescindibles
- Patrón **Master-Detail**: click en row → abre drawer overlay (sin reflow de tabla).
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
5. **Detail Drawer overlay docked (no modal)**
   - panel derecho (~400px) que se superpone sobre columnas (sin reflow)
   - sin backdrop, sin bloqueo de interacción
   - tabs internos (Info/Charts/Settings)

### 6.2 Tabla / filtros
- **DataTable** (TanStack wrapper)
  - sticky header, sorting, row hover/selected, column width, empty/loading
  - **columnVisibility** support (TanStack VisibilityState)
- **Pagination** minimalista ("1–100 of 115", rows per page, prev/next)
- **StatusBadge / Chips** (Excellent/Good/etc)
- **SegmentedTabs** (pill)

### 6.3 Left Rail: Customize Columns Panel (UniFi pattern)
El **Left Rail** (columna 2) contiene:
1. **FilterPanel** (existente): accordions, checkboxes, search
2. **CustomizeColumnsPanel** (nuevo): control de visibilidad de columnas

**Comportamiento:**
- El toggle de colapso del Left Rail oculta BOTH (Filters + Customize Columns)
- "Done" en CustomizeColumnsPanel colapsa todo el Left Rail
- "Restore" vuelve a los defaults de columnas
- Cambios aplican inmediatamente (no hay botón "Apply")
- Persistencia en `localStorage` por tabla (keys: `columns:matches`, `columns:jobs`, etc.)

**Implementación:**
- `components/tables/CustomizeColumnsPanel.tsx`: UI del panel
- `lib/hooks/use-column-visibility.ts`: hook con localStorage persistence
- Cada tabla define `COLUMN_OPTIONS` y `DEFAULT_VISIBILITY`
- FilterPanel acepta `children` para renderizar CustomizeColumnsPanel

**UI del panel:**
- Header "Customize Columns"
- Checkbox "All" con estado indeterminado
- Lista de checkboxes (~30 max) con scroll interno
- Footer: "Restore" (defaults) + "Done" (colapsa Left Rail)
- Columnas con `enableHiding: false` no aparecen en el panel (siempre visibles)

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

### ✅ HealthSummary (implementado)
```typescript
type HealthStatus = "healthy" | "warning" | "critical";

interface HealthCard {
  id: string;
  title: string;
  status: HealthStatus;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "stable";
}

interface OverviewCounts {
  matchesLive: number;
  matchesScheduledToday: number;
  incidentsActive: number;
  incidentsCritical: number;
  jobsRunning: number;
  jobsFailedLast24h: number;
  predictionsMissing: number;
  predictionsTotal: number;
}

interface HealthSummary {
  coveragePct: number;
  counts: OverviewCounts;
  cards: HealthCard[];
  lastUpdated: string;                       // ISO timestamp
}
```

### ✅ UpcomingMatch (implementado)
```typescript
interface UpcomingMatch {
  id: number;
  home: string;
  away: string;
  kickoffISO: string;
  leagueName: string;
  hasPrediction: boolean;
}
```

### ✅ ActiveIncident (implementado)
```typescript
interface ActiveIncident {
  id: number;
  title: string;
  severity: "critical" | "warning" | "info";
  createdAt: string;
  type: string;
}
```

### ✅ OverviewData (implementado)
```typescript
interface OverviewData {
  health: HealthSummary;
  upcomingMatches: UpcomingMatch[];
  activeIncidents: ActiveIncident[];
}
```

### ✅ DataQualityCheck (implementado)
```typescript
type DataQualityStatus = "passing" | "warning" | "failing";
type DataQualityCategory = "coverage" | "consistency" | "completeness" | "freshness" | "odds";

interface DataQualityCheck {
  id: number;
  name: string;
  category: DataQualityCategory;
  status: DataQualityStatus;
  lastRunAt: string;                         // ISO timestamp
  currentValue?: number | string;
  threshold?: number | string;
  affectedCount: number;
  description?: string;
}
```

### ✅ DataQualityCheckDetail (implementado)
```typescript
interface DataQualityAffectedItem {
  id: string | number;
  label: string;
  kind: "match" | "team" | "league" | "job";
  details?: Record<string, string | number>;
}

interface DataQualityHistoryEntry {
  timestamp: string;                         // ISO timestamp
  status: DataQualityStatus;
  value?: number | string;
}

interface DataQualityCheckDetail extends DataQualityCheck {
  affectedItems: DataQualityAffectedItem[];
  history: DataQualityHistoryEntry[];
}
```

### ✅ DataQualityFilters (implementado)
```typescript
interface DataQualityFilters {
  status?: DataQualityStatus[];
  category?: DataQualityCategory[];
  search?: string;
}
```

### ✅ AnalyticsReportRow (implementado)
```typescript
type AnalyticsReportType = "model_performance" | "prediction_accuracy" | "system_metrics" | "api_usage";
type AnalyticsReportStatus = "ok" | "warning" | "stale";
type AnalyticsTimeRange = "7d" | "30d" | "90d";

interface AnalyticsReportRow {
  id: number;
  type: AnalyticsReportType;
  title: string;
  periodLabel: string;                       // e.g. "Last 7 days"
  lastUpdated: string;                       // ISO timestamp
  status?: AnalyticsReportStatus;
  summary: Record<string, string | number>;  // e.g. accuracy, brier, p95, errorRate
}
```

### ✅ AnalyticsReportDetail (implementado)
```typescript
interface AnalyticsBreakdownTable {
  columns: string[];
  rows: (string | number)[][];
}

interface AnalyticsSeriesPlaceholder {
  label: string;
  points: number;
}

interface AnalyticsReportDetail {
  row: AnalyticsReportRow;
  breakdownTable?: AnalyticsBreakdownTable;
  seriesPlaceholder?: AnalyticsSeriesPlaceholder[];
}
```

### ✅ AnalyticsFilters (implementado)
```typescript
interface AnalyticsFilters {
  type?: AnalyticsReportType[];
  timeRange?: AnalyticsTimeRange;
  league?: string[];
  search?: string;
}
```

### ✅ AuditEventRow (implementado)
```typescript
type AuditEventType =
  | "job_run"
  | "prediction_generated"
  | "prediction_frozen"
  | "incident_ack"
  | "incident_resolve"
  | "config_changed"
  | "data_quality_check"
  | "system"
  | "user_action";

type AuditSeverity = "info" | "warning" | "error";

type AuditActor =
  | { kind: "user"; id: number; name: string }
  | { kind: "system"; name: string };

interface AuditEventRow {
  id: number;
  timestamp: string;                         // ISO timestamp
  type: AuditEventType;
  severity?: AuditSeverity;
  actor: AuditActor;
  message: string;
  entity?: { kind: "match" | "job" | "prediction" | "incident" | "check"; id: number };
}
```

### ✅ AuditEventDetail (implementado)
```typescript
interface AuditEventContext {
  requestId?: string;
  correlationId?: string;
  ip?: string;
  userAgent?: string;
  env?: "prod" | "staging" | "local";
}

interface AuditRelatedEvent {
  id: number;
  timestamp: string;
  message: string;
}

interface AuditEventDetail extends AuditEventRow {
  context?: AuditEventContext;
  payload?: Record<string, unknown>;
  related?: AuditRelatedEvent[];
}
```

### ✅ AuditFilters (implementado)
```typescript
interface AuditFilters {
  type?: AuditEventType[];
  severity?: AuditSeverity[];
  actorKind?: ("user" | "system")[];
  timeRange?: "1h" | "24h" | "7d" | "30d";
  search?: string;
}
```

### ✅ SettingsSummary (implementado - read-only Phase 0)
```typescript
type SettingsSection = "general" | "timezone" | "notifications" | "api_keys" | "model_versions" | "feature_flags" | "users";
type Environment = "prod" | "staging" | "local";
type ApiKeyStatus = "missing" | "configured" | "invalid";
type UserRole = "admin" | "readonly";

interface ModelVersionInfo {
  modelA: string;
  shadow: string;
  updatedAt: string;                           // ISO timestamp
}

interface FeatureFlag {
  id: string;
  name: string;
  description?: string;
  enabled: boolean;
  updatedAt?: string;                          // ISO timestamp
}

interface SettingsUser {
  id: number;
  email: string;
  role: UserRole;
  lastLogin?: string;                          // ISO timestamp
  createdAt?: string;                          // ISO timestamp
}

interface SettingsSummary {
  lastUpdated: string;                         // ISO timestamp
  environment: Environment;
  timezoneDisplay: string;
  narrativeProvider: string;
  apiFootballKeyStatus: ApiKeyStatus;
  modelVersions: ModelVersionInfo;
  featureFlags: FeatureFlag[];
  users: SettingsUser[];
}
```

### ✅ SettingsFilters (implementado)
```typescript
interface FeatureFlagsFilters {
  search?: string;
  enabled?: boolean;
}

interface UsersFilters {
  search?: string;
  role?: UserRole[];
}
```

### ✅ PredictionRow (implementado)
```typescript
type PredictionStatus = "generated" | "missing" | "frozen" | "evaluated";
type ModelType = "A" | "Shadow";
type PickOutcome = "home" | "draw" | "away";
type MatchResult = "home" | "draw" | "away" | "unknown";
type PredictionTimeRange = "24h" | "48h" | "7d" | "30d";

interface PredictionProbs {
  home: number;
  draw: number;
  away: number;
}

interface PredictionRow {
  id: number;
  matchId: number;
  matchLabel: string;                    // e.g., "Real Madrid vs Barcelona"
  leagueName: string;
  kickoffISO: string;
  model: ModelType;
  status: PredictionStatus;
  generatedAt?: string;                  // ISO timestamp
  probs?: PredictionProbs;
  pick?: PickOutcome;
  result?: MatchResult;                  // if evaluated
}
```

### ✅ PredictionDetail (implementado)
```typescript
interface PredictionFeature {
  name: string;
  value: string | number;
}

interface PredictionEvaluation {
  accuracy?: number;                     // 0-1
  brier?: number;                        // Brier score
  notes?: string;
}

interface PredictionHistoryEntry {
  ts: string;                            // ISO timestamp
  status: PredictionStatus;
  model: ModelType;
}

interface PredictionDetail extends PredictionRow {
  featuresTop?: PredictionFeature[];
  evaluation?: PredictionEvaluation;
  history?: PredictionHistoryEntry[];
}
```

### ✅ PredictionFilters (implementado)
```typescript
interface PredictionFilters {
  status?: PredictionStatus[];
  model?: ModelType[];
  league?: string[];
  timeRange?: PredictionTimeRange;
  search?: string;
}
```

### ✅ PredictionCoverage (implementado)
```typescript
interface PredictionCoverage {
  totalMatches: number;
  withPrediction: number;
  missingCount: number;
  coveragePct: number;
  periodLabel: string;                   // e.g., "Next 24 hours"
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
1. **Layout**: TopBar + IconSidebar + FilterPanel + Table + Drawer overlay funcional.
2. **Drawer overlay docked**: abre/cierra sin reflow de tabla; se superpone (~400px); sin backdrop; tabla interactiva donde no está cubierta.
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
- Responsive: en <1280px, drawer usa Sheet overlay modal; en desktop (>=1280px) drawer es overlay docked (no modal).
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

---

## 13) URL State Convention (Fase 10)

Todas las secciones persisten filtros y selección en la URL para permitir deep-linking, refresh sin perder estado, y URLs compartibles.

### 13.1 Convención de Query Params

| Tipo | Formato | Ejemplo |
|------|---------|---------|
| Selection (id) | `?id=<number>` | `?id=123` |
| Search | `?q=<string>` | `?q=arsenal` |
| Multi-select filter | Repetir param | `?status=live&status=ft` |
| Single-select filter | Param único | `?range=24h` |

### 13.2 URLs por Sección

```
/matches?id=123&status=live&status=ft&league=Premier%20League&q=arsenal
/jobs?id=456&status=running&status=failed&job=global_sync&q=sync
/incidents?id=789&status=active&severity=critical&type=job_failure&q=error
/data-quality?id=101&status=failing&category=coverage&q=match
/analytics?id=201&type=model_performance&q=accuracy
/audit?id=301&type=job_run&severity=error&actor=system&range=24h&q=sync
/predictions?id=401&status=missing&model=A&league=Premier%20League&range=24h&q=real
```

### 13.3 Utilidad Compartida

Todas las funciones de parse/serialize están en `lib/url-state.ts`:

```typescript
// Parse numeric ID from URL (null if invalid)
parseNumericId(param: string | null): number | null

// Parse array param with validation
parseArrayParam<T>(searchParams, key, validValues): T[]

// Parse single param with validation
parseSingleParam<T>(param, validValues): T | null

// Build URLSearchParams from filter state
buildSearchParams(filters): URLSearchParams

// Toggle value in array filter
toggleArrayValue<T>(current, value, checked): T[]
```

### 13.4 Implementación en Pages

Cada page sigue este patrón:

```typescript
// 1. Parse URL state con useMemo
const selectedStatuses = useMemo(
  () => parseArrayParam<Status>(searchParams, "status", VALID_STATUSES),
  [searchParams]
);

// 2. Build URL helper
const buildUrl = useCallback((overrides) => {
  const params = buildSearchParams({ ...currentFilters, ...overrides });
  return `${BASE_PATH}${params.toString() ? `?${params}` : ""}`;
}, [currentFilters]);

// 3. Handler actualiza URL con router.replace
const handleFilterChange = useCallback((value, checked) => {
  const newValues = toggleArrayValue(current, value, checked);
  router.replace(buildUrl({ filter: newValues }), { scroll: false });
}, [current, router, buildUrl]);
```

### 13.5 UX Keyboard

| Tecla | Acción |
|-------|--------|
| `ESC` | Cierra drawer (desktop inline y mobile sheet) |
| `Enter` / `Space` | Abre drawer desde fila enfocada |
| `Arrow Up/Down` | Navega entre filas de la tabla |
| `Home` / `End` | Salta a primera/última fila |

Focus management: Al cerrar drawer, foco vuelve a la fila previamente seleccionada.

---

## 14) Build & Development Notes

### 14.1 Comandos

```bash
cd dashboard
npm install          # Instalar dependencias
npm run dev          # Desarrollo (Turbopack)
npm run build        # Build producción
npm run lint         # Lint check
```

### 14.2 Caveats Conocidos

**Turbopack + Multiple Lockfiles**
Next.js 14+ con Turbopack puede mostrar warning sobre workspace root cuando existen múltiples `package-lock.json` (root + dashboard). Este warning es cosmético y no afecta el build.

```
⚠ Warning: Next.js inferred your workspace root, but it may not be correct.
```

**Solución (opcional)**: Configurar `turbopack.root` en `next.config.ts` o mantener un solo lockfile.

**Turbopack + Sandbox (CI restringido)**
En entornos con sandbox estricto (CI restringido, verificadores aislados), `npm run build` con Turbopack puede **FALLAR** por permisos de binding o escritura:

```
Error: EPERM: operation not permitted, bind
Error: EACCES: permission denied, open '/tmp/.fontsource/...'
```

**Solución**: Ejecutar build en entorno local o CI con permisos normales. El sandbox de Claude Code / verificadores no soporta Turbopack.

### 14.3 Phase 0 Indicators

Todas las secciones muestran claramente que están en Phase 0 con mocks:
- Settings: "Read-only in Phase 0" en controles deshabilitados
- Data tables: Loading/error/empty states con datos mock
- Overview cards: Datos calculados desde mocks

---

## 15) Overview Drawer Routing (SSOT)

El Overview page tiene tiles/cards clickeables que abren un drawer overlay con tabs.
Las keys de panel y tab son canónicas y estables (no renombrar).

### 15.1 Panel Keys

| Key | Component | Description |
|-----|-----------|-------------|
| `overall` | OverallOpsBar | System-wide status rollup |
| `jobs` | JobsCompactTile | Jobs health + runs |
| `predictions` | PredictionsCompactTile | Predictions coverage + missing |
| `fastpath` | FastpathCompactTile | Fastpath/narrative health |
| `pit` | PitProgressCompactTile | PIT evaluation progress |
| `movement` | MovementSummaryTile | Lineup/market movement |
| `sota` | SotaEnrichmentSection | SOTA enrichment status |
| `sentry` | SentryHealthCard | Sentry errors + issues |
| `budget` | ApiBudgetCard | API-Football budget |
| `llm` | LlmCostCard | LLM cost tracking |
| `upcoming` | UpcomingMatchesList | Upcoming matches |
| `incidents` | ActiveIncidentsList | Active incidents |

### 15.2 Tab Keys

| Key | Label | Usage |
|-----|-------|-------|
| `summary` | Summary | Default view from ops/rollup data |
| `issues` | Issues | Sentry issues list (paginated) |
| `timeline` | Timeline | Timeline/history view |
| `missing` | Missing | Predictions missing list (paginated) |
| `movers` | Top Movers | Movement top movers list |
| `runs` | Runs | Job runs history |
| `links` | Links | External links (runbooks, docs) |

### 15.3 Tabs by Panel

| Panel | Available Tabs |
|-------|----------------|
| `overall` | summary |
| `jobs` | summary, runs, links |
| `predictions` | summary, missing |
| `fastpath` | summary, runs |
| `pit` | summary, timeline |
| `movement` | summary, movers |
| `sota` | summary |
| `sentry` | summary, issues |
| `budget` | summary |
| `llm` | summary |
| `upcoming` | summary |
| `incidents` | summary, timeline |

### 15.4 URL Format

```
/overview?panel=<panel>&tab=<tab>
```

Examples:
- `/overview?panel=sentry&tab=issues` - Sentry issues list
- `/overview?panel=predictions&tab=missing` - Missing predictions
- `/overview?panel=movement&tab=movers` - Top movers
- `/overview?panel=jobs` - Jobs summary (tab defaults to summary)

### 15.5 Implementation

- SSOT: `lib/overview-drawer.ts`
- Types: `OverviewPanel`, `OverviewTab`
- Helpers: `parsePanel()`, `parseTab()`, `buildOverviewDrawerUrl()`
- Default tab per panel: `DEFAULT_TAB_BY_PANEL`
- Valid tabs per panel: `TABS_BY_PANEL`
