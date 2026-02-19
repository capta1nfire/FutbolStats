# ADS — Design System Spec (v0) · Bon Jogo Ops Dashboard

> **Rol**: ADS (Auditor Design System) define estándares visuales.  
> **Implementación**: Claude (codificador dashboard) ejecuta cambios en UI siguiendo este spec.  
> **Modo**: **Dark-only** (por decisión actual). Light Mode no se implementa todavía.

---

## 0) Objetivo

Establecer una **fuente de verdad** para la **consistencia visual** del dashboard, alineado al “look” **UniFi Network**:

- **Enterprise-dense**, desktop-first.
- **Flat**: separación por **surfaces + dividers** (alpha 7%) y sombras mínimas.
- **Color contenido**: el azul primario y colores de estado se usan como **señal**, no como bloque dominante.
- **Evitar duplicación**: un solo patrón para Drawer, Tabs, Table, Badges y Tooltip.

---

## 1) Fuente de verdad (SSOT)

- **Tokens**: `dashboard/app/globals.css` (CSS variables + `@theme inline`).
- **Primitivos UI**: `dashboard/components/ui/` (shadcn/Radix wrappers).
- **Shell & patrones**:
  - `dashboard/components/shell/DetailDrawer.tsx` (drawer canónico)
  - `dashboard/components/shell/FilterPanel.tsx`
  - `dashboard/components/shell/TopBar.tsx`
  - `dashboard/components/shell/IconSidebar.tsx`

> Regla: **No hardcodear** colores/sombras/spacing en páginas. Si falta un token/utility, se agrega al SSOT.

---

## 2) Tokens canónicos (dark-only)

### 2.1 Colores (semánticos)

**Fuente actual**: `globals.css` ya define la mayoría. Falta estandarizar “strong text” y tooltip/shadows como tokens de sistema.

```text
BG / Surfaces
- --background:       #131416
- --surface:          #1c1e21
- --surface-elevated: #232326
- --accent:           #282b2f          (selected/hover surface)

Text
- --foreground:       #dee0e3
- --muted-foreground: #b7bcc2
- [ADD] --foreground-strong: #ffffff   (para KPIs/texto enfático; evitar hardcode #fff)

Interactive
- --primary:          #4797ff
- --primary-hover:    #71b7ff
- --ring:             #4797ff

Lines
- --border:           rgba(249,250,250,0.07)  (divider signature)
- --input:            rgba(249,250,250,0.10)

Status (base)
- --success:          #37be5f
- --warning:          #eab308
- --error:            #ef4444
- --info:             #06b6d4
```

### 2.2 Status “3 canales” (text/bg/border)

**Regla**: el sistema usa un set semántico **único** para status. Badges, rows, banners, etc. consumen esto.  
**Prohibido**: `bg-green-500/15`, `border-red-500/30`, etc.

```text
[ADD] --status-success-text / bg / border
[ADD] --status-warning-text / bg / border
[ADD] --status-error-text   / bg / border
[ADD] --status-info-text    / bg / border

Valores sugeridos (derivados del JSON de Bon Jogo):
- bg:    rgba(color, 0.15)
- border rgba(color, 0.30)
- text:  color sólido
```

> Nota: si preferimos OKLCH/OKLAB para mayor “control perceptual”, se mantiene, pero el **consumo** siempre es vía tokens semánticos.

### 2.3 Radius

```text
--radius-sm: 4px
--radius-md: 6px
--radius-lg: 8px
--radius-full: 9999px
```

**Decisión ADS**: `md=6px` es el “default” de interacción (inputs/buttons/tabs). Evitar mezclar 4/6 sin motivo.

### 2.4 Sombras (mínimas, pero consistentes)

Actualmente existe `.shadow-elevation-left` con:

```text
box-shadow: -8px 0 16px rgba(0,0,0,0.28)
```

**Decisión ADS** (UniFi-like): el Drawer debe “separar” más fuerte. Se recomienda introducir un token/utility:

```text
[ADD] --shadow-drawer-left: -4px 0 48px rgba(0,0,0,1), 0 0 1px rgba(249,250,250,0.08)
[ADD] --shadow-tooltip:     0 8px 24px rgba(0,0,0,1), 0 0 1px rgba(249,250,250,0.08)
[ADD] --tooltip-bg:         (preferido: --surface-elevated o token dedicado)
```

> Regla: **no** usar `shadow-[...]` inline en componentes.

### 2.5 Spacing y tipografía (resumen operativo)

**Spacing**: baseline 4px; se aceptan 6px en micro-ajustes si está tokenizado/preset (ej. densidad tabla).  
**Tipografía**: Inter como sans.  
**Micro sizes**: 10px y 11px existen en UI; deben ser **tokens** (no `text-[11px]` arbitrario).

---

## 3) Component recipes (canónicos)

## 3.1 Drawer (side panel) — **un solo patrón**

**Component canonical**: `DetailDrawer` (y todas las secciones deben converger aquí).

```text
Ancho:            400px (fijo)
BG:               bg-sidebar (≈ --background)
Separación:       shadow-drawer-left (no border-left como separador primario)
Layout:           header fijo + (tabs opcionales) + scroll interno + footer acciones
Desktop:          docked/inline (sin overlay modal)
Mobile <1280:     Sheet (Radix) como fallback
```

**Header**

```text
Padding:          px-4 py-3 (12px 16px)
Title:            14px, 600
Close:            icon button ghost, 32px
```

**Reglas**
- **DO**: mismo BG + shadow + padding en todos los drawers (incluye Overview/Sentry/Health).
- **DON’T**: mezclar drawers `fixed + border-l` con drawers `absolute + shadow` en desktop.

## 3.2 DrawerTabs — icon segmented (estándar)

**Uso**: tabs dentro de drawer (Info/Logs/Timeline/etc).  
**No usar** pill tabs primarios dentro del drawer.

```text
Container:
- bg: accent/50
- radius: 6px (md)
- padding: 2px (0.5)
- gap: 2px

Item:
- height: 32px
- radius: 6px (md)  ← decisión ADS para coherencia (evitar 4px aquí)
- padding: 4px

Active:
- bg: accent
- text/icon: primary

Inactive:
- bg: transparent
- text/icon: muted-foreground
- hover: text-foreground
```

## 3.3 Tabs “pill” (permitidas SOLO fuera del drawer)

Ejemplo: navegación de topbar o filtros macro.  
**Regla**: si el tab es navegación interna de un panel, debe ser `DrawerTabs` (icon).

## 3.4 Table system (único)

**Decisión ADS**: **un solo sistema de tabla** (`DataTable`). No crear "dos tablas" divergentes.

### Densidad unificada (P1 decisión)

> **Decisión P1**: No se crean presets de densidad (`compact`/`stacked`) porque DataTable
> ya tiene un sistema unificado. Todas las tablas (Jobs, Matches, Incidents, Audit, DataQuality)
> comparten el mismo spacing: header `px-3 pt-3 pb-2` (~36px), cell `px-3 py-2.5` (~44px).
> La diferencia visual entre "compact" (Jobs) y "stacked" (Matches) es contenido de celda
> (1 línea vs 2 líneas con `flex-col gap-0.5`), no padding ni row height distinto.
> Si en el futuro se necesitan presets reales, se agregarán como prop de DataTable.

```text
Header:     px-3 pt-3 pb-2 (~36px)
Cell:       px-3 py-2.5 (~44px row)
Hover:      bg-accent/50
Selected:   bg-primary/10 border-l-2 border-l-primary
Dividers:   border-border (7% alpha)
Typography: primary text-sm; meta text-xs muted
```

**Reglas**
- **DO**: usar DataTable para todas las tablas; diferencias de layout van dentro de la celda.
- **DON'T**: cada sección inventa su propio padding/line-height/row height.

## 3.5 Badges — Status chips (`StatusChip`)

**Component**: `dashboard/components/ui/status-chip.tsx`

**Base**: `inline-flex items-center rounded-full text-xs px-2 py-0.5 whitespace-nowrap`.

**Status mapping**:
- `success`: Passing / Success / Generated / OK / Ready
- `warning`: Warning / Frozen / Stale
- `error`: Critical / Failed / Missing
- `info`: Building / Running / Info

**Reglas**
- **DO**: usar `status-*-(text/bg/border)` en todos los badges de estado.
- **DON'T**: `bg-green-500/15`, `border-yellow-500/30`, etc.

**W/D/L (resultados de partido)**: Win → `status-success`, Loss → `status-error`, Draw → **neutral** (`text-muted-foreground`). Draw NO es warning — es un resultado neutral sin connotación negativa.

## 3.6 Badges — Tag chips (`TagChip`)

**Component**: `dashboard/components/ui/tag-chip.tsx`

**Propósito**: Badges de **identidad/tipo/categoría** — NO expresan estado (ok/warn/error).
Ejemplos: tipo de auditoría, categoría de data quality, tier PROD/TITAN, tipo de torneo.

**Base**: `inline-flex items-center gap-1 rounded-full text-xs font-medium px-2 py-0.5 border whitespace-nowrap`.

**Paleta (7 tonos)**:

| Tono | Token prefix | Uso típico |
|------|-------------|------------|
| `purple` | `--tag-purple-*` | Internacional, TITAN |
| `cyan` | `--tag-cyan-*` | Odds, coverage |
| `pink` | `--tag-pink-*` | Scraping |
| `indigo` | `--tag-indigo-*` | Matching, TITAN tier |
| `blue` | `--tag-blue-*` | PROD tier, league |
| `orange` | `--tag-orange-*` | Cup, medium priority |
| `gray` | `--tag-gray-*` | Default/unknown |

**Restricciones**:
- **NO green/emerald**: reservados exclusivamente para `status-success`.
- **Máximo 7 tonos**: no agregar más sin aprobación ADS.
- **amber → orange**: no existe tono amber; usar orange como equivalente.

**Reglas**
- **DO**: usar TagChip para badges que clasifican tipo/categoría/tier.
- **DON'T**: usar TagChip para estados (ok/warn/error) — eso es StatusChip.
- **DON'T**: usar green/emerald en tags — visualmente confunde con "success".

## 3.7 Tile / Card convention

**Decisión P2**: No se crean componentes wrapper (`<Panel>`, `<TileCard>`). Las clases literales
son claras y grep-able. Se documenta la convención para consistencia.

### Surfaces

| Contexto | Clase | Token |
|----------|-------|-------|
| Card/Tile standard | `bg-surface` | `--surface: #1c1e21` |
| Background/nested | `bg-background` | `--background: #131416` |
| Legacy (equivalent) | `bg-card` | `--card: #1c1e21` (idéntico a surface) |

> `bg-surface` es el estándar forward. `bg-card` es legacy pero visualmente idéntico.
> Nuevos componentes deben usar `bg-surface`.

### Padding

| Tipo | Padding | Uso |
|------|---------|-----|
| Card grande | `p-4` (16px) | Cards con múltiples secciones (Overview, ML Health) |
| Tile compacto | `p-3` (12px) | Tiles de grid (compact tiles, SOTA cards) |

### Border y radius

```text
border border-border rounded-lg
```

Todos los tiles/cards usan `rounded-lg` (8px) y `border-border` (7% alpha).

### Ejemplo canónico

```tsx
// Card grande
<div className="bg-surface border border-border rounded-lg p-4">

// Tile compacto
<div className="bg-surface border border-border rounded-lg p-3 h-full flex flex-col">
```

## 3.8 Tooltip

```text
BG:           tooltip-bg (o surface-elevated si se decide)
Text:         popover-foreground o foreground-strong (según contraste)
Font-size:    11px
Padding:      8px 16px
Radius:       4px
Shadow:       shadow-tooltip (utility/token)
```

**Regla**: tooltip nunca usa colores/sombras hardcode.

---

## 4) Estados (consistencia obligatoria)

### 4.1 Hover / Selected

- Hover: `bg-accent/50`
- Selected: `bg-accent/30` (o token `--bg-selected`)
- Divider: `border-border` siempre.

### 4.2 Focus

Se mantiene **ring 2px** por accesibilidad, pero debe ser uniforme.

```text
focus-visible: ring-2 ring-ring ring-offset-2 ring-offset-background
```

---

## 5) DO / DON’T (reglas duras)

### DO
- **Token-driven**: colores, sombras, spacing y tamaños críticos salen de tokens/utilities.
- **Unificar antes de inventar**: si un patrón existe (DrawerTabs/Table density), se reutiliza.
- **Dividers 7%** como firma visual del sistema.

### DON’T
- No hardcodear hex/rgba en componentes (`bg-[#...]`, `shadow-[...]`, `border-[#...]`).
- No usar `*-500/15` para estados en producción UI; usar tokens semánticos.
- No introducir un “drawer especial” por sección.

---

## 6) Backlog de estandarización (P0/P1)

### P0 (rompe coherencia hoy)
- **Drawer unificado**: todos los drawers convergen a `DetailDrawer` (mismo BG + shadow + layout).
- **DrawerTabs unificado**: eliminar pill tabs primarios dentro de drawers; usar icon segmented.
- **Sombra de drawer**: introducir `shadow-drawer-left` (UniFi-like) y aplicarlo siempre (incluye Overview).
- **Tooltip tokenizado**: mover bg/shadow a tokens/utilities.

### P1 (deuda frecuente / anti-drift)
- **Badges semánticos**: migrar clases `bg-*-500/15` a tokens `status-*`.
- **Table density presets**: implementar presets `compact/stacked` en un solo sistema de tabla.
- **Micro-typography tokens**: 10px y 11px como tokens (no `text-[...]`).
- **Motion**: normalizar a una sola “familia” (duración + easing).

---

## 7) Checklist de revisión (para PRs UI)

- [ ] No hay colores hardcodeados en nuevos componentes.
- [ ] Drawer usa el patrón canónico (BG + shadow + padding).
- [ ] Tabs dentro del drawer usan `DrawerTabs` (icon).
- [ ] Tabla usa preset de densidad (compact/stacked) definido por el sistema.
- [ ] Badges consumen tokens semánticos (text/bg/border).
- [ ] Tooltip consume tokens/utilities (sin inline shadow).
- [ ] `npm run lint:visual` pasa sin violations.

---

## 8) Anti-drift / Visual Lint (P3)

### Script

```bash
npm run lint:visual          # STRICT (default) — falla si hay violations
npm run lint:visual:report   # Solo reporte, exit 0 siempre
npm run lint:all             # eslint + tsc --noEmit + visual lint
```

**Implementación**: `dashboard/scripts/visual-lint.mjs` (Node.js puro, sin dependencias externas).

### Reglas

| ID | Pattern | Qué detecta |
|----|---------|-------------|
| R1 | `bg-[#` | Hex background hardcodeado |
| R2 | `text-[#` | Hex text hardcodeado |
| R3 | `border-[#` | Hex border hardcodeado |
| R4 | `shadow-[` | Shadow inline |
| R5 | `(bg\|text\|border)-(green\|yellow\|red\|blue\|purple\|emerald)-{300..900}` | Palette status sin token |
| R6 | `(bg\|text\|border)-*-500/` | Opacity variants sin token |

**Scope**: `dashboard/components/` y `dashboard/app/`.
**Excluye**: `globals.css`, `node_modules/`, `.next/`, archivos `.test.`/`.spec.`/`.stories.`, líneas de comentario.

### Allowlist

Archivo: `dashboard/visual-lint.allowlist.json`

Cada entrada debe tener:
- `file`: glob del archivo
- `pattern`: regex del match permitido
- `reason`: por qué es válido
- `owner`: quién aprobó la excepción (`ADS` o `Claude`)
- `removeBy` (opcional): cuándo se planea migrar

**No agregar entries sin motivo documentado.** La allowlist no debe crecer sin control.

### Cómo agregar una excepción

1. Verificar que el color es genuinamente data-viz o decorativo (NO status ni tag)
2. Agregar entry en `visual-lint.allowlist.json` con los 4 campos obligatorios
3. Re-ejecutar `npm run lint:visual` para confirmar que pasa

