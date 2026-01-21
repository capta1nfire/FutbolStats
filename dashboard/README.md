# FutbolStats Ops Dashboard

Operations dashboard for FutbolStats - Phase 0 (UI-first)

## Quick Start

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Stack

- **Next.js 14+** (App Router)
- **TypeScript**
- **Tailwind CSS** (dark theme, UniFi-inspired)
- **shadcn/ui** (Radix primitives)
- **TanStack Table v8** (DataTable)
- **TanStack Query** (data fetching)
- **Lucide Icons**

## Project Structure

```
dashboard/
├── app/
│   ├── layout.tsx              # Root layout (Inter font, providers)
│   ├── globals.css             # Tailwind + design tokens
│   ├── page.tsx                # Redirects to /overview
│   └── (shell)/                # Route group with shell layout
│       ├── layout.tsx          # TopBar + IconSidebar + main
│       ├── overview/           # Health cards, coverage, upcoming
│       ├── matches/            # Table + filters + drawer
│       ├── jobs/               # Job runs table
│       ├── incidents/          # Active incidents
│       ├── data-quality/       # DQ checks
│       ├── analytics/          # Reports (model perf, etc.)
│       ├── audit/              # Audit trail
│       ├── predictions/        # Prediction coverage
│       └── settings/           # Read-only config
├── components/
│   ├── shell/                  # TopBar, IconSidebar, FilterPanel, DetailDrawer
│   ├── tables/                 # DataTable (TanStack wrapper)
│   ├── [section]/              # Section-specific components
│   ├── ui/                     # shadcn/ui components
│   └── providers.tsx           # TanStack Query provider
├── lib/
│   ├── types/                  # TypeScript contracts
│   ├── mocks/                  # Mock data and factories
│   ├── hooks/                  # Data fetching hooks
│   ├── url-state.ts            # URL state utilities
│   └── utils.ts                # cn utility
└── DASHBOARD_PRO_BY_DAVID.md   # SSOT spec
```

## Design Tokens

Located in `app/globals.css`. Key tokens:

| Token | Value | Usage |
|-------|-------|-------|
| `--background` | `#0d0d0d` | Page background |
| `--surface` | `#1a1a1a` | Cards, panels |
| `--surface-elevated` | `#252525` | Popovers |
| `--foreground` | `#f5f5f5` | Primary text |
| `--muted-foreground` | `#9ca3af` | Secondary text |
| `--primary` | `#3b82f6` | Accent blue |
| `--border` | `#2d2d2d` | Borders |
| `--success` | `#4ade80` | Success state |
| `--warning` | `#eab308` | Warning state |
| `--error` | `#ef4444` | Error state |

## Mock Data

Mocks are configured in `lib/mocks/config.ts`:

```typescript
export const mockConfig = {
  useMockData: true,
  simulateLatency: 800,
  scenario: "normal" | "empty" | "error" | "large"
};
```

Change `scenario` to test different states.

## URL State Convention

All pages persist filters and selection in URL query params for deep-linking and shareability.

### Query Param Convention

| Type | Format | Example |
|------|--------|---------|
| Selection | `?id=<number>` | `?id=123` |
| Search | `?q=<string>` | `?q=arsenal` |
| Multi-select | Repeated params | `?status=live&status=ft` |
| Single-select | Single param | `?range=24h` |

### Example URLs by Section

```
/matches?id=123&status=live&status=ft&league=Premier%20League&q=arsenal
/jobs?id=456&status=running&status=failed&job=global_sync&q=sync
/incidents?id=789&status=active&severity=critical&type=job_failure&q=error
/data-quality?id=101&status=failing&category=coverage&q=match
/analytics?id=201&type=model_performance&q=accuracy
/audit?id=301&type=job_run&severity=error&actor=system&range=24h&q=sync
/predictions?id=401&status=missing&model=A&league=Premier%20League&range=24h&q=real
```

### Deep Links

- **Canonical URL**: `/matches?id=123`
- **Deep-link**: `/matches/123` → redirects to canonical
- Uses `router.replace` with `scroll: false` (no history pollution)

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `ESC` | Close drawer |
| `Enter`/`Space` | Open drawer from focused row |
| `Arrow Up/Down` | Navigate table rows |
| `Home`/`End` | Jump to first/last row |

## Breakpoints

- **Desktop (≥1280px)**: Drawer inline, pushes content (no overlay)
- **Mobile/Tablet (<1280px)**: Drawer as Sheet overlay

Uses `useSyncExternalStore` for SSR-safe media query detection.

## Phase 0 Scope

✅ Implemented:
- Shell layout (TopBar, IconSidebar, FilterPanel collapsible)
- All sections: Matches, Jobs, Incidents, Data Quality, Analytics, Audit, Predictions, Settings
- Full URL state persistence for all filters and selection
- Mock data with loading/empty/error states
- Dark theme with UniFi-inspired tokens
- Keyboard navigation (ESC, Enter, arrows)
- Accessible with aria-labels

🚫 Not in Phase 0:
- Backend integration
- Real data
- Settings mutations (read-only UI)

## Known Caveats

1. **Turbopack + Multiple Lockfiles**: Warning about workspace root when multiple `package-lock.json` files exist. Cosmetic only, does not affect build.

2. **Turbopack + Sandbox**: In restricted sandbox environments (CI, verifiers), `npm run build` may **fail** with `EPERM: operation not permitted` or `EACCES: permission denied`. Run build in local or normal CI environment.

3. **SSR Media Query**: On server-render, `useIsDesktop()` defaults to `false` (mobile). Desktop users may see brief Sheet→inline drawer transition.

4. **Mock Data Only**: All data is client-side mocks. No API calls.

5. **No Authentication**: Dashboard is fully open. No auth flow implemented.

## Acceptance Criteria Checklist

- [ ] Layout: TopBar + IconSidebar + FilterPanel + Table + Drawer inline
- [ ] Drawer: inline (desktop), pushes content, no overlay
- [ ] Table: sticky header, sorting, row hover/selected
- [ ] Filters: collapsible panel, accordions, checkboxes with counts
- [ ] Navigation: all routes exist, no 404, active states
- [ ] Deep links: `/matches/123` works
- [ ] Visual: dark theme, no shadows, tokens applied
- [ ] A11y: focus states visible, keyboard navigation
- [ ] States: loading, empty, error with retry
