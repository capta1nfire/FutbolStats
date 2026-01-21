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
│       ├── overview/           # Placeholder
│       ├── matches/            # Full implementation
│       │   ├── page.tsx        # Matches table + filters + drawer
│       │   └── [matchId]/      # Deep-link redirect
│       ├── predictions/        # Placeholder
│       ├── jobs/               # Placeholder
│       ├── incidents/          # Placeholder
│       ├── analytics/          # Placeholder
│       ├── data-quality/       # Placeholder
│       ├── audit/              # Placeholder
│       └── settings/           # Placeholder
├── components/
│   ├── shell/                  # TopBar, IconSidebar, FilterPanel, DetailDrawer
│   ├── tables/                 # DataTable (TanStack wrapper), Pagination
│   ├── matches/                # MatchesTable, MatchDetailDrawer, StatusDot
│   ├── ui/                     # shadcn/ui components
│   └── providers.tsx           # TanStack Query provider
├── lib/
│   ├── types/                  # TypeScript contracts (MatchSummary, etc.)
│   ├── mocks/                  # Mock data and factories
│   ├── hooks/                  # useMatches, useMatch
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

## URL Routing

- **Canonical URL**: `/matches?id=123`
- **Deep-link**: `/matches/123` → redirects to canonical
- Uses `router.replace` with `scroll: false` (no history pollution)

## Breakpoints

- **Desktop (≥1280px)**: Drawer inline, pushes content (no overlay)
- **Mobile/Tablet (<1280px)**: Drawer as Sheet overlay

Uses `useSyncExternalStore` for SSR-safe media query detection.

## Phase 0 Scope

✅ Implemented:
- Shell layout (TopBar, IconSidebar, FilterPanel collapsible)
- Matches page with table, filters, inline drawer
- URL sync for match selection
- Mock data with loading/empty/error states
- Dark theme with UniFi-inspired tokens

🚫 Not in Phase 0:
- Backend integration
- Real data
- Settings mutations
- Analytics charts

## Known Limitations

1. **SSR Media Query Flicker**: On server-render, `useIsDesktop()` defaults to `false` (mobile).
   Desktop users may see a brief flicker from Sheet to inline drawer on first load.

2. **Filter State Not Persisted**: Filter selections reset on page refresh.
   URL only persists the selected match ID.

3. **Mock Data Only**: All data is client-side mocks. No API calls.

4. **No Authentication**: Dashboard is fully open. No auth flow implemented.

5. **Single Language**: UI is English-only. No i18n support.

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
