/**
 * Overview Drawer Routing - SSOT for panel/tab keys
 *
 * This file defines the canonical keys for overview drawer panels and tabs.
 * These keys are used in URL query params and must remain stable.
 *
 * URL format: /overview?panel=<OverviewPanel>&tab=<OverviewTab>
 */

// ============================================================================
// Panel Keys (tiles/cards that can open the drawer)
// ============================================================================

/**
 * Overview panel keys - map 1:1 to tile component names
 */
export const OVERVIEW_PANELS = [
  "overall",     // OverallOpsBar
  "jobs",        // JobsCompactTile
  "predictions", // PredictionsCompactTile
  "fastpath",    // FastpathCompactTile
  "pit",         // PitProgressCompactTile
  "movement",    // MovementSummaryTile
  "sota",        // SotaEnrichmentSection
  "sentry",      // SentryHealthCard
  "budget",      // ApiBudgetCard
  "llm",         // LlmCostCard
  "upcoming",    // UpcomingMatchesList
  "incidents",   // ActiveIncidentsList
  "match",       // Match detail (from Today Matches)
] as const;

export type OverviewPanel = (typeof OVERVIEW_PANELS)[number];

// ============================================================================
// Tab Keys (views within each panel)
// ============================================================================

/**
 * Overview tab keys - common across panels
 */
export const OVERVIEW_TABS = [
  "summary",   // Default summary view (from ops/rollup)
  "issues",    // Sentry issues list
  "timeline",  // Timeline view (placeholder for future)
  "missing",   // Predictions missing list
  "movers",    // Movement top movers
  "runs",      // Job runs history
  "links",     // External links (runbooks, docs)
] as const;

export type OverviewTab = (typeof OVERVIEW_TABS)[number];

// ============================================================================
// Default Tab by Panel
// ============================================================================

/**
 * Default tab to show when opening each panel
 */
export const DEFAULT_TAB_BY_PANEL: Record<OverviewPanel, OverviewTab> = {
  overall: "summary",
  jobs: "summary",
  predictions: "summary",
  fastpath: "summary",
  pit: "summary",
  movement: "summary",
  sota: "summary",
  sentry: "summary",
  budget: "summary",
  llm: "summary",
  upcoming: "summary",
  incidents: "summary",
  match: "summary",
};

// ============================================================================
// Tabs Available by Panel
// ============================================================================

/**
 * Which tabs are available for each panel
 */
export const TABS_BY_PANEL: Record<OverviewPanel, OverviewTab[]> = {
  overall: ["summary"],
  jobs: ["summary", "runs", "links"],
  predictions: ["summary", "missing"],
  fastpath: ["summary", "runs"],
  pit: ["summary", "timeline"],
  movement: ["summary", "movers"],
  sota: ["summary"],
  sentry: ["summary", "issues"],
  budget: ["summary"],
  llm: ["summary"],
  upcoming: ["summary"],
  incidents: ["summary", "timeline"],
  match: ["summary"],
};

// ============================================================================
// Panel Metadata
// ============================================================================

/**
 * Display metadata for each panel
 */
export const PANEL_META: Record<OverviewPanel, { title: string; icon?: string }> = {
  overall: { title: "System Status" },
  jobs: { title: "Jobs Health" },
  predictions: { title: "Predictions" },
  fastpath: { title: "Fastpath" },
  pit: { title: "PIT Progress" },
  movement: { title: "Movement" },
  sota: { title: "SOTA Enrichment" },
  sentry: { title: "Sentry Health" },
  budget: { title: "API Budget" },
  llm: { title: "LLM Cost" },
  upcoming: { title: "Upcoming Matches" },
  incidents: { title: "Active Incidents" },
  match: { title: "Match Details" },
};

/**
 * Display metadata for each tab
 */
export const TAB_META: Record<OverviewTab, { label: string }> = {
  summary: { label: "Summary" },
  issues: { label: "Issues" },
  timeline: { label: "Timeline" },
  missing: { label: "Missing" },
  movers: { label: "Top Movers" },
  runs: { label: "Runs" },
  links: { label: "Links" },
};

// ============================================================================
// Parsing Helpers
// ============================================================================

/**
 * Parse and validate panel from URL query param
 * Returns null if invalid
 */
export function parsePanel(value: string | null | undefined): OverviewPanel | null {
  if (!value) return null;
  const lower = value.toLowerCase();
  if (OVERVIEW_PANELS.includes(lower as OverviewPanel)) {
    return lower as OverviewPanel;
  }
  return null;
}

/**
 * Parse and validate tab from URL query param
 * Returns null if invalid
 */
export function parseTab(value: string | null | undefined): OverviewTab | null {
  if (!value) return null;
  const lower = value.toLowerCase();
  if (OVERVIEW_TABS.includes(lower as OverviewTab)) {
    return lower as OverviewTab;
  }
  return null;
}

/**
 * Validate that a tab is available for a given panel
 */
export function isTabValidForPanel(panel: OverviewPanel, tab: OverviewTab): boolean {
  return TABS_BY_PANEL[panel].includes(tab);
}

/**
 * Get the effective tab for a panel (validates and falls back to default)
 */
export function getEffectiveTab(panel: OverviewPanel, tab: OverviewTab | null): OverviewTab {
  if (tab && isTabValidForPanel(panel, tab)) {
    return tab;
  }
  return DEFAULT_TAB_BY_PANEL[panel];
}

// ============================================================================
// URL Helpers
// ============================================================================

/**
 * Build URL search params for overview drawer
 */
export function buildOverviewDrawerParams(opts: {
  panel: OverviewPanel;
  tab?: OverviewTab;
  matchId?: number;
}): URLSearchParams {
  const params = new URLSearchParams();
  params.set("panel", opts.panel);
  if (opts.tab && opts.tab !== DEFAULT_TAB_BY_PANEL[opts.panel]) {
    params.set("tab", opts.tab);
  }
  if (opts.matchId !== undefined) {
    params.set("matchId", opts.matchId.toString());
  }
  return params;
}

/**
 * Build full URL path for overview drawer
 */
export function buildOverviewDrawerUrl(opts: {
  panel: OverviewPanel;
  tab?: OverviewTab;
  matchId?: number;
}): string {
  const params = buildOverviewDrawerParams(opts);
  return `/overview?${params.toString()}`;
}

/**
 * Parse drawer state from URL search params
 */
export function parseDrawerStateFromParams(searchParams: URLSearchParams): {
  panel: OverviewPanel | null;
  tab: OverviewTab | null;
  matchId: number | null;
} {
  const panel = parsePanel(searchParams.get("panel"));
  const tab = parseTab(searchParams.get("tab"));
  const matchIdStr = searchParams.get("matchId");
  const matchId = matchIdStr ? parseInt(matchIdStr, 10) : null;
  return { panel, tab, matchId: Number.isNaN(matchId) ? null : matchId };
}
