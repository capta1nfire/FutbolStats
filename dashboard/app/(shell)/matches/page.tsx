"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMatchesApi, useMatch, useColumnVisibility, usePageSize } from "@/lib/hooks";
import { MatchSummary, MatchFilters, MatchStatus, MATCH_STATUSES } from "@/lib/types";
import { getMatchesMockSync } from "@/lib/mocks";
import {
  MatchesTable,
  MatchesFilterPanel,
  MatchDetailDrawer,
  MATCHES_COLUMN_OPTIONS,
  MATCHES_DEFAULT_VISIBILITY,
  type MatchesView,
  type TimeRange,
  type DateRangeValue,
} from "@/components/matches";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";

const BASE_PATH = "/matches";

/** Valid view values */
const VALID_VIEWS: MatchesView[] = ["upcoming", "finished", "calendar"];

/** Valid time range values */
const VALID_TIME_RANGES: TimeRange[] = ["24h", "48h", "7d"];

/** Convert time range to hours for API */
function timeRangeToHours(range: TimeRange): number {
  switch (range) {
    case "24h": return 24;
    case "48h": return 48;
    case "7d": return 168;
    default: return 168;
  }
}

/**
 * Matches Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function MatchesPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const selectedMatchId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );

  // Parse view from URL (default: upcoming)
  const activeView = useMemo((): MatchesView => {
    const viewParam = searchParams.get("view");
    if (viewParam && VALID_VIEWS.includes(viewParam as MatchesView)) {
      return viewParam as MatchesView;
    }
    return "upcoming";
  }, [searchParams]);

  // Parse time range from URL (default: 7d)
  const selectedTimeRange = useMemo((): TimeRange => {
    const rangeParam = searchParams.get("range");
    if (rangeParam && VALID_TIME_RANGES.includes(rangeParam as TimeRange)) {
      return rangeParam as TimeRange;
    }
    return "7d";
  }, [searchParams]);

  // Parse selected statuses from URL (for calendar view)
  const selectedStatuses = useMemo(
    () => parseArrayParam<MatchStatus>(searchParams, "status", MATCH_STATUSES as unknown as MatchStatus[]),
    [searchParams]
  );

  // Parse selected leagues from URL
  const selectedLeagues = useMemo(() => {
    const leagueParam = searchParams.getAll("league");
    return leagueParam.length > 0 ? leagueParam : [];
  }, [searchParams]);
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

  // Parse custom date range from URL (for calendar view)
  const customDateRange = useMemo((): DateRangeValue | undefined => {
    const fromParam = searchParams.get("from");
    const toParam = searchParams.get("to");
    if (fromParam && toParam) {
      const from = new Date(fromParam);
      const to = new Date(toParam);
      if (!isNaN(from.getTime()) && !isNaN(to.getTime())) {
        return { from, to };
      }
    }
    // Default: last 7 days
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    return { from: weekAgo, to: now };
  }, [searchParams]);

  // Normalize URL if id param is invalid
  const selectedIdParam = searchParams.get("id");
  useEffect(() => {
    if (selectedIdParam && selectedMatchId === null) {
      const params = new URLSearchParams(searchParams.toString());
      params.delete("id");
      const search = params.toString();
      router.replace(`${BASE_PATH}${search ? `?${search}` : ""}`, { scroll: false });
    }
  }, [selectedIdParam, selectedMatchId, router, searchParams]);

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("matches");

  // Column visibility with localStorage persistence
  const {
    columnVisibility,
    setColumnVisibility,
    setColumnVisible,
    resetToDefault,
  } = useColumnVisibility("matches", MATCHES_DEFAULT_VISIBILITY);

  // Construct filters for query (used for both API and mock fallback)
  const filters: MatchFilters = useMemo(() => {
    // Map view to status filter
    let status: MatchFilters["status"];
    if (activeView === "upcoming") {
      status = ["scheduled"];
    } else if (activeView === "finished") {
      status = ["ft"];
    } else {
      // Calendar view: use selected statuses or show all
      status = selectedStatuses.length > 0 ? selectedStatuses : undefined;
    }

    return {
      status,
      leagues: selectedLeagues.length > 0 ? selectedLeagues : undefined,
      search: searchValue || undefined,
    };
  }, [activeView, selectedLeagues, selectedStatuses, searchValue]);

  // Calculate hours for API based on view
  const hoursForApi = useMemo(() => {
    if (activeView === "calendar" && customDateRange) {
      // Calculate hours from date range
      const diffMs = customDateRange.to.getTime() - customDateRange.from.getTime();
      return Math.ceil(diffMs / (1000 * 60 * 60));
    }
    return timeRangeToHours(selectedTimeRange);
  }, [activeView, customDateRange, selectedTimeRange]);

  // Fetch data from API with view-specific status
  const {
    matches: apiMatches,
    pagination,
    isDegraded,
    isLoading,
    error,
    refetch,
  } = useMatchesApi({
    status: filters.status,
    hours: hoursForApi,
    page: currentPage,
    limit: pageSize,
  });

  // Use API data if available, fallback to mocks
  const mockMatches = useMemo(() => getMatchesMockSync(filters), [filters]);
  const matches = apiMatches ?? mockMatches;

  const { data: selectedMatch } = useMatch(selectedMatchId);

  // Drawer is open when there's a selected match
  const drawerOpen = selectedMatchId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
      view?: MatchesView;
      range?: TimeRange;
      league?: string[];
      status?: MatchStatus[];
      q?: string;
      from?: string;
      to?: string;
    }) => {
      const view = overrides.view ?? activeView;
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedMatchId : overrides.id,
        view,
        // Only include range for upcoming/finished views
        range: view !== "calendar" ? (overrides.range ?? selectedTimeRange) : undefined,
        // Only include from/to for calendar view
        from: view === "calendar" ? (overrides.from ?? customDateRange?.from.toISOString().split("T")[0]) : undefined,
        to: view === "calendar" ? (overrides.to ?? customDateRange?.to.toISOString().split("T")[0]) : undefined,
        // Only include status for calendar view
        status: view === "calendar" ? (overrides.status ?? selectedStatuses) : undefined,
        league: overrides.league ?? selectedLeagues,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedMatchId, activeView, selectedTimeRange, customDateRange, selectedStatuses, selectedLeagues, searchValue]
  );

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (match: MatchSummary) => {
      router.replace(buildUrl({ id: match.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL, preserve filters
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle view change - reset page to 1
  const handleViewChange = useCallback(
    (view: MatchesView) => {
      setCurrentPage(1);
      router.replace(buildUrl({ view, id: null }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle time range change - reset page to 1
  const handleTimeRangeChange = useCallback(
    (range: TimeRange) => {
      setCurrentPage(1);
      router.replace(buildUrl({ range }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle custom date range change (calendar view) - reset page to 1
  const handleCustomDateRangeChange = useCallback(
    (range: DateRangeValue) => {
      setCurrentPage(1);
      router.replace(
        buildUrl({
          from: range.from.toISOString().split("T")[0],
          to: range.to.toISOString().split("T")[0],
        }),
        { scroll: false }
      );
    },
    [router, buildUrl]
  );

  // Handle status filter change (calendar view)
  const handleStatusChange = useCallback(
    (status: MatchStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleLeagueChange = useCallback(
    (league: string, checked: boolean) => {
      const newLeagues = toggleArrayValue(selectedLeagues, league, checked);
      setCurrentPage(1); // Reset pagination on filter change
      router.replace(buildUrl({ league: newLeagues }), { scroll: false });
    },
    [selectedLeagues, router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      setCurrentPage(1); // Reset pagination on search change
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle "Customize Columns" link click
  const handleCustomizeColumnsClick = useCallback(() => {
    setCustomizeColumnsOpen(true);
  }, []);

  // Handle "Done" button - collapses entire Left Rail (UniFi behavior)
  const handleCustomizeColumnsDone = useCallback(() => {
    setLeftRailCollapsed(true);
    setCustomizeColumnsOpen(false);
  }, []);

  // Handle Left Rail toggle - close CustomizeColumns when collapsing
  const handleLeftRailToggle = useCallback(() => {
    const newCollapsed = !leftRailCollapsed;
    setLeftRailCollapsed(newCollapsed);
    if (newCollapsed) {
      setCustomizeColumnsOpen(false);
    }
  }, [leftRailCollapsed]);

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* Left Rail: FilterPanel */}
      <MatchesFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        activeView={activeView}
        onViewChange={handleViewChange}
        selectedTimeRange={selectedTimeRange}
        onTimeRangeChange={handleTimeRangeChange}
        customDateRange={customDateRange}
        onCustomDateRangeChange={handleCustomDateRangeChange}
        matches={matches}
        selectedStatuses={selectedStatuses}
        onStatusChange={handleStatusChange}
        selectedLeagues={selectedLeagues}
        searchValue={searchValue}
        onLeagueChange={handleLeagueChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
      />

      {/* Customize Columns Panel (separate column, hidden when Left Rail collapsed) */}
      <CustomizeColumnsPanel
        open={customizeColumnsOpen && !leftRailCollapsed}
        columns={MATCHES_COLUMN_OPTIONS}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={setColumnVisible}
        onRestore={resetToDefault}
        onDone={handleCustomizeColumnsDone}
        onCollapse={handleLeftRailToggle}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table */}
        <MatchesTable
          data={matches}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedMatchId={selectedMatchId}
          onRowClick={handleRowClick}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisibility}
        />

        {/* Pagination */}
        <Pagination
          currentPage={currentPage}
          totalItems={isDegraded ? matches.length : pagination.total}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
        />

        {/* Degraded indicator */}
        {isDegraded && (
          <div className="absolute bottom-12 left-1/2 -translate-x-1/2 text-xs text-muted-foreground/70 bg-surface px-2 py-1 rounded border border-border">
            Showing mock data
          </div>
        )}
      </div>

      {/* Detail Drawer (overlay on desktop, sheet on mobile) */}
      <MatchDetailDrawer
        match={selectedMatch ?? null}
        open={drawerOpen}
        onClose={handleCloseDrawer}
      />
    </div>
  );
}

/**
 * Loading fallback for Suspense
 */
function MatchesLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Matches Page
 *
 * Master-detail pattern with:
 * - FilterPanel (collapsible, left) with Upcoming/Finished tabs
 * - DataTable (center)
 * - DetailDrawer (overlay, right, no reflow)
 *
 * URL sync (full state):
 * - Canonical: /matches?view=upcoming&range=7d&id=123&league=Premier%20League&q=arsenal
 * - Uses router.replace with scroll:false
 */
export default function MatchesPage() {
  return (
    <Suspense fallback={<MatchesLoading />}>
      <MatchesPageContent />
    </Suspense>
  );
}
