"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMatchesApi, useMatch, useColumnVisibility, usePageSize } from "@/lib/hooks";
import { MatchSummary, MatchStatus, MatchFilters, MATCH_STATUSES } from "@/lib/types";
import { getLeaguesMock, getMatchesMock } from "@/lib/mocks";
import {
  MatchesTable,
  MatchesFilterPanel,
  MatchDetailDrawer,
  MATCHES_COLUMN_OPTIONS,
  MATCHES_DEFAULT_VISIBILITY,
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

/**
 * Matches Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function MatchesPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Available leagues for filter
  const availableLeagues = useMemo(() => getLeaguesMock(), []);

  // Parse URL state
  const selectedMatchId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );
  const selectedStatuses = useMemo(
    () => parseArrayParam<MatchStatus>(searchParams, "status", MATCH_STATUSES),
    [searchParams]
  );
  const selectedLeagues = useMemo(
    () => parseArrayParam<string>(searchParams, "league", availableLeagues),
    [searchParams, availableLeagues]
  );
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

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

  // Construct filters for query
  const filters: MatchFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    leagues: selectedLeagues.length > 0 ? selectedLeagues : undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedLeagues, searchValue]);

  // Fetch data from API with mock fallback
  const {
    matches: apiMatches,
    pagination,
    isDegraded,
    isLoading,
    error,
    refetch,
  } = useMatchesApi({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    page: currentPage,
    limit: pageSize,
  });

  // Use API data if available, fallback to mocks
  const mockMatches = useMemo(() => getMatchesMock(filters), [filters]);
  const matches = apiMatches ?? mockMatches;

  const { data: selectedMatch } = useMatch(selectedMatchId);

  // Drawer is open when there's a selected match
  const drawerOpen = selectedMatchId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
      status?: MatchStatus[];
      league?: string[];
      q?: string;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedMatchId : overrides.id,
        status: overrides.status ?? selectedStatuses,
        league: overrides.league ?? selectedLeagues,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedMatchId, selectedStatuses, selectedLeagues, searchValue]
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

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: MatchStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleLeagueChange = useCallback(
    (league: string, checked: boolean) => {
      const newLeagues = toggleArrayValue(selectedLeagues, league, checked);
      router.replace(buildUrl({ league: newLeagues }), { scroll: false });
    },
    [selectedLeagues, router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
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
        selectedStatuses={selectedStatuses}
        selectedLeagues={selectedLeagues}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
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
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (overlay, right, no reflow)
 *
 * URL sync (full state):
 * - Canonical: /matches?id=123&status=live&status=ft&league=Premier%20League&q=arsenal
 * - Uses router.replace with scroll:false
 */
export default function MatchesPage() {
  return (
    <Suspense fallback={<MatchesLoading />}>
      <MatchesPageContent />
    </Suspense>
  );
}
