"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMatchesApi, useMatchApi, useMatch, useColumnVisibility, usePageSize, useTeamLogos, useCompactPredictions } from "@/lib/hooks";
import { MatchSummary, MatchFilters, MatchStatus, MATCH_STATUSES } from "@/lib/types";
import { computeGap20, type DivergenceCategory } from "@/lib/predictions";
import { getMatchesMockSync } from "@/lib/mocks";
import {
  MatchesTable,
  MatchesFilterPanel,
  MatchDetailDrawer,
  MATCHES_COLUMN_OPTIONS,
  MATCHES_DEFAULT_VISIBILITY,
  type LocalDate,
} from "@/components/matches";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";
import { useRegion } from "@/components/providers/RegionProvider";
import { isValidLocalDate } from "@/lib/region";

const BASE_PATH = "/matches";

/** Valid divergence filter values */
const VALID_DIVERGENCES: DivergenceCategory[] = ["AGREE", "DISAGREE", "STRONG_FAV_DISAGREE"];

/**
 * Matches Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function MatchesPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { getTodayLocalDate, localDateToUtcStartIso, localDateToUtcEndIso } = useRegion();

  // Parse URL state
  const selectedMatchId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );

  // Parse selected statuses from URL
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

  // Parse selected divergences from URL
  const selectedDivergences = useMemo(
    () => parseArrayParam<DivergenceCategory>(searchParams, "div", VALID_DIVERGENCES),
    [searchParams]
  );

  // Parse selected date from URL as LocalDate string
  const selectedDate = useMemo((): LocalDate => {
    const dateParam = searchParams.get("date");
    if (dateParam && isValidLocalDate(dateParam)) {
      return dateParam;
    }
    // Default: today in user's timezone
    return getTodayLocalDate();
  }, [searchParams, getTodayLocalDate]);

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

  // Compact predictions with localStorage persistence
  const { compactPredictions, setCompactPredictions } = useCompactPredictions();

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("matches");

  // Team logos for shields
  const { getLogoUrl } = useTeamLogos();

  // Column visibility with localStorage persistence
  const {
    columnVisibility,
    setColumnVisibility,
    setColumnVisible,
    resetToDefault,
  } = useColumnVisibility("matches", MATCHES_DEFAULT_VISIBILITY);

  // Construct filters for query (used for both API and mock fallback)
  const filters: MatchFilters = useMemo(() => {
    return {
      status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
      leagues: selectedLeagues.length > 0 ? selectedLeagues : undefined,
      search: searchValue || undefined,
    };
  }, [selectedLeagues, selectedStatuses, searchValue]);

  // Calculate UTC time range for selected date
  const { fromTime, toTime } = useMemo(() => {
    return {
      fromTime: localDateToUtcStartIso(selectedDate),
      toTime: localDateToUtcEndIso(selectedDate),
    };
  }, [selectedDate, localDateToUtcStartIso, localDateToUtcEndIso]);

  // Fetch data from API
  const {
    matches: apiMatches,
    pagination,
    isDegraded,
    isLoading,
    error,
    refetch,
  } = useMatchesApi({
    status: filters.status,
    fromTime,
    toTime,
    page: currentPage,
    limit: pageSize,
  });

  // Use API data if available, fallback to mocks
  const mockMatches = useMemo(() => getMatchesMockSync(filters), [filters]);
  const rawMatches = apiMatches ?? mockMatches;

  // Apply client-side filters (league name, search, divergence)
  // Backend filters by date/status, client filters the rest
  const matches = useMemo(() => {
    let filtered = rawMatches;

    // Filter by league name (client-side, backend uses league_id)
    if (filters.leagues && filters.leagues.length > 0) {
      filtered = filtered.filter((m) => filters.leagues!.includes(m.leagueName));
    }

    // Filter by search text (client-side)
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(
        (m) =>
          m.home.toLowerCase().includes(searchLower) ||
          m.away.toLowerCase().includes(searchLower) ||
          m.leagueName.toLowerCase().includes(searchLower)
      );
    }

    // Filter by divergence category (client-side, uses computeGap20)
    if (selectedDivergences.length > 0) {
      filtered = filtered.filter((m) => {
        if (!m.modelA || !m.market) {
          // No prediction data: only include if AGREE is selected (treat as neutral)
          return selectedDivergences.includes("AGREE");
        }
        const gap20 = computeGap20(m.modelA, m.market);
        if (!gap20) return selectedDivergences.includes("AGREE");
        return selectedDivergences.includes(gap20.category);
      });
    }

    return filtered;
  }, [rawMatches, filters.leagues, filters.search, selectedDivergences]);

  // Find selected match from current list first (no extra fetch needed for basic info)
  // Falls back to backend match lookup, then mock data if needed
  const { match: apiSelectedMatch, isLoading: isMatchApiLoading } = useMatchApi(selectedMatchId);
  const { data: mockSelectedMatch } = useMatch(selectedMatchId);

  const selectedMatchFromList = useMemo(() => {
    if (!selectedMatchId) return null;
    return matches.find((m) => m.id === selectedMatchId) ?? null;
  }, [selectedMatchId, matches]);

  const selectedMatch = useMemo(() => {
    if (!selectedMatchId) return null;
    // Fallback to backend lookup, then mock data
    return selectedMatchFromList ?? apiSelectedMatch ?? mockSelectedMatch ?? null;
  }, [selectedMatchId, selectedMatchFromList, apiSelectedMatch, mockSelectedMatch]);

  // Drawer-specific loading: only when deep-link/pagination fallback is fetching the match
  const isSelectedMatchLoading = selectedMatchId !== null && !selectedMatchFromList && isMatchApiLoading;

  // Drawer is open when there's a selected match
  const drawerOpen = selectedMatchId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
      league?: string[];
      status?: MatchStatus[];
      div?: DivergenceCategory[];
      q?: string;
      date?: LocalDate;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedMatchId : overrides.id,
        date: overrides.date ?? selectedDate,
        status: overrides.status ?? selectedStatuses,
        league: overrides.league ?? selectedLeagues,
        div: overrides.div ?? selectedDivergences,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedMatchId, selectedDate, selectedStatuses, selectedLeagues, selectedDivergences, searchValue]
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

  // Handle date change - reset page to 1
  const handleDateChange = useCallback(
    (date: LocalDate) => {
      setCurrentPage(1);
      router.replace(buildUrl({ date }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle status filter change
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

  const handleDivergenceChange = useCallback(
    (category: DivergenceCategory, checked: boolean) => {
      const newDivergences = toggleArrayValue(selectedDivergences, category, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ div: newDivergences }), { scroll: false });
    },
    [selectedDivergences, router, buildUrl]
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

  // Handle "Done" button - closes CustomizeColumnsPanel only
  const handleCustomizeColumnsDone = useCallback(() => {
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
        selectedDate={selectedDate}
        onDateChange={handleDateChange}
        matches={matches}
        selectedStatuses={selectedStatuses}
        onStatusChange={handleStatusChange}
        selectedLeagues={selectedLeagues}
        selectedDivergences={selectedDivergences}
        searchValue={searchValue}
        onLeagueChange={handleLeagueChange}
        onDivergenceChange={handleDivergenceChange}
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
        onCollapse={handleCustomizeColumnsDone}
        compactPredictions={compactPredictions}
        onCompactPredictionsChange={setCompactPredictions}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background min-w-0">
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
          getLogoUrl={getLogoUrl}
          compactPredictions={compactPredictions}
        />

        {/* Pagination */}
        <Pagination
          currentPage={currentPage}
          totalItems={isDegraded ? matches.length : pagination.total}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
        />

      </div>

      {/* Detail Drawer (overlay on desktop, sheet on mobile) */}
      <MatchDetailDrawer
        match={selectedMatch ?? null}
        open={drawerOpen}
        onClose={handleCloseDrawer}
        isLoading={isSelectedMatchLoading}
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
 * - FilterPanel (collapsible, left) with date picker and filters
 * - DataTable (center)
 * - DetailDrawer (overlay, right, no reflow)
 *
 * URL sync (full state):
 * - Canonical: /matches?date=2026-01-23&id=123&league=Premier%20League&q=arsenal
 * - Uses router.replace with scroll:false
 */
export default function MatchesPage() {
  return (
    <Suspense fallback={<MatchesLoading />}>
      <MatchesPageContent />
    </Suspense>
  );
}
