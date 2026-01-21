"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMatches, useMatch } from "@/lib/hooks";
import { MatchSummary, MatchStatus, MatchFilters, MATCH_STATUSES } from "@/lib/types";
import { getLeaguesMock } from "@/lib/mocks";
import {
  MatchesTable,
  MatchesFilterPanel,
  MatchDetailDrawer,
} from "@/components/matches";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader2 } from "lucide-react";

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
  const [filterCollapsed, setFilterCollapsed] = useState(false);

  // Construct filters for query
  const filters: MatchFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    leagues: selectedLeagues.length > 0 ? selectedLeagues : undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedLeagues, searchValue]);

  // Fetch data
  const {
    data: matches = [],
    isLoading,
    error,
    refetch,
  } = useMatches(filters);

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
        id: overrides.id ?? selectedMatchId,
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

  return (
    <div className="h-full flex overflow-hidden">
      {/* FilterPanel */}
      <MatchesFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
        selectedStatuses={selectedStatuses}
        selectedLeagues={selectedLeagues}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onLeagueChange={handleLeagueChange}
        onSearchChange={handleSearchChange}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Matches</h1>
          <span className="text-sm text-muted-foreground">
            {matches.length} matches
          </span>
        </div>

        {/* Table */}
        <MatchesTable
          data={matches}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedMatchId={selectedMatchId}
          onRowClick={handleRowClick}
        />
      </div>

      {/* Detail Drawer (inline) */}
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
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading matches...</p>
      </div>
    </div>
  );
}

/**
 * Matches Page
 *
 * Master-detail pattern with:
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (inline, right, pushes content)
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
