"use client";

import { Suspense, useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useMatches, useMatch } from "@/lib/hooks";
import { MatchSummary, MatchStatus, MatchFilters } from "@/lib/types";
import {
  MatchesTable,
  MatchesFilterPanel,
  MatchDetailDrawer,
} from "@/components/matches";
import { Loader2 } from "lucide-react";

/**
 * Parse and validate match ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
function parseMatchId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Matches Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function MatchesPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL state: selected match ID (sanitized)
  const selectedIdParam = searchParams.get("id");
  const selectedMatchId = parseMatchId(selectedIdParam);

  // Normalize URL if id param is invalid
  useEffect(() => {
    if (selectedIdParam && selectedMatchId === null) {
      // Invalid id in URL → normalize to /matches
      router.replace("/matches", { scroll: false });
    }
  }, [selectedIdParam, selectedMatchId, router]);

  // UI state
  const [filterCollapsed, setFilterCollapsed] = useState(false);
  const [selectedStatuses, setSelectedStatuses] = useState<MatchStatus[]>([]);
  const [selectedLeagues, setSelectedLeagues] = useState<string[]>([]);
  const [searchValue, setSearchValue] = useState("");

  // Construct filters
  const filters: MatchFilters = {
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    leagues: selectedLeagues.length > 0 ? selectedLeagues : undefined,
    search: searchValue || undefined,
  };

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

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (match: MatchSummary) => {
      router.replace(`/matches?id=${match.id}`, { scroll: false });
    },
    [router]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace("/matches", { scroll: false });
  }, [router]);

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: MatchStatus, checked: boolean) => {
      setSelectedStatuses((prev) =>
        checked ? [...prev, status] : prev.filter((s) => s !== status)
      );
    },
    []
  );

  const handleLeagueChange = useCallback(
    (league: string, checked: boolean) => {
      setSelectedLeagues((prev) =>
        checked ? [...prev, league] : prev.filter((l) => l !== league)
      );
    },
    []
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
        onSearchChange={setSearchValue}
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
 * URL sync:
 * - Canonical: /matches?id=123
 * - Uses router.replace with scroll:false
 */
export default function MatchesPage() {
  return (
    <Suspense fallback={<MatchesLoading />}>
      <MatchesPageContent />
    </Suspense>
  );
}
