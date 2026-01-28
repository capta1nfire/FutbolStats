"use client";

import { Suspense, useState, useCallback, useMemo, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  useDataQualityChecksApi,
  useDataQualityCheckApi,
  useOpsOverview,
  useColumnVisibility,
  usePageSize,
  useSotaView,
  useLeagueVisibility,
  useFeatureCoverage,
} from "@/lib/hooks";
import {
  DataQualityCheck,
  DataQualityFilters,
} from "@/lib/types";
import {
  DataQualityTable,
  DataQualityDetailDrawer,
} from "@/components/data-quality";
import { SotaEnrichmentSection } from "@/components/overview";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  SotaFilterPanel,
  SotaStatusFilter,
  SotaSourceFilter,
  SotaViewTabs,
  FeatureCoverageMatrix,
  CoverageRangeFilter,
  FeatureCoverageLeague,
  LeagueFilterPanel,
  FeatureCoverageDetailDrawer,
  SOTA_COLUMN_OPTIONS,
  SOTA_DEFAULT_VISIBILITY,
} from "@/components/sota";
import {
  parseCellParam,
  buildCellParam,
} from "@/lib/types/feature-coverage-detail";
import {
  parseStringId,
  buildSearchParams,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";
import { ClipboardCheck } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

const BASE_PATH = "/sota";

/**
 * SOTA Check IDs (allowlist)
 *
 * These are the 5 SOTA enrichment checks that we filter for.
 */
const SOTA_CHECK_IDS = new Set([
  "dq_understat_coverage_ft_14d",
  "dq_weather_coverage_ns_48h",
  "dq_venue_geo_coverage",
  "dq_team_profile_coverage",
  "dq_sofascore_xi_coverage_ns_48h",
]);

/**
 * SOTA Hub Page Content
 *
 * Two sections:
 * 1. SOTA Enrichment cards (from ops.json) - same as Overview
 * 2. SOTA Data Quality table (5 specific checks)
 */
function SotaPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const rawIdParam = searchParams.get("id");
  const rawCellParam = searchParams.get("cell");

  // Parse URL state - ID is string for backend compatibility
  const selectedCheckId = useMemo(
    () => parseStringId(rawIdParam),
    [rawIdParam]
  );

  // Parse URL state - cell selection for feature coverage detail drawer
  const cellSelection = useMemo(
    () => parseCellParam(rawCellParam),
    [rawCellParam]
  );
  const cellDrawerOpen = cellSelection !== null;

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

  // View state with localStorage persistence
  const { activeView, setActiveView } = useSotaView();

  // Filter state (local, not URL-synced for simplicity)
  const [selectedStatuses, setSelectedStatuses] = useState<SotaStatusFilter[]>([]);
  const [selectedSources, setSelectedSources] = useState<SotaSourceFilter[]>([]);
  const [searchValue, setSearchValue] = useState("");

  // Tier filter state for Features view (all enabled by default)
  const [enabledTiers, setEnabledTiers] = useState<Set<string>>(
    new Set(["tier1", "tier1b", "tier1c", "tier1d"])
  );

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("sota");

  // Column visibility with localStorage persistence
  const { columnVisibility, setColumnVisibility, setColumnVisible, resetToDefault } = useColumnVisibility(
    "sota",
    SOTA_DEFAULT_VISIBILITY
  );

  // League visibility for Feature Coverage Matrix with localStorage persistence
  const {
    leagueVisibility,
    isLeagueVisible,
    setLeagueVisible,
    setAllLeaguesVisible,
    resetToDefault: resetLeaguesToDefault,
  } = useLeagueVisibility();

  // Available leagues from Feature Coverage data
  const [availableLeagues, setAvailableLeagues] = useState<FeatureCoverageLeague[]>([]);

  // Coverage range filter for Features view
  const [coverageRangeFilter, setCoverageRangeFilter] = useState<CoverageRangeFilter>("all");

  // Features pagination state (separate from enrichment view)
  const [featuresCurrentPage, setFeaturesCurrentPage] = useState(1);
  const { pageSize: featuresPageSize, setPageSize: setFeaturesPageSize } = usePageSize("sota-features");
  const [totalFeatures, setTotalFeatures] = useState(0);

  // Fetch SOTA enrichment data from ops.json (same source as Overview)
  const {
    sotaEnrichment,
    isSotaEnrichmentDegraded,
    isLoading: isOpsLoading,
  } = useOpsOverview();

  // Construct filters - always category=coverage
  const filters: DataQualityFilters = useMemo(() => ({
    category: ["coverage"],
  }), []);

  // Fetch data quality checks from API with mock fallback
  const {
    data: allChecks = [],
    isLoading: isDqLoading,
    error,
    isApiDegraded: isDqDegraded,
    refetch,
  } = useDataQualityChecksApi(filters, 1, 100);

  // Filter to only SOTA checks (allowlist)
  const checks = useMemo(() => {
    return allChecks.filter((check) => SOTA_CHECK_IDS.has(check.id));
  }, [allChecks]);

  // Fetch check detail from API with mock fallback
  const {
    data: selectedCheck,
    isLoading: isLoadingDetail,
  } = useDataQualityCheckApi(selectedCheckId);

  // Drawer is open when there's a selected check
  const drawerOpen = selectedCheckId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: { id?: string | null; cell?: string | null }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedCheckId : overrides.id,
        cell: overrides.cell === undefined ? rawCellParam : overrides.cell,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedCheckId, rawCellParam]
  );

  // P1: Make URL state truly shareable by syncing the persisted view (localStorage) with query params.
  // - ?id=... should force Enrichment view
  // - ?cell=... should force Features view
  // - if both exist, prefer ?id=... and clear ?cell=... to avoid ambiguous UI state
  useEffect(() => {
    const hasId = selectedCheckId !== null;
    const hasCell = cellSelection !== null;

    if (!hasId && !hasCell) return;

    if (hasId && hasCell) {
      if (activeView !== "enrichment") setActiveView("enrichment");
      if (rawCellParam) {
        router.replace(buildUrl({ cell: null }), { scroll: false });
      }
      setCustomizeColumnsOpen(false);
      return;
    }

    if (hasId && activeView !== "enrichment") {
      setActiveView("enrichment");
      if (rawCellParam) {
        router.replace(buildUrl({ cell: null }), { scroll: false });
      }
      setCustomizeColumnsOpen(false);
      return;
    }

    if (hasCell && activeView !== "features") {
      setActiveView("features");
      if (rawIdParam) {
        router.replace(buildUrl({ id: null }), { scroll: false });
      }
      setCustomizeColumnsOpen(false);
    }
  }, [
    activeView,
    buildUrl,
    cellSelection,
    rawCellParam,
    rawIdParam,
    router,
    selectedCheckId,
    setActiveView,
    setCustomizeColumnsOpen,
  ]);

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (check: DataQualityCheck) => {
      router.replace(buildUrl({ id: check.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle cell click in Feature Coverage Matrix - open detail drawer
  const handleCellClick = useCallback(
    (featureKey: string, leagueId: number) => {
      router.replace(
        buildUrl({ cell: buildCellParam(featureKey, leagueId) }),
        { scroll: false }
      );
    },
    [router, buildUrl]
  );

  // Handle cell detail drawer close
  const handleCellDrawerClose = useCallback(() => {
    router.replace(buildUrl({ cell: null }), { scroll: false });
  }, [router, buildUrl]);

  // Feature coverage data for detail drawer (shared TanStack Query cache - no double fetch)
  const { data: featureCoverageData } = useFeatureCoverage();

  // Handlers for Customize Columns
  const handleCustomizeColumnsClick = useCallback(() => {
    setCustomizeColumnsOpen(true);
  }, []);

  // Done only closes Customize Columns panel (keeps FilterPanel visible)
  const handleCustomizeColumnsDone = useCallback(() => {
    setCustomizeColumnsOpen(false);
  }, []);

  const handleLeftRailToggle = useCallback(() => {
    setLeftRailCollapsed((prev) => !prev);
    if (!leftRailCollapsed) {
      setCustomizeColumnsOpen(false);
    }
  }, [leftRailCollapsed]);

  // Filter handlers
  const handleStatusChange = useCallback((status: SotaStatusFilter, checked: boolean) => {
    setSelectedStatuses((prev) =>
      checked ? [...prev, status] : prev.filter((s) => s !== status)
    );
    setCurrentPage(1);
  }, []);

  const handleSourceChange = useCallback((source: SotaSourceFilter, checked: boolean) => {
    setSelectedSources((prev) =>
      checked ? [...prev, source] : prev.filter((s) => s !== source)
    );
    setCurrentPage(1);
  }, []);

  const handleSearchChange = useCallback((value: string) => {
    setSearchValue(value);
    setCurrentPage(1);
  }, []);

  // Tier filter handler (for Features view)
  const handleTierChange = useCallback((tierId: string, checked: boolean) => {
    setEnabledTiers((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(tierId);
      } else {
        next.delete(tierId);
      }
      return next;
    });
    setFeaturesCurrentPage(1);
  }, []);

  // League filter handlers
  const handleLeagueVisibilityChange = useCallback(
    (leagueId: number, visible: boolean) => {
      setLeagueVisible(leagueId, visible);
    },
    [setLeagueVisible]
  );

  const handleAllLeaguesChange = useCallback(
    (visible: boolean, leagueIds: number[]) => {
      setAllLeaguesVisible(visible, leagueIds);
    },
    [setAllLeaguesVisible]
  );

  const handleLeaguesLoaded = useCallback((leagues: FeatureCoverageLeague[]) => {
    setAvailableLeagues(leagues);
  }, []);

  // Features total count handler (for pagination)
  const handleTotalFeaturesChange = useCallback((total: number) => {
    setTotalFeatures(total);
  }, []);

  // Pagination - client-side since we're filtering
  const paginatedChecks = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return checks.slice(start, start + pageSize);
  }, [checks, currentPage, pageSize]);

  // Handle view change — close drawers and clean URL params from the abandoned view
  const handleViewChange = useCallback(
    (nextView: typeof activeView) => {
      setActiveView(nextView);
      if (nextView === "features") {
        // Leaving enrichment → clear ?id=
        router.replace(buildUrl({ id: null }), { scroll: false });
      } else {
        // Leaving features → clear ?cell=
        router.replace(buildUrl({ cell: null }), { scroll: false });
      }
      setCustomizeColumnsOpen(false);
    },
    [setActiveView, router, buildUrl]
  );

  // Header content with view tabs
  const headerContent = (
    <SotaViewTabs activeView={activeView} onViewChange={handleViewChange} />
  );

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* FilterPanel */}
      <SotaFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        headerContent={headerContent}
        selectedStatuses={selectedStatuses}
        selectedSources={selectedSources}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onSourceChange={handleSourceChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
        activeView={activeView}
        enabledTiers={enabledTiers}
        onTierChange={handleTierChange}
        coverageRangeFilter={coverageRangeFilter}
        onCoverageRangeChange={setCoverageRangeFilter}
      />

      {/* Customize Columns Panel (Enrichment view) */}
      {activeView === "enrichment" && (
        <CustomizeColumnsPanel
          open={customizeColumnsOpen && !leftRailCollapsed}
          columns={SOTA_COLUMN_OPTIONS}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisible}
          onRestore={resetToDefault}
          onDone={handleCustomizeColumnsDone}
          onCollapse={handleLeftRailToggle}
        />
      )}

      {/* League Filter Panel (Features view) */}
      {activeView === "features" && (
        <LeagueFilterPanel
          open={customizeColumnsOpen && !leftRailCollapsed}
          leagues={availableLeagues}
          leagueVisibility={leagueVisibility}
          onLeagueVisibilityChange={handleLeagueVisibilityChange}
          onAllLeaguesChange={handleAllLeaguesChange}
          onRestore={resetLeaguesToDefault}
          onDone={handleCustomizeColumnsDone}
          onCollapse={handleLeftRailToggle}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Content area - different layout per view */}
        {activeView === "enrichment" ? (
          /* Enrichment View - Scrollable content */
          <ScrollArea className="flex-1 min-h-0">
            <div className="p-6 space-y-6">
              {/* SOTA Enrichment Section (cards) */}
              <SotaEnrichmentSection
                data={sotaEnrichment}
                isMockFallback={isSotaEnrichmentDegraded}
              />

              {/* Data Quality Section */}
              <div className="space-y-3">
                {/* Section header */}
                <div className="flex items-center gap-2">
                  <ClipboardCheck className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-semibold text-foreground">Data Quality Checks</h2>
                  <span className="text-xs text-muted-foreground">
                    ({checks.length} checks)
                  </span>
                </div>

                {/* Table container */}
                <div className="border border-border rounded-lg overflow-hidden">
                  <DataQualityTable
                    data={paginatedChecks}
                    isLoading={isDqLoading}
                    error={error}
                    onRetry={() => refetch()}
                    selectedCheckId={selectedCheckId}
                    onRowClick={handleRowClick}
                    columnVisibility={columnVisibility}
                    onColumnVisibilityChange={setColumnVisibility}
                  />
                </div>

                {/* Pagination */}
                {checks.length > 0 && (
                  <Pagination
                    currentPage={currentPage}
                    totalItems={checks.length}
                    pageSize={pageSize}
                    onPageChange={setCurrentPage}
                    onPageSizeChange={setPageSize}
                  />
                )}
              </div>
            </div>
          </ScrollArea>
        ) : (
          /* Features View - Full height with fixed header/footer */
          <>
            <FeatureCoverageMatrix
              isLeagueVisible={isLeagueVisible}
              onLeaguesLoaded={handleLeaguesLoaded}
              currentPage={featuresCurrentPage}
              pageSize={featuresPageSize}
              onTotalFeaturesChange={handleTotalFeaturesChange}
              enabledTiers={enabledTiers}
              coverageRangeFilter={coverageRangeFilter}
              onCellClick={handleCellClick}
            />
            <Pagination
              currentPage={featuresCurrentPage}
              totalItems={totalFeatures}
              pageSize={featuresPageSize}
              onPageChange={setFeaturesCurrentPage}
              onPageSizeChange={setFeaturesPageSize}
              pageSizeOptions={[25, 50, 100]}
            />
          </>
        )}
      </div>

      {/* Detail Drawer — only in Enrichment view */}
      {activeView === "enrichment" && (
        <DataQualityDetailDrawer
          check={selectedCheck ?? null}
          open={drawerOpen}
          onClose={handleCloseDrawer}
          isLoading={isLoadingDetail}
        />
      )}

      {/* Feature Coverage Detail Drawer (Features view only) */}
      {activeView === "features" && (
        <FeatureCoverageDetailDrawer
          selection={cellSelection}
          open={cellDrawerOpen}
          onClose={handleCellDrawerClose}
          coverageData={featureCoverageData?.data ?? null}
        />
      )}
    </div>
  );
}

/**
 * Loading fallback for Suspense
 */
function SotaLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * SOTA Hub Page
 *
 * Dedicated hub for SOTA enrichment:
 * 1. SOTA Enrichment cards (Understat, Weather, Venue Geo, Team Profiles, Sofascore XI)
 * 2. SOTA Data Quality checks table
 *
 * URL patterns:
 * - /sota - hub view
 * - /sota?id=dq_understat_coverage_ft_14d - with enrichment drawer open
 * - /sota?cell=home_goals_scored_avg:39 - with feature coverage detail drawer open
 */
export default function SotaPage() {
  return (
    <Suspense fallback={<SotaLoading />}>
      <SotaPageContent />
    </Suspense>
  );
}
