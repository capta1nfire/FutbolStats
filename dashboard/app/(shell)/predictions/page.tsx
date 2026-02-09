"use client";

import { Suspense, useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  usePredictionsApi,
  usePrediction,
  usePredictionCoverageFromData,
  useColumnVisibility,
  usePageSize,
  useTeamLogos,
} from "@/lib/hooks";
import { getPredictionLeaguesMock } from "@/lib/mocks";
import {
  PredictionRow,
  PredictionStatus,
  ModelType,
  PredictionTimeRange,
  PREDICTION_STATUSES,
  MODEL_TYPES,
  PREDICTION_TIME_RANGES,
} from "@/lib/types";
import {
  PredictionsTable,
  PredictionsFilterPanel,
  PredictionDetailDrawer,
  PredictionsCoverageCard,
  BenchmarkMatrix,
  BenchmarkViewTabs,
  PREDICTIONS_COLUMN_OPTIONS,
  PREDICTIONS_DEFAULT_VISIBILITY,
} from "@/components/predictions";
import type { PredictionsView } from "@/components/predictions";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseNumericId,
  parseArrayParam,
  parseSingleParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";
import { Database } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const BASE_PATH = "/predictions";

/**
 * Predictions Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function PredictionsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const selectedPredictionId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );
  const selectedStatuses = useMemo(
    () => parseArrayParam<PredictionStatus>(searchParams, "status", PREDICTION_STATUSES),
    [searchParams]
  );
  const selectedModels = useMemo(
    () => parseArrayParam<ModelType>(searchParams, "model", MODEL_TYPES),
    [searchParams]
  );
  const selectedTimeRange = useMemo(
    () => parseSingleParam<PredictionTimeRange>(searchParams.get("range"), PREDICTION_TIME_RANGES),
    [searchParams]
  );
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

  // Raw league param from URL (validated later against available leagues)
  const leagueParamRaw = useMemo(
    () => searchParams.getAll("league"),
    [searchParams]
  );

  // Normalize URL if id param is invalid
  const selectedIdParam = searchParams.get("id");
  useEffect(() => {
    if (selectedIdParam && selectedPredictionId === null) {
      const params = new URLSearchParams(searchParams.toString());
      params.delete("id");
      const search = params.toString();
      router.replace(`${BASE_PATH}${search ? `?${search}` : ""}`, { scroll: false });
    }
  }, [selectedIdParam, selectedPredictionId, router, searchParams]);

  // View tab state with localStorage persistence
  const [activeView, setActiveViewState] = useState<PredictionsView>("predictions");
  const isViewInitialized = useRef(false);
  useEffect(() => {
    if (isViewInitialized.current) return;
    isViewInitialized.current = true;
    try {
      const stored = localStorage.getItem("predictions:activeView");
      if (stored === "predictions" || stored === "benchmark") {
        setActiveViewState(stored);
      }
    } catch { /* ignore */ }
  }, []);
  useEffect(() => {
    if (!isViewInitialized.current) return;
    try { localStorage.setItem("predictions:activeView", activeView); } catch { /* ignore */ }
  }, [activeView]);
  const setActiveView = useCallback((view: PredictionsView) => {
    setActiveViewState(view);
  }, []);

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("predictions");

  // Team logos for shields
  const { getLogoUrl } = useTeamLogos();

  // Column visibility with localStorage persistence
  const { columnVisibility, setColumnVisibility, setColumnVisible, resetToDefault } = useColumnVisibility(
    "predictions",
    PREDICTIONS_DEFAULT_VISIBILITY
  );

  // Handlers for Customize Columns
  const handleCustomizeColumnsClick = useCallback(() => {
    setCustomizeColumnsOpen(true);
  }, []);

  // Done collapses entire Left Rail (UniFi behavior)
  const handleCustomizeColumnsDone = useCallback(() => {
    setLeftRailCollapsed(true);
    setCustomizeColumnsOpen(false);
  }, []);

  const handleLeftRailToggle = useCallback(() => {
    setLeftRailCollapsed((prev) => !prev);
    if (!leftRailCollapsed) {
      setCustomizeColumnsOpen(false);
    }
  }, [leftRailCollapsed]);

  // Fetch data from API with server-side filtering and real pagination
  const {
    predictions: apiPredictions,
    pagination,
    isLoading,
    error,
    isDegraded,
    refetch,
  } = usePredictionsApi({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    model: selectedModels.length > 0 ? selectedModels : undefined,
    q: searchValue || undefined,
    timeRange: selectedTimeRange || undefined,
    page: currentPage,
    limit: pageSize,
  });

  // Derive available leagues from real data, fallback to mock if empty
  const availableLeagues = useMemo(() => {
    const leagues = [...new Set(apiPredictions.map((p) => p.leagueName))].sort();
    return leagues.length > 0 ? leagues : getPredictionLeaguesMock();
  }, [apiPredictions]);

  // Validate league params against available leagues
  const selectedLeagues = useMemo(() => {
    return leagueParamRaw.filter((l) => availableLeagues.includes(l));
  }, [leagueParamRaw, availableLeagues]);

  // Apply league filter client-side (since backend filters by league_ids, not names)
  const predictions = useMemo(() => {
    if (selectedLeagues.length === 0) return apiPredictions;
    return apiPredictions.filter((p) =>
      selectedLeagues.some((l) => p.leagueName.toLowerCase().includes(l.toLowerCase()))
    );
  }, [apiPredictions, selectedLeagues]);

  const {
    data: selectedPrediction,
    isLoading: isLoadingDetail,
  } = usePrediction(selectedPredictionId);

  // Calculate coverage from current predictions
  const periodLabel = selectedTimeRange
    ? selectedTimeRange === "24h" ? "Next 24 hours"
      : selectedTimeRange === "48h" ? "Next 48 hours"
      : selectedTimeRange === "7d" ? "Next 7 days"
      : "Next 30 days"
    : "Next 3 days";
  const coverage = usePredictionCoverageFromData(predictions, periodLabel);
  const isLoadingCoverage = isLoading;

  // Drawer is open when there's a selected prediction
  const drawerOpen = selectedPredictionId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
      status?: PredictionStatus[];
      model?: ModelType[];
      league?: string[];
      range?: PredictionTimeRange | null;
      q?: string;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedPredictionId : overrides.id,
        status: overrides.status ?? selectedStatuses,
        model: overrides.model ?? selectedModels,
        league: overrides.league ?? selectedLeagues,
        range: overrides.range === undefined ? selectedTimeRange : overrides.range,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedPredictionId, selectedStatuses, selectedModels, selectedLeagues, selectedTimeRange, searchValue]
  );

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (prediction: PredictionRow) => {
      router.replace(buildUrl({ id: prediction.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL, preserve filters
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle "View missing" from coverage card
  const handleViewMissing = useCallback(() => {
    router.replace(buildUrl({ status: ["missing"] }), { scroll: false });
  }, [router, buildUrl]);

  // Handle filter changes - reset to page 1 when filters change
  const handleStatusChange = useCallback(
    (status: PredictionStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleModelChange = useCallback(
    (model: ModelType, checked: boolean) => {
      const newModels = toggleArrayValue(selectedModels, model, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ model: newModels }), { scroll: false });
    },
    [selectedModels, router, buildUrl]
  );

  const handleLeagueChange = useCallback(
    (league: string, checked: boolean) => {
      const newLeagues = toggleArrayValue(selectedLeagues, league, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ league: newLeagues }), { scroll: false });
    },
    [selectedLeagues, router, buildUrl]
  );

  const handleTimeRangeChange = useCallback(
    (timeRange: PredictionTimeRange | null) => {
      setCurrentPage(1);
      router.replace(buildUrl({ range: timeRange }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      setCurrentPage(1);
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle page change
  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page);
  }, []);

  // Benchmark view: full-width, no filter panel
  if (activeView === "benchmark") {
    return (
      <div className="h-full flex flex-col overflow-hidden bg-background">
        {/* Header with view tabs */}
        <div className="h-12 flex items-center justify-between px-6 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Predictions</h1>
          <div className="w-[180px]">
            <BenchmarkViewTabs activeView={activeView} onViewChange={setActiveView} />
          </div>
        </div>
        <BenchmarkMatrix />
      </div>
    );
  }

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* FilterPanel */}
      <PredictionsFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        selectedStatuses={selectedStatuses}
        selectedModels={selectedModels}
        selectedLeagues={selectedLeagues}
        selectedTimeRange={selectedTimeRange}
        searchValue={searchValue}
        availableLeagues={availableLeagues}
        onStatusChange={handleStatusChange}
        onModelChange={handleModelChange}
        onLeagueChange={handleLeagueChange}
        onTimeRangeChange={handleTimeRangeChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
      />

      {/* Customize Columns Panel */}
      <CustomizeColumnsPanel
        open={customizeColumnsOpen && !leftRailCollapsed}
        columns={PREDICTIONS_COLUMN_OPTIONS}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={setColumnVisible}
        onRestore={resetToDefault}
        onDone={handleCustomizeColumnsDone}
        onCollapse={handleLeftRailToggle}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Header with view tabs and mock indicator */}
        <div className="h-12 flex items-center justify-between px-6 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Predictions</h1>
          <div className="flex items-center gap-3">
            {isDegraded && !isLoading && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)]">
                      <Database className="h-3.5 w-3.5 text-[var(--status-warning-text)]" />
                      <span className="text-[10px] text-[var(--status-warning-text)] font-medium">
                        mock
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>Using mock data - backend unavailable</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            <div className="w-[180px]">
              <BenchmarkViewTabs activeView={activeView} onViewChange={setActiveView} />
            </div>
          </div>
        </div>

        {/* Coverage Card */}
        <div className="p-4 border-b border-border">
          <PredictionsCoverageCard
            coverage={coverage}
            onViewMissing={handleViewMissing}
            isLoading={isLoadingCoverage}
          />
        </div>

        {/* Table */}
        <PredictionsTable
          data={predictions}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedPredictionId={selectedPredictionId}
          onRowClick={handleRowClick}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisibility}
          getLogoUrl={getLogoUrl}
        />

        {/* Pagination - using real total from backend */}
        <Pagination
          currentPage={currentPage}
          totalItems={pagination.total}
          pageSize={pageSize}
          onPageChange={handlePageChange}
          onPageSizeChange={setPageSize}
        />
      </div>

      {/* Detail Drawer (inline on desktop, sheet on mobile) */}
      <PredictionDetailDrawer
        prediction={selectedPrediction ?? null}
        open={drawerOpen}
        onClose={handleCloseDrawer}
        isLoading={isLoadingDetail}
      />
    </div>
  );
}

/**
 * Loading fallback for Suspense
 */
function PredictionsLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Predictions Page
 *
 * Master-detail pattern with URL sync (full state):
 * - Canonical: /predictions?id=123&status=missing&model=A&league=Premier%20League&range=24h&q=real
 * - Uses router.replace with scroll:false
 * - Real pagination from backend
 */
export default function PredictionsPage() {
  return (
    <Suspense fallback={<PredictionsLoading />}>
      <PredictionsPageContent />
    </Suspense>
  );
}
