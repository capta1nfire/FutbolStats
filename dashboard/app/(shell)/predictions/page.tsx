"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { usePredictions, usePrediction, usePredictionCoverage } from "@/lib/hooks";
import { getPredictionLeaguesMock } from "@/lib/mocks";
import {
  PredictionRow,
  PredictionStatus,
  ModelType,
  PredictionTimeRange,
  PredictionFilters,
  PREDICTION_STATUSES,
  MODEL_TYPES,
  PREDICTION_TIME_RANGES,
} from "@/lib/types";
import {
  PredictionsTable,
  PredictionsFilterPanel,
  PredictionDetailDrawer,
  PredictionsCoverageCard,
} from "@/components/predictions";
import {
  parseNumericId,
  parseArrayParam,
  parseSingleParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader2 } from "lucide-react";

const BASE_PATH = "/predictions";

/**
 * Predictions Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function PredictionsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Available leagues for filter
  const availableLeagues = useMemo(() => getPredictionLeaguesMock(), []);

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
  const selectedLeagues = useMemo(
    () => parseArrayParam<string>(searchParams, "league", availableLeagues),
    [searchParams, availableLeagues]
  );
  const selectedTimeRange = useMemo(
    () => parseSingleParam<PredictionTimeRange>(searchParams.get("range"), PREDICTION_TIME_RANGES),
    [searchParams]
  );
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
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

  // UI state (non-URL)
  const [filterCollapsed, setFilterCollapsed] = useState(false);

  // Construct filters for query
  const filters: PredictionFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    model: selectedModels.length > 0 ? selectedModels : undefined,
    league: selectedLeagues.length > 0 ? selectedLeagues : undefined,
    timeRange: selectedTimeRange || undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedModels, selectedLeagues, selectedTimeRange, searchValue]);

  // Fetch data
  const {
    data: predictions = [],
    isLoading,
    error,
    refetch,
  } = usePredictions(filters);

  const {
    data: selectedPrediction,
    isLoading: isLoadingDetail,
  } = usePrediction(selectedPredictionId);

  const {
    data: coverage,
    isLoading: isLoadingCoverage,
  } = usePredictionCoverage("24h");

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

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: PredictionStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleModelChange = useCallback(
    (model: ModelType, checked: boolean) => {
      const newModels = toggleArrayValue(selectedModels, model, checked);
      router.replace(buildUrl({ model: newModels }), { scroll: false });
    },
    [selectedModels, router, buildUrl]
  );

  const handleLeagueChange = useCallback(
    (league: string, checked: boolean) => {
      const newLeagues = toggleArrayValue(selectedLeagues, league, checked);
      router.replace(buildUrl({ league: newLeagues }), { scroll: false });
    },
    [selectedLeagues, router, buildUrl]
  );

  const handleTimeRangeChange = useCallback(
    (timeRange: PredictionTimeRange | null) => {
      router.replace(buildUrl({ range: timeRange }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* FilterPanel */}
      <PredictionsFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
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
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Predictions</h1>
          <span className="text-sm text-muted-foreground">
            {predictions.length} predictions
          </span>
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
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading predictions...</p>
      </div>
    </div>
  );
}

/**
 * Predictions Page
 *
 * Master-detail pattern with URL sync (full state):
 * - Canonical: /predictions?id=123&status=missing&model=A&league=Premier%20League&range=24h&q=real
 * - Uses router.replace with scroll:false
 */
export default function PredictionsPage() {
  return (
    <Suspense fallback={<PredictionsLoading />}>
      <PredictionsPageContent />
    </Suspense>
  );
}
