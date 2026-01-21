"use client";

import { Suspense, useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { usePredictions, usePrediction, usePredictionCoverage } from "@/lib/hooks";
import { getPredictionLeaguesMock } from "@/lib/mocks";
import {
  PredictionRow,
  PredictionStatus,
  ModelType,
  PredictionTimeRange,
  PredictionFilters,
} from "@/lib/types";
import {
  PredictionsTable,
  PredictionsFilterPanel,
  PredictionDetailDrawer,
  PredictionsCoverageCard,
} from "@/components/predictions";
import { Loader2 } from "lucide-react";

/**
 * Parse and validate prediction ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
function parsePredictionId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Predictions Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function PredictionsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL state: selected prediction ID (sanitized)
  const selectedIdParam = searchParams.get("id");
  const selectedPredictionId = parsePredictionId(selectedIdParam);

  // Normalize URL if id param is invalid
  useEffect(() => {
    if (selectedIdParam && selectedPredictionId === null) {
      // Invalid id in URL → normalize to /predictions
      router.replace("/predictions", { scroll: false });
    }
  }, [selectedIdParam, selectedPredictionId, router]);

  // UI state
  const [filterCollapsed, setFilterCollapsed] = useState(false);
  const [selectedStatuses, setSelectedStatuses] = useState<PredictionStatus[]>([]);
  const [selectedModels, setSelectedModels] = useState<ModelType[]>([]);
  const [selectedLeagues, setSelectedLeagues] = useState<string[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<PredictionTimeRange | null>(null);
  const [searchValue, setSearchValue] = useState("");

  // Available leagues for filter
  const availableLeagues = getPredictionLeaguesMock();

  // Construct filters
  const filters: PredictionFilters = {
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    model: selectedModels.length > 0 ? selectedModels : undefined,
    league: selectedLeagues.length > 0 ? selectedLeagues : undefined,
    timeRange: selectedTimeRange || undefined,
    search: searchValue || undefined,
  };

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

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (prediction: PredictionRow) => {
      router.replace(`/predictions?id=${prediction.id}`, { scroll: false });
    },
    [router]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace("/predictions", { scroll: false });
  }, [router]);

  // Handle "View missing" from coverage card
  const handleViewMissing = useCallback(() => {
    setSelectedStatuses(["missing"]);
  }, []);

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: PredictionStatus, checked: boolean) => {
      setSelectedStatuses((prev) =>
        checked ? [...prev, status] : prev.filter((s) => s !== status)
      );
    },
    []
  );

  const handleModelChange = useCallback(
    (model: ModelType, checked: boolean) => {
      setSelectedModels((prev) =>
        checked ? [...prev, model] : prev.filter((m) => m !== model)
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

  const handleTimeRangeChange = useCallback(
    (timeRange: PredictionTimeRange | null) => {
      setSelectedTimeRange(timeRange);
    },
    []
  );

  return (
    <div className="h-full flex overflow-hidden">
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
        onSearchChange={setSearchValue}
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
 * Master-detail pattern with:
 * - Coverage card (top)
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (inline on desktop, right, pushes content)
 *
 * URL sync:
 * - Canonical: /predictions?id=123
 * - Uses router.replace with scroll:false
 */
export default function PredictionsPage() {
  return (
    <Suspense fallback={<PredictionsLoading />}>
      <PredictionsPageContent />
    </Suspense>
  );
}
