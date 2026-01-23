"use client";

import { Suspense, useState, useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  useAnalyticsReportsApi,
  useAnalyticsReport,
  useColumnVisibility,
  usePageSize,
  useOpsHistoryApi,
  usePredictionsPerformanceApi,
} from "@/lib/hooks";
import {
  AnalyticsReportRow,
  AnalyticsReportType,
  ANALYTICS_REPORT_TYPES,
} from "@/lib/types";
import {
  AnalyticsTable,
  AnalyticsFilterPanel,
  AnalyticsDetailDrawer,
  OpsHistorySummary,
  PredictionsPerformanceCard,
  ANALYTICS_COLUMN_OPTIONS,
  ANALYTICS_DEFAULT_VISIBILITY,
} from "@/components/analytics";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseStringId,
  parseArrayParam,
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

const BASE_PATH = "/analytics";

/**
 * Analytics Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function AnalyticsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state - ID is string for backend compatibility (e.g., "model_perf_14d_20")
  const selectedReportId = useMemo(
    () => parseStringId(searchParams.get("id")),
    [searchParams]
  );
  const selectedTypes = useMemo(
    () => parseArrayParam<AnalyticsReportType>(searchParams, "type", ANALYTICS_REPORT_TYPES),
    [searchParams]
  );
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

  // Note: No URL normalization needed for string IDs - parseStringId accepts any non-empty string

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("analytics");

  // Column visibility with localStorage persistence
  const { columnVisibility, setColumnVisibility, setColumnVisible, resetToDefault } = useColumnVisibility(
    "analytics",
    ANALYTICS_DEFAULT_VISIBILITY
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

  // Fetch reports from API with server-side filtering and real pagination
  const {
    reports,
    pagination,
    isLoading,
    error,
    isDegraded: isReportsDegraded,
    refetch,
  } = useAnalyticsReportsApi({
    type: selectedTypes.length > 0 ? selectedTypes[0] : undefined, // API supports single type filter
    q: searchValue || undefined,
    page: currentPage,
    limit: pageSize,
  });

  const {
    data: selectedReport,
    isLoading: isLoadingDetail,
  } = useAnalyticsReport(selectedReportId);

  // Fetch real ops data
  const {
    data: opsHistory,
    isLoading: isLoadingHistory,
    isApiDegraded: isHistoryDegraded,
  } = useOpsHistoryApi(30);

  const {
    data: predictionsPerformance,
    isLoading: isLoadingPerformance,
    isApiDegraded: isPerformanceDegraded,
  } = usePredictionsPerformanceApi(7);

  // Drawer is open when there's a selected report
  const drawerOpen = selectedReportId !== null;

  // Build URL with current filters
  // Note: id is string for backend compatibility
  const buildUrl = useCallback(
    (overrides: {
      id?: number | string | null;
      type?: AnalyticsReportType[];
      q?: string;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedReportId : overrides.id,
        type: overrides.type ?? selectedTypes,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedReportId, selectedTypes, searchValue]
  );

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (report: AnalyticsReportRow) => {
      router.replace(buildUrl({ id: report.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL, preserve filters
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle filter changes - reset to page 1 when filters change
  const handleTypeChange = useCallback(
    (type: AnalyticsReportType, checked: boolean) => {
      const newTypes = toggleArrayValue(selectedTypes, type, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ type: newTypes }), { scroll: false });
    },
    [selectedTypes, router, buildUrl]
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

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* FilterPanel */}
      <AnalyticsFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        selectedTypes={selectedTypes}
        searchValue={searchValue}
        onTypeChange={handleTypeChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
      />

      {/* Customize Columns Panel */}
      <CustomizeColumnsPanel
        open={customizeColumnsOpen && !leftRailCollapsed}
        columns={ANALYTICS_COLUMN_OPTIONS}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={setColumnVisible}
        onRestore={resetToDefault}
        onDone={handleCustomizeColumnsDone}
        onCollapse={handleLeftRailToggle}
      />

      {/* Main content: Summary Cards + Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Header with mock indicator */}
        <div className="h-12 flex items-center justify-between px-6 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Analytics</h1>
          {isReportsDegraded && !isLoading && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-yellow-500/10 border border-yellow-500/20">
                    <Database className="h-3.5 w-3.5 text-yellow-400" />
                    <span className="text-[10px] text-yellow-400 font-medium">
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
        </div>

        {/* Real Ops Data Summary Cards */}
        <div className="p-4 border-b border-border space-y-4">
          <OpsHistorySummary
            data={opsHistory}
            isLoading={isLoadingHistory}
            isDegraded={isHistoryDegraded}
          />
          <PredictionsPerformanceCard
            data={predictionsPerformance}
            isLoading={isLoadingPerformance}
            isDegraded={isPerformanceDegraded}
          />
        </div>

        {/* Table */}
        <AnalyticsTable
          data={reports}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedReportId={selectedReportId}
          onRowClick={handleRowClick}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisibility}
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
      <AnalyticsDetailDrawer
        report={selectedReport ?? null}
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
function AnalyticsLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Analytics Page
 *
 * Master-detail pattern with URL sync (full state):
 * - Canonical: /analytics?id=123&type=model_performance&q=accuracy
 * - Uses router.replace with scroll:false
 */
export default function AnalyticsPage() {
  return (
    <Suspense fallback={<AnalyticsLoading />}>
      <AnalyticsPageContent />
    </Suspense>
  );
}
