"use client";

import { Suspense, useState, useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useDataQualityChecksApi, useDataQualityCheckApi, useColumnVisibility, usePageSize, useOpsOverview } from "@/lib/hooks";
import {
  DataQualityCheck,
  DataQualityStatus,
  DataQualityCategory,
  DataQualityFilters,
  DATA_QUALITY_STATUSES,
  DATA_QUALITY_CATEGORIES,
} from "@/lib/types";
import {
  DataQualityTable,
  DataQualityFilterPanel,
  DataQualityDetailDrawer,
  TelemetrySummaryCard,
  DATA_QUALITY_COLUMN_OPTIONS,
  DATA_QUALITY_DEFAULT_VISIBILITY,
} from "@/components/data-quality";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseStringId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";

const BASE_PATH = "/data-quality";

/**
 * Data Quality Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function DataQualityPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state - ID is string for backend compatibility (e.g., "dq_quarantined_odds_24h")
  const selectedCheckId = useMemo(
    () => parseStringId(searchParams.get("id")),
    [searchParams]
  );
  const selectedStatuses = useMemo(
    () => parseArrayParam<DataQualityStatus>(searchParams, "status", DATA_QUALITY_STATUSES),
    [searchParams]
  );
  const selectedCategories = useMemo(
    () => parseArrayParam<DataQualityCategory>(searchParams, "category", DATA_QUALITY_CATEGORIES),
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
  const { pageSize, setPageSize } = usePageSize("data-quality");

  // Column visibility with localStorage persistence
  const { columnVisibility, setColumnVisibility, setColumnVisible, resetToDefault } = useColumnVisibility(
    "data-quality",
    DATA_QUALITY_DEFAULT_VISIBILITY
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

  // Construct filters for query
  const filters: DataQualityFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    category: selectedCategories.length > 0 ? selectedCategories : undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedCategories, searchValue]);

  // Fetch real telemetry data from ops.json
  const {
    telemetry,
    isTelemetryDegraded,
    isLoading: isLoadingTelemetry,
  } = useOpsOverview();

  // Fetch data quality checks from API with mock fallback
  const {
    data: checks = [],
    pagination,
    isLoading,
    error,
    isApiDegraded: isChecksDegraded,
    refetch,
  } = useDataQualityChecksApi(filters, currentPage, pageSize);

  // Fetch check detail from API with mock fallback
  const {
    data: selectedCheck,
    isLoading: isLoadingDetail,
  } = useDataQualityCheckApi(selectedCheckId);

  // Drawer is open when there's a selected check
  const drawerOpen = selectedCheckId !== null;

  // Build URL with current filters
  // Note: id is now string for backend compatibility
  const buildUrl = useCallback(
    (overrides: {
      id?: string | null;
      status?: DataQualityStatus[];
      category?: DataQualityCategory[];
      q?: string;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedCheckId : overrides.id,
        status: overrides.status ?? selectedStatuses,
        category: overrides.category ?? selectedCategories,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedCheckId, selectedStatuses, selectedCategories, searchValue]
  );

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (check: DataQualityCheck) => {
      router.replace(buildUrl({ id: check.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL, preserve filters
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: DataQualityStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleCategoryChange = useCallback(
    (category: DataQualityCategory, checked: boolean) => {
      const newCategories = toggleArrayValue(selectedCategories, category, checked);
      router.replace(buildUrl({ category: newCategories }), { scroll: false });
    },
    [selectedCategories, router, buildUrl]
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
      <DataQualityFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        selectedStatuses={selectedStatuses}
        selectedCategories={selectedCategories}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onCategoryChange={handleCategoryChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
      />

      {/* Customize Columns Panel */}
      <CustomizeColumnsPanel
        open={customizeColumnsOpen && !leftRailCollapsed}
        columns={DATA_QUALITY_COLUMN_OPTIONS}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={setColumnVisible}
        onRestore={resetToDefault}
        onDone={handleCustomizeColumnsDone}
        onCollapse={handleLeftRailToggle}
      />

      {/* Main content: Telemetry + Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Telemetry Summary (real data from ops.json) */}
        <div className="p-4 border-b border-border">
          <TelemetrySummaryCard
            telemetry={telemetry}
            isLoading={isLoadingTelemetry}
            isDegraded={isTelemetryDegraded}
          />
        </div>

        {/* API status indicator - only show when degraded */}
        {isChecksDegraded && (
          <div className="px-4 py-2 bg-yellow-500/10 border-b border-yellow-500/30">
            <span className="text-xs text-yellow-400">
              Quality Checks (degraded - using cached data)
            </span>
          </div>
        )}

        {/* Table */}
        <DataQualityTable
          data={checks}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedCheckId={selectedCheckId}
          onRowClick={handleRowClick}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisibility}
        />

        {/* Pagination */}
        <Pagination
          currentPage={currentPage}
          totalItems={pagination?.total ?? checks.length}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
        />
      </div>

      {/* Detail Drawer (inline on desktop, sheet on mobile) */}
      <DataQualityDetailDrawer
        check={selectedCheck ?? null}
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
function DataQualityLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Data Quality Page
 *
 * Master-detail pattern with URL sync (full state):
 * - Canonical: /data-quality?id=123&status=failing&category=coverage&q=match
 * - Uses router.replace with scroll:false
 */
export default function DataQualityPage() {
  return (
    <Suspense fallback={<DataQualityLoading />}>
      <DataQualityPageContent />
    </Suspense>
  );
}
