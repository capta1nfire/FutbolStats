"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useDataQualityChecks, useDataQualityCheck, useColumnVisibility } from "@/lib/hooks";
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
  DATA_QUALITY_COLUMN_OPTIONS,
  DATA_QUALITY_DEFAULT_VISIBILITY,
} from "@/components/data-quality";
import { CustomizeColumnsPanel } from "@/components/tables";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader2 } from "lucide-react";

const BASE_PATH = "/data-quality";

/**
 * Data Quality Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function DataQualityPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const selectedCheckId = useMemo(
    () => parseNumericId(searchParams.get("id")),
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

  // Normalize URL if id param is invalid
  const selectedIdParam = searchParams.get("id");
  useEffect(() => {
    if (selectedIdParam && selectedCheckId === null) {
      const params = new URLSearchParams(searchParams.toString());
      params.delete("id");
      const search = params.toString();
      router.replace(`${BASE_PATH}${search ? `?${search}` : ""}`, { scroll: false });
    }
  }, [selectedIdParam, selectedCheckId, router, searchParams]);

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

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

  // Fetch data
  const {
    data: checks = [],
    isLoading,
    error,
    refetch,
  } = useDataQualityChecks(filters);

  const {
    data: selectedCheck,
    isLoading: isLoadingDetail,
  } = useDataQualityCheck(selectedCheckId);

  // Drawer is open when there's a selected check
  const drawerOpen = selectedCheckId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
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

  // Calculate summary counts
  const passingCount = checks.filter((c) => c.status === "passing").length;
  const warningCount = checks.filter((c) => c.status === "warning").length;
  const failingCount = checks.filter((c) => c.status === "failing").length;

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

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Data Quality</h1>
          <div className="flex items-center gap-4 text-sm">
            <span className="text-green-400">{passingCount} passing</span>
            <span className="text-yellow-400">{warningCount} warning</span>
            <span className="text-red-400">{failingCount} failing</span>
          </div>
        </div>

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
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading data quality checks...</p>
      </div>
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
