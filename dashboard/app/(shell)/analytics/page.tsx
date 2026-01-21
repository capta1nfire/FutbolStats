"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAnalyticsReports, useAnalyticsReport } from "@/lib/hooks";
import {
  AnalyticsReportRow,
  AnalyticsReportType,
  AnalyticsFilters,
  ANALYTICS_REPORT_TYPES,
} from "@/lib/types";
import {
  AnalyticsTable,
  AnalyticsFilterPanel,
  AnalyticsDetailDrawer,
} from "@/components/analytics";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader2 } from "lucide-react";

const BASE_PATH = "/analytics";

/**
 * Analytics Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function AnalyticsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const selectedReportId = useMemo(
    () => parseNumericId(searchParams.get("id")),
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

  // Normalize URL if id param is invalid
  const selectedIdParam = searchParams.get("id");
  useEffect(() => {
    if (selectedIdParam && selectedReportId === null) {
      const params = new URLSearchParams(searchParams.toString());
      params.delete("id");
      const search = params.toString();
      router.replace(`${BASE_PATH}${search ? `?${search}` : ""}`, { scroll: false });
    }
  }, [selectedIdParam, selectedReportId, router, searchParams]);

  // UI state (non-URL)
  const [filterCollapsed, setFilterCollapsed] = useState(false);

  // Construct filters for query
  const filters: AnalyticsFilters = useMemo(() => ({
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    search: searchValue || undefined,
  }), [selectedTypes, searchValue]);

  // Fetch data
  const {
    data: reports = [],
    isLoading,
    error,
    refetch,
  } = useAnalyticsReports(filters);

  const {
    data: selectedReport,
    isLoading: isLoadingDetail,
  } = useAnalyticsReport(selectedReportId);

  // Drawer is open when there's a selected report
  const drawerOpen = selectedReportId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
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

  // Handle filter changes
  const handleTypeChange = useCallback(
    (type: AnalyticsReportType, checked: boolean) => {
      const newTypes = toggleArrayValue(selectedTypes, type, checked);
      router.replace(buildUrl({ type: newTypes }), { scroll: false });
    },
    [selectedTypes, router, buildUrl]
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
      <AnalyticsFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
        selectedTypes={selectedTypes}
        searchValue={searchValue}
        onTypeChange={handleTypeChange}
        onSearchChange={handleSearchChange}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Analytics</h1>
          <span className="text-sm text-muted-foreground">
            {reports.length} reports
          </span>
        </div>

        {/* Table */}
        <AnalyticsTable
          data={reports}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedReportId={selectedReportId}
          onRowClick={handleRowClick}
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
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading analytics...</p>
      </div>
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
