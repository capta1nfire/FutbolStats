"use client";

import { Suspense, useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAnalyticsReports, useAnalyticsReport } from "@/lib/hooks";
import {
  AnalyticsReportRow,
  AnalyticsReportType,
  AnalyticsFilters,
} from "@/lib/types";
import {
  AnalyticsTable,
  AnalyticsFilterPanel,
  AnalyticsDetailDrawer,
} from "@/components/analytics";
import { Loader2 } from "lucide-react";

/**
 * Parse and validate report ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
function parseReportId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Analytics Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function AnalyticsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL state: selected report ID (sanitized)
  const selectedIdParam = searchParams.get("id");
  const selectedReportId = parseReportId(selectedIdParam);

  // Normalize URL if id param is invalid
  useEffect(() => {
    if (selectedIdParam && selectedReportId === null) {
      // Invalid id in URL → normalize to /analytics
      router.replace("/analytics", { scroll: false });
    }
  }, [selectedIdParam, selectedReportId, router]);

  // UI state
  const [filterCollapsed, setFilterCollapsed] = useState(false);
  const [selectedTypes, setSelectedTypes] = useState<AnalyticsReportType[]>([]);
  const [searchValue, setSearchValue] = useState("");

  // Construct filters
  const filters: AnalyticsFilters = {
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    search: searchValue || undefined,
  };

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

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (report: AnalyticsReportRow) => {
      router.replace(`/analytics?id=${report.id}`, { scroll: false });
    },
    [router]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace("/analytics", { scroll: false });
  }, [router]);

  // Handle filter changes
  const handleTypeChange = useCallback(
    (type: AnalyticsReportType, checked: boolean) => {
      setSelectedTypes((prev) =>
        checked ? [...prev, type] : prev.filter((t) => t !== type)
      );
    },
    []
  );

  return (
    <div className="h-full flex overflow-hidden">
      {/* FilterPanel */}
      <AnalyticsFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
        selectedTypes={selectedTypes}
        searchValue={searchValue}
        onTypeChange={handleTypeChange}
        onSearchChange={setSearchValue}
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
 * Master-detail pattern with:
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (inline on desktop, right, pushes content)
 *
 * URL sync:
 * - Canonical: /analytics?id=123
 * - Uses router.replace with scroll:false
 */
export default function AnalyticsPage() {
  return (
    <Suspense fallback={<AnalyticsLoading />}>
      <AnalyticsPageContent />
    </Suspense>
  );
}
