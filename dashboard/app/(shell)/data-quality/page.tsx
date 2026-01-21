"use client";

import { Suspense, useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useDataQualityChecks, useDataQualityCheck } from "@/lib/hooks";
import {
  DataQualityCheck,
  DataQualityStatus,
  DataQualityCategory,
  DataQualityFilters,
} from "@/lib/types";
import {
  DataQualityTable,
  DataQualityFilterPanel,
  DataQualityDetailDrawer,
} from "@/components/data-quality";
import { Loader2 } from "lucide-react";

/**
 * Parse and validate check ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
function parseCheckId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Data Quality Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function DataQualityPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL state: selected check ID (sanitized)
  const selectedIdParam = searchParams.get("id");
  const selectedCheckId = parseCheckId(selectedIdParam);

  // Normalize URL if id param is invalid
  useEffect(() => {
    if (selectedIdParam && selectedCheckId === null) {
      // Invalid id in URL → normalize to /data-quality
      router.replace("/data-quality", { scroll: false });
    }
  }, [selectedIdParam, selectedCheckId, router]);

  // UI state
  const [filterCollapsed, setFilterCollapsed] = useState(false);
  const [selectedStatuses, setSelectedStatuses] = useState<DataQualityStatus[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<DataQualityCategory[]>([]);
  const [searchValue, setSearchValue] = useState("");

  // Construct filters
  const filters: DataQualityFilters = {
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    category: selectedCategories.length > 0 ? selectedCategories : undefined,
    search: searchValue || undefined,
  };

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

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (check: DataQualityCheck) => {
      router.replace(`/data-quality?id=${check.id}`, { scroll: false });
    },
    [router]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace("/data-quality", { scroll: false });
  }, [router]);

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: DataQualityStatus, checked: boolean) => {
      setSelectedStatuses((prev) =>
        checked ? [...prev, status] : prev.filter((s) => s !== status)
      );
    },
    []
  );

  const handleCategoryChange = useCallback(
    (category: DataQualityCategory, checked: boolean) => {
      setSelectedCategories((prev) =>
        checked ? [...prev, category] : prev.filter((c) => c !== category)
      );
    },
    []
  );

  // Calculate summary counts
  const passingCount = checks.filter((c) => c.status === "passing").length;
  const warningCount = checks.filter((c) => c.status === "warning").length;
  const failingCount = checks.filter((c) => c.status === "failing").length;

  return (
    <div className="h-full flex overflow-hidden">
      {/* FilterPanel */}
      <DataQualityFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
        selectedStatuses={selectedStatuses}
        selectedCategories={selectedCategories}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onCategoryChange={handleCategoryChange}
        onSearchChange={setSearchValue}
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
 * Master-detail pattern with:
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (inline on desktop, right, pushes content)
 *
 * URL sync:
 * - Canonical: /data-quality?id=123
 * - Uses router.replace with scroll:false
 */
export default function DataQualityPage() {
  return (
    <Suspense fallback={<DataQualityLoading />}>
      <DataQualityPageContent />
    </Suspense>
  );
}
