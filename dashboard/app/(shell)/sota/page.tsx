"use client";

import { Suspense, useState, useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useDataQualityChecksApi, useDataQualityCheckApi, useOpsOverview } from "@/lib/hooks";
import {
  DataQualityCheck,
  DataQualityFilters,
} from "@/lib/types";
import {
  DataQualityTable,
  DataQualityDetailDrawer,
} from "@/components/data-quality";
import { SotaEnrichmentSection } from "@/components/overview";
import { Pagination } from "@/components/tables";
import {
  parseStringId,
  buildSearchParams,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";
import { Database, Sparkles, ClipboardCheck } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
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
 * Default column visibility for SOTA table
 * (simpler than Data Quality - hide some columns)
 */
const SOTA_COLUMN_VISIBILITY: Record<string, boolean> = {
  status: true,
  name: true,
  category: false, // Always coverage, no need to show
  current: true,
  threshold: true,
  affected: true,
  lastRun: true,
};

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

  // Parse URL state - ID is string for backend compatibility
  const selectedCheckId = useMemo(
    () => parseStringId(searchParams.get("id")),
    [searchParams]
  );

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

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

  // Combined loading state
  const isLoading = isOpsLoading || isDqLoading;

  // Any degradation (ops or data quality)
  const isAnyDegraded = isSotaEnrichmentDegraded || isDqDegraded;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: { id?: string | null }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedCheckId : overrides.id,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedCheckId]
  );

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

  // Pagination - client-side since we're filtering
  const paginatedChecks = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return checks.slice(start, start + pageSize);
  }, [checks, currentPage, pageSize]);

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Header */}
        <div className="h-12 flex items-center justify-between px-6 border-b border-border shrink-0">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <h1 className="text-lg font-semibold text-foreground">SOTA</h1>
          </div>
          {isAnyDegraded && !isLoading && (
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

        {/* Scrollable content area */}
        <ScrollArea className="flex-1">
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
                  columnVisibility={SOTA_COLUMN_VISIBILITY}
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
 * - /sota?id=dq_understat_coverage_ft_14d - with drawer open
 */
export default function SotaPage() {
  return (
    <Suspense fallback={<SotaLoading />}>
      <SotaPageContent />
    </Suspense>
  );
}
