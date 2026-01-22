"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useJobRunsApi, useJobRun, useColumnVisibility, usePageSize } from "@/lib/hooks";
import { JobRun, JobStatus, JobFilters, JOB_STATUSES, JOB_NAMES } from "@/lib/types";
import { getJobRunsMock } from "@/lib/mocks";
import {
  JobsTable,
  JobsFilterPanel,
  JobDetailDrawer,
  JOBS_COLUMN_OPTIONS,
  JOBS_DEFAULT_VISIBILITY,
} from "@/components/jobs";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";

const BASE_PATH = "/jobs";

/**
 * Jobs Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function JobsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const selectedJobId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );
  const selectedStatuses = useMemo(
    () => parseArrayParam<JobStatus>(searchParams, "status", JOB_STATUSES),
    [searchParams]
  );
  const selectedJobs = useMemo(
    () => parseArrayParam<string>(searchParams, "job", [...JOB_NAMES]),
    [searchParams]
  );
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

  // Normalize URL if id param is invalid
  const selectedIdParam = searchParams.get("id");
  useEffect(() => {
    if (selectedIdParam && selectedJobId === null) {
      const params = new URLSearchParams(searchParams.toString());
      params.delete("id");
      const search = params.toString();
      router.replace(`${BASE_PATH}${search ? `?${search}` : ""}`, { scroll: false });
    }
  }, [selectedIdParam, selectedJobId, router, searchParams]);

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("jobs");

  // Column visibility with localStorage persistence
  const {
    columnVisibility,
    setColumnVisibility,
    setColumnVisible,
    resetToDefault,
  } = useColumnVisibility("jobs", JOBS_DEFAULT_VISIBILITY);

  // Construct filters for query
  const filters: JobFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    jobName: selectedJobs.length > 0 ? selectedJobs : undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedJobs, searchValue]);

  // Fetch data from API with mock fallback
  const {
    runs: apiRuns,
    pagination,
    isDegraded,
    isLoading,
    error,
    refetch,
  } = useJobRunsApi({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    jobName: selectedJobs.length > 0 ? selectedJobs[0] : undefined, // API only supports single job_name
    page: currentPage,
    limit: pageSize,
  });

  // Use API data if available, fallback to mocks
  const mockRuns = useMemo(() => getJobRunsMock(filters), [filters]);
  const jobRuns = apiRuns ?? mockRuns;

  const { data: selectedJob } = useJobRun(selectedJobId);

  // Drawer is open when there's a selected job
  const drawerOpen = selectedJobId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
      status?: JobStatus[];
      job?: string[];
      q?: string;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedJobId : overrides.id,
        status: overrides.status ?? selectedStatuses,
        job: overrides.job ?? selectedJobs,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedJobId, selectedStatuses, selectedJobs, searchValue]
  );

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (job: JobRun) => {
      router.replace(buildUrl({ id: job.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL, preserve filters
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle filter changes - reset page to 1 when filters change (P1 auditor check)
  const handleStatusChange = useCallback(
    (status: JobStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      setCurrentPage(1); // Reset pagination on filter change
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleJobChange = useCallback(
    (job: string, checked: boolean) => {
      const newJobs = toggleArrayValue(selectedJobs, job, checked);
      setCurrentPage(1); // Reset pagination on filter change
      router.replace(buildUrl({ job: newJobs }), { scroll: false });
    },
    [selectedJobs, router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      setCurrentPage(1); // Reset pagination on search change
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle "Customize Columns" link click
  const handleCustomizeColumnsClick = useCallback(() => {
    setCustomizeColumnsOpen(true);
  }, []);

  // Handle "Done" button - collapses entire Left Rail (UniFi behavior)
  const handleCustomizeColumnsDone = useCallback(() => {
    setLeftRailCollapsed(true);
    setCustomizeColumnsOpen(false);
  }, []);

  // Handle Left Rail toggle - close CustomizeColumns when collapsing
  const handleLeftRailToggle = useCallback(() => {
    const newCollapsed = !leftRailCollapsed;
    setLeftRailCollapsed(newCollapsed);
    if (newCollapsed) {
      setCustomizeColumnsOpen(false);
    }
  }, [leftRailCollapsed]);

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* Left Rail: FilterPanel */}
      <JobsFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        selectedStatuses={selectedStatuses}
        selectedJobs={selectedJobs}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onJobChange={handleJobChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
      />

      {/* Customize Columns Panel (separate column, hidden when Left Rail collapsed) */}
      <CustomizeColumnsPanel
        open={customizeColumnsOpen && !leftRailCollapsed}
        columns={JOBS_COLUMN_OPTIONS}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={setColumnVisible}
        onRestore={resetToDefault}
        onDone={handleCustomizeColumnsDone}
        onCollapse={handleLeftRailToggle}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table */}
        <JobsTable
          data={jobRuns}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedJobId={selectedJobId}
          onRowClick={handleRowClick}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisibility}
        />

        {/* Pagination */}
        <Pagination
          currentPage={currentPage}
          totalItems={isDegraded ? jobRuns.length : pagination.total}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
        />

        {/* Degraded indicator */}
        {isDegraded && (
          <div className="absolute bottom-12 left-1/2 -translate-x-1/2 text-xs text-muted-foreground/70 bg-surface px-2 py-1 rounded border border-border">
            Showing mock data
          </div>
        )}
      </div>

      {/* Detail Drawer (inline on desktop, sheet on mobile) */}
      <JobDetailDrawer
        job={selectedJob ?? null}
        open={drawerOpen}
        onClose={handleCloseDrawer}
      />
    </div>
  );
}

/**
 * Loading fallback for Suspense
 */
function JobsLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Jobs Page
 *
 * Master-detail pattern with URL sync (full state):
 * - Canonical: /jobs?id=123&status=running&status=failed&job=global_sync&q=error
 * - Uses router.replace with scroll:false
 */
export default function JobsPage() {
  return (
    <Suspense fallback={<JobsLoading />}>
      <JobsPageContent />
    </Suspense>
  );
}
