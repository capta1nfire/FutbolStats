"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useJobRuns, useJobRun } from "@/lib/hooks";
import { JobRun, JobStatus, JobFilters, JOB_STATUSES, JOB_NAMES } from "@/lib/types";
import {
  JobsTable,
  JobsFilterPanel,
  JobDetailDrawer,
} from "@/components/jobs";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader2 } from "lucide-react";

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
  const [filterCollapsed, setFilterCollapsed] = useState(false);

  // Construct filters for query
  const filters: JobFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    jobName: selectedJobs.length > 0 ? selectedJobs : undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedJobs, searchValue]);

  // Fetch data
  const {
    data: jobRuns = [],
    isLoading,
    error,
    refetch,
  } = useJobRuns(filters);

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
        id: overrides.id ?? selectedJobId,
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

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: JobStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleJobChange = useCallback(
    (job: string, checked: boolean) => {
      const newJobs = toggleArrayValue(selectedJobs, job, checked);
      router.replace(buildUrl({ job: newJobs }), { scroll: false });
    },
    [selectedJobs, router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  return (
    <div className="h-full flex overflow-hidden">
      {/* FilterPanel */}
      <JobsFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
        selectedStatuses={selectedStatuses}
        selectedJobs={selectedJobs}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onJobChange={handleJobChange}
        onSearchChange={handleSearchChange}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Jobs</h1>
          <span className="text-sm text-muted-foreground">
            {jobRuns.length} runs
          </span>
        </div>

        {/* Table */}
        <JobsTable
          data={jobRuns}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedJobId={selectedJobId}
          onRowClick={handleRowClick}
        />
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
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading jobs...</p>
      </div>
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
