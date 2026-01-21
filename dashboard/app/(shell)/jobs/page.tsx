"use client";

import { Suspense, useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useJobRuns, useJobRun } from "@/lib/hooks";
import { JobRun, JobStatus, JobFilters } from "@/lib/types";
import {
  JobsTable,
  JobsFilterPanel,
  JobDetailDrawer,
} from "@/components/jobs";
import { Loader2 } from "lucide-react";

/**
 * Parse and validate job ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
function parseJobId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Jobs Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function JobsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL state: selected job ID (sanitized)
  const selectedIdParam = searchParams.get("id");
  const selectedJobId = parseJobId(selectedIdParam);

  // Normalize URL if id param is invalid
  useEffect(() => {
    if (selectedIdParam && selectedJobId === null) {
      // Invalid id in URL → normalize to /jobs
      router.replace("/jobs", { scroll: false });
    }
  }, [selectedIdParam, selectedJobId, router]);

  // UI state
  const [filterCollapsed, setFilterCollapsed] = useState(false);
  const [selectedStatuses, setSelectedStatuses] = useState<JobStatus[]>([]);
  const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
  const [searchValue, setSearchValue] = useState("");

  // Construct filters
  const filters: JobFilters = {
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    jobName: selectedJobs.length > 0 ? selectedJobs : undefined,
    search: searchValue || undefined,
  };

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

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (job: JobRun) => {
      router.replace(`/jobs?id=${job.id}`, { scroll: false });
    },
    [router]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace("/jobs", { scroll: false });
  }, [router]);

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: JobStatus, checked: boolean) => {
      setSelectedStatuses((prev) =>
        checked ? [...prev, status] : prev.filter((s) => s !== status)
      );
    },
    []
  );

  const handleJobChange = useCallback(
    (job: string, checked: boolean) => {
      setSelectedJobs((prev) =>
        checked ? [...prev, job] : prev.filter((j) => j !== job)
      );
    },
    []
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
        onSearchChange={setSearchValue}
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
 * Master-detail pattern with:
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (inline on desktop, right, pushes content)
 *
 * URL sync:
 * - Canonical: /jobs?id=123
 * - Uses router.replace with scroll:false
 */
export default function JobsPage() {
  return (
    <Suspense fallback={<JobsLoading />}>
      <JobsPageContent />
    </Suspense>
  );
}
