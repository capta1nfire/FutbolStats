"use client";

import { useQuery } from "@tanstack/react-query";
import { JobRun, JobDefinition, JobFilters, JobStatus } from "@/lib/types";
import {
  mockConfig,
  simulateDelay,
  getJobRunsMock,
  getJobRunMock,
  getJobDefinitionsMock,
} from "@/lib/mocks";
import {
  parseJobRuns,
  extractPagination,
  extractMetadata,
  extractJobsSummary,
  buildJobDefinitions,
  mapStatusFilter,
  JobsPagination,
  JobsSummary,
} from "@/lib/api/jobs";

/**
 * Response from useJobRunsApi hook
 */
export interface UseJobRunsApiResult {
  /** Parsed job runs, null if unavailable */
  runs: JobRun[] | null;
  /** Pagination info */
  pagination: JobsPagination;
  /** Per-job health summary */
  jobsSummary: JobsSummary;
  /** Job definitions built from summary */
  jobDefinitions: JobDefinition[];
  /** True if data fetch failed or parsing failed */
  isDegraded: boolean;
  /** Request ID for debugging */
  requestId?: string;
  /** When backend generated this data */
  generatedAt: string | null;
  /** Whether data is from backend cache */
  cached: boolean;
  /** Age of backend cache in seconds */
  cacheAgeSeconds: number;
  /** Loading state */
  isLoading: boolean;
  /** Error object if fetch failed */
  error: Error | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Query params for the API
 */
interface JobsQueryParams {
  status?: string;
  job_name?: string;
  hours?: number;
  page?: number;
  limit?: number;
}

/**
 * Internal response type
 */
interface JobsData {
  runs: JobRun[] | null;
  pagination: JobsPagination;
  jobsSummary: JobsSummary;
  requestId?: string;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Build query string from params
 */
function buildQueryString(params: JobsQueryParams): string {
  const searchParams = new URLSearchParams();

  if (params.status) searchParams.set("status", params.status);
  if (params.job_name) searchParams.set("job_name", params.job_name);
  if (params.hours) searchParams.set("hours", params.hours.toString());
  if (params.page) searchParams.set("page", params.page.toString());
  if (params.limit) searchParams.set("limit", params.limit.toString());

  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

/**
 * Fetch job runs from proxy endpoint
 */
async function fetchJobRuns(params: JobsQueryParams): Promise<JobsData> {
  const queryString = buildQueryString(params);
  const response = await fetch(`/api/jobs${queryString}`, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  const requestId = response.headers.get("x-request-id") || undefined;

  if (!response.ok) {
    return {
      runs: null,
      pagination: { total: 0, page: 1, limit: 50, pages: 1 },
      jobsSummary: {},
      requestId,
      generatedAt: null,
      cached: false,
      cacheAgeSeconds: 0,
    };
  }

  const data = await response.json();
  const runs = parseJobRuns(data);
  const pagination = extractPagination(data);
  const jobsSummary = extractJobsSummary(data);
  const metadata = extractMetadata(data);

  return {
    runs,
    pagination,
    jobsSummary,
    requestId,
    generatedAt: metadata.generatedAt,
    cached: metadata.cached,
    cacheAgeSeconds: metadata.cacheAgeSeconds,
  };
}

/**
 * Hook to fetch job runs from backend via /api/jobs proxy
 *
 * Features:
 * - Enterprise-safe: uses same-origin proxy, no secrets exposed
 * - Graceful degradation: returns isDegraded=true if fetch/parse fails
 * - Pagination support
 * - Status/job name filtering
 * - Includes jobs_summary with per-job health
 *
 * Usage:
 * ```tsx
 * const { runs, isDegraded, pagination, jobsSummary, isLoading } = useJobRunsApi({
 *   status: ["failed"],
 *   page: 1,
 *   limit: 50,
 * });
 *
 * if (isLoading) return <Loader />;
 *
 * const displayRuns = runs ?? mockRuns;
 * ```
 */
export function useJobRunsApi(options?: {
  status?: JobStatus[];
  jobName?: string;
  hours?: number;
  page?: number;
  limit?: number;
  enabled?: boolean;
}): UseJobRunsApiResult {
  const {
    status = [],
    jobName,
    hours = 24, // 24 hours default
    page = 1,
    limit = 50,
    enabled = true,
  } = options || {};

  // Map frontend status filter to backend
  const backendStatus = mapStatusFilter(status);

  const queryParams: JobsQueryParams = {
    status: backendStatus,
    job_name: jobName,
    hours,
    page,
    limit,
  };

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["jobs-api", queryParams],
    queryFn: () => fetchJobRuns(queryParams),
    retry: 1,
    staleTime: 30_000, // Consider data fresh for 30s
    refetchInterval: 30_000, // Refetch every 30s
    refetchOnWindowFocus: false,
    throwOnError: false,
    enabled,
  });

  const runs = data?.runs ?? null;
  const pagination = data?.pagination ?? { total: 0, page: 1, limit: 50, pages: 1 };
  const jobsSummary = data?.jobsSummary ?? {};
  const jobDefinitions = buildJobDefinitions(jobsSummary);
  const requestId = data?.requestId;
  const generatedAt = data?.generatedAt ?? null;
  const cached = data?.cached ?? false;
  const cacheAgeSeconds = data?.cacheAgeSeconds ?? 0;
  const isDegraded = !!error || runs === null;

  return {
    runs,
    pagination,
    jobsSummary,
    jobDefinitions,
    isDegraded,
    requestId,
    generatedAt,
    cached,
    cacheAgeSeconds,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}

/**
 * Hook to fetch job runs (mock fallback)
 * @deprecated Use useJobRunsApi for real data with mock fallback
 */
export function useJobRuns(filters?: JobFilters) {
  return useQuery<JobRun[], Error>({
    queryKey: ["jobRuns", filters],
    queryFn: async () => {
      await simulateDelay();
      return getJobRunsMock(filters, mockConfig.scenario);
    },
    enabled: mockConfig.useMockData,
    refetchInterval: 30000, // Refetch every 30s for live updates
  });
}

/**
 * Hook to fetch a single job run by ID
 */
export function useJobRun(id: number | null) {
  return useQuery<JobRun | undefined, Error>({
    queryKey: ["jobRun", id],
    queryFn: async () => {
      await simulateDelay();
      if (mockConfig.scenario === "error") {
        throw new Error("Failed to fetch job run");
      }
      return id ? getJobRunMock(id) : undefined;
    },
    enabled: mockConfig.useMockData && id !== null,
  });
}

/**
 * Hook to fetch job definitions
 */
export function useJobDefinitions() {
  return useQuery<JobDefinition[], Error>({
    queryKey: ["jobDefinitions"],
    queryFn: async () => {
      await simulateDelay();
      if (mockConfig.scenario === "error") {
        throw new Error("Failed to fetch job definitions");
      }
      return getJobDefinitionsMock();
    },
    enabled: mockConfig.useMockData,
    staleTime: 60000, // Consider fresh for 1 minute
  });
}
