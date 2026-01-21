"use client";

import { useQuery } from "@tanstack/react-query";
import { JobRun, JobDefinition, JobFilters } from "@/lib/types";
import {
  mockConfig,
  simulateDelay,
  getJobRunsMock,
  getJobRunMock,
  getJobDefinitionsMock,
} from "@/lib/mocks";

/**
 * Hook to fetch job runs
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
