"use client";

import { useQuery } from "@tanstack/react-query";
import type { BenchmarkMatrixResponse, BenchmarkPeriod } from "@/lib/types/benchmark-matrix";

async function fetchBenchmarkMatrix(period: BenchmarkPeriod): Promise<BenchmarkMatrixResponse> {
  const res = await fetch(`/api/benchmark-matrix?period=${period}`);
  if (!res.ok) {
    throw new Error(`Benchmark matrix fetch failed: ${res.status}`);
  }
  return res.json();
}

export function useBenchmarkMatrix(period: BenchmarkPeriod = "30d") {
  return useQuery<BenchmarkMatrixResponse>({
    queryKey: ["benchmark-matrix", period],
    queryFn: () => fetchBenchmarkMatrix(period),
    staleTime: 5 * 60_000,
    retry: 1,
  });
}
