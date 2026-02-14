import { useQuery } from "@tanstack/react-query";
import type { CoverageMapData, CoverageWindow } from "@/lib/types/coverage-map";

function currentSeason(): number {
  const now = new Date();
  return now.getMonth() >= 6 ? now.getFullYear() : now.getFullYear() - 1;
}

async function fetchCoverageMap(
  timeWindow: CoverageWindow
): Promise<CoverageMapData> {
  const params = new URLSearchParams({ window: timeWindow });
  if (timeWindow === "season_to_date") {
    params.set("season", String(currentSeason()));
  }
  const res = await fetch(`/api/coverage-map?${params}`);
  if (!res.ok) throw new Error(`Coverage map API ${res.status}`);
  const json = await res.json();
  return json.data ?? json;
}

export function useCoverageMap(timeWindow: CoverageWindow = "since_2023") {
  return useQuery({
    queryKey: ["coverage-map", timeWindow],
    queryFn: () => fetchCoverageMap(timeWindow),
    staleTime: 15 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
}
