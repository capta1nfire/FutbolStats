"use client";

import { useQuery } from "@tanstack/react-query";
import { UpcomingMatch } from "@/lib/types";
import { useRegion } from "@/components/providers/RegionProvider";

/**
 * Extended match for today with status and scores
 */
export interface TodayMatchItem extends UpcomingMatch {
  status?: string;
  homeScore?: number;
  awayScore?: number;
  elapsed?: number;
  elapsedExtra?: number;
}

/**
 * Response from useTodayMatches hook
 */
export interface UseTodayMatchesResult {
  /** Parsed today's matches, null if unavailable */
  matches: TodayMatchItem[] | null;
  /** True if data fetch failed or parsing failed */
  isDegraded: boolean;
  /** Loading state */
  isLoading: boolean;
  /** Error object if fetch failed */
  error: Error | null;
  /** Refetch function */
  refetch: () => void;
}

/**
 * Parse matches from backend response
 * Backend returns: { data: { matches: [...] } }
 */
function parseMatches(response: unknown): TodayMatchItem[] | null {
  if (!response || typeof response !== "object") return null;

  // Backend wraps data in a "data" object
  const data = (response as Record<string, unknown>).data as Record<string, unknown> | undefined;
  const matches = data?.matches ?? (response as Record<string, unknown>).matches;

  if (!Array.isArray(matches)) return null;

  return matches.map((m: Record<string, unknown>) => {
    // Score can come as score.home/score.away or home_goals/away_goals
    const score = m.score as Record<string, unknown> | undefined;
    const homeScore = score?.home != null ? Number(score.home) : (m.home_goals != null ? Number(m.home_goals) : undefined);
    const awayScore = score?.away != null ? Number(score.away) : (m.away_goals != null ? Number(m.away_goals) : undefined);

    return {
      id: Number(m.id) || 0,
      home: String(m.home || ""),
      away: String(m.away || ""),
      kickoffISO: String(m.kickoff_iso || m.kickoffISO || ""),
      leagueName: String(m.league_name || m.leagueName || ""),
      // Has prediction if model_a exists
      hasPrediction: Boolean(m.model_a),
      status: String(m.status || "NS"),
      homeScore,
      awayScore,
      elapsed: m.elapsed != null ? Number(m.elapsed) : undefined,
      elapsedExtra: m.elapsed_extra != null ? Number(m.elapsed_extra) : undefined,
    };
  });
}

/**
 * Convert UTC ISO string to local date string in given timezone
 */
function utcToLocalDate(isoUtc: string, timeZone: string): string {
  const date = new Date(isoUtc);
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);

  const year = parts.find((p) => p.type === "year")?.value || "2026";
  const month = parts.find((p) => p.type === "month")?.value || "01";
  const day = parts.find((p) => p.type === "day")?.value || "01";

  return `${year}-${month}-${day}`;
}

/**
 * Filter matches to only include those with kickoff on the given local date
 */
function filterMatchesByLocalDate(
  matches: TodayMatchItem[],
  localDate: string,
  timeZone: string
): TodayMatchItem[] {
  return matches.filter((match) => {
    // Convert kickoff UTC to local date in user's timezone
    const kickoffLocalDate = utcToLocalDate(match.kickoffISO, timeZone);
    return kickoffLocalDate === localDate;
  });
}

/**
 * Calculate UTC time range that covers "today" in any timezone
 * Returns from_time and to_time in ISO format
 */
function getTodayUtcRange(localDate: string): { fromTime: string; toTime: string } {
  // Parse the local date (YYYY-MM-DD)
  const [year, month, day] = localDate.split("-").map(Number);

  // To cover "today" in any timezone (UTC-12 to UTC+14),
  // we need a window from (localDate - 14 hours) to (localDate + 1 day + 12 hours)
  // Simplified: fetch from start of day UTC-14h to end of day UTC+14h
  const startOfDay = new Date(Date.UTC(year, month - 1, day, 0, 0, 0));
  const endOfDay = new Date(Date.UTC(year, month - 1, day, 23, 59, 59));

  // Expand by 14 hours each direction to cover all timezones
  const fromTime = new Date(startOfDay.getTime() - 14 * 60 * 60 * 1000);
  const toTime = new Date(endOfDay.getTime() + 14 * 60 * 60 * 1000);

  return {
    fromTime: fromTime.toISOString(),
    toTime: toTime.toISOString(),
  };
}

/**
 * Fetch matches from backend for a specific date range
 */
async function fetchMatches(fromTime: string, toTime: string): Promise<TodayMatchItem[] | null> {
  // Fetch all matches (including live/finished) with explicit date range
  const params = new URLSearchParams({
    limit: "200",
    status: "ALL",
    from_time: fromTime,
    to_time: toTime,
  });

  const response = await fetch(`/api/matches?${params.toString()}`, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    return null;
  }

  const data = await response.json();
  return parseMatches(data);
}

/**
 * Hook to fetch today's matches
 *
 * Features:
 * - Fetches all matches scheduled for today (in user's timezone)
 * - Includes live (1H, 2H, HT) and finished (FT) matches
 * - Auto-refetch every 60s
 * - Graceful degradation on errors
 */
export function useTodayMatches(): UseTodayMatchesResult {
  const { getTodayLocalDate, region } = useRegion();
  const today = getTodayLocalDate();
  const timeZone = region.timeZone;

  // Calculate UTC range that covers "today" in user's timezone
  const { fromTime, toTime } = getTodayUtcRange(today);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["today-matches", today, timeZone],
    queryFn: () => fetchMatches(fromTime, toTime),
    retry: 1,
    staleTime: 30_000,
    refetchInterval: 60_000,
    refetchOnWindowFocus: false,
    throwOnError: false,
  });

  // Filter matches client-side to only include today's matches in user's timezone
  const matches = data ? filterMatchesByLocalDate(data, today, timeZone) : null;
  const isDegraded = !!error || data === null;

  return {
    matches,
    isDegraded,
    isLoading,
    error: error as Error | null,
    refetch: () => refetch(),
  };
}
