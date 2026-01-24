import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy route handler for /dashboard/predictions/missing.json
 *
 * Query params:
 * - hours: number (default: 48, range: 1-168)
 * - league_ids: comma-separated league IDs (optional)
 * - page: number (default: 1)
 * - limit: number (default: 20, max: 100)
 *
 * Returns paginated list of matches missing predictions.
 * Hardened with timeout, retry, and no internal detail leakage.
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);

  // Validate and sanitize query params
  const hours = sanitizeHours(searchParams.get("hours"));
  const leagueIds = sanitizeLeagueIds(searchParams.get("league_ids"));
  const page = sanitizePage(searchParams.get("page"));
  const limit = sanitizeLimit(searchParams.get("limit"));

  const params = new URLSearchParams({
    hours: hours.toString(),
    page: page.toString(),
    limit: limit.toString(),
  });

  // Only add league_ids if provided
  if (leagueIds) {
    params.set("league_ids", leagueIds);
  }

  const { data, status, requestId } = await proxyFetch(
    "/dashboard/predictions/missing.json",
    params,
    { prefix: "pred-missing" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}

/**
 * Sanitize hours parameter (1-168, default 48)
 */
function sanitizeHours(value: string | null): number {
  const num = parseInt(value || "48", 10);
  if (isNaN(num) || num < 1) return 48;
  return Math.min(num, 168); // Max 7 days
}

/**
 * Sanitize league_ids parameter (comma-separated integers)
 */
function sanitizeLeagueIds(value: string | null): string | null {
  if (!value) return null;

  // Parse and validate each ID
  const ids = value
    .split(",")
    .map((id) => parseInt(id.trim(), 10))
    .filter((id) => !isNaN(id) && id > 0);

  return ids.length > 0 ? ids.join(",") : null;
}

/**
 * Sanitize page parameter
 */
function sanitizePage(value: string | null): number {
  const num = parseInt(value || "1", 10);
  if (isNaN(num) || num < 1) return 1;
  return num;
}

/**
 * Sanitize limit parameter (max 100)
 */
function sanitizeLimit(value: string | null): number {
  const num = parseInt(value || "20", 10);
  if (isNaN(num) || num < 1) return 20;
  return Math.min(num, 100);
}
