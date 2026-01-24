import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy route handler for /dashboard/sentry/issues.json
 *
 * Query params:
 * - range: "1h" | "24h" | "7d" (default: "24h")
 * - page: number (default: 1)
 * - limit: number (default: 20, max: 100)
 *
 * Returns paginated Sentry issues list.
 * Hardened with timeout, retry, and no internal detail leakage.
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);

  // Validate and sanitize query params
  const range = sanitizeRange(searchParams.get("range"));
  const page = sanitizePage(searchParams.get("page"));
  const limit = sanitizeLimit(searchParams.get("limit"));

  const params = new URLSearchParams({
    range,
    page: page.toString(),
    limit: limit.toString(),
  });

  const { data, status, requestId } = await proxyFetch(
    "/dashboard/sentry/issues.json",
    params,
    { prefix: "sentry-issues" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}

/**
 * Sanitize range parameter
 */
function sanitizeRange(value: string | null): string {
  const validRanges = ["1h", "24h", "7d"];
  if (value && validRanges.includes(value)) {
    return value;
  }
  return "24h";
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
