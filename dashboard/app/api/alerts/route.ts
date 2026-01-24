import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy route handler for /dashboard/ops/alerts.json
 *
 * Query params:
 * - status: "firing" | "resolved" | "all" (default: "all")
 * - limit: number (default: 50, max: 100)
 * - unread_only: boolean (default: false)
 *
 * Returns paginated alerts list from Grafana webhook → DB.
 * Hardened with timeout, retry, and no internal detail leakage.
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);

  // Validate and sanitize query params
  const status = sanitizeStatus(searchParams.get("status"));
  const limit = sanitizeLimit(searchParams.get("limit"));
  const unreadOnly = searchParams.get("unread_only") === "true";

  const params = new URLSearchParams();
  if (status !== "all") {
    params.set("status", status);
  }
  params.set("limit", limit.toString());
  if (unreadOnly) {
    params.set("unread_only", "true");
  }

  const { data, status: httpStatus, requestId } = await proxyFetch(
    "/dashboard/ops/alerts.json",
    params.toString() ? params : undefined,
    { prefix: "alerts" }
  );

  return NextResponse.json(data, {
    status: httpStatus,
    headers: standardHeaders(requestId),
  });
}

/**
 * Sanitize status parameter
 */
function sanitizeStatus(value: string | null): "firing" | "resolved" | "all" {
  if (value === "firing" || value === "resolved") {
    return value;
  }
  return "all";
}

/**
 * Sanitize limit parameter (max 100)
 */
function sanitizeLimit(value: string | null): number {
  const num = parseInt(value || "50", 10);
  if (isNaN(num) || num < 1) return 50;
  return Math.min(num, 100);
}
