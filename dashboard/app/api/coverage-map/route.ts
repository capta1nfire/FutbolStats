import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy route for /dashboard/coverage-map.json
 *
 * Passes through all query params (window, season, league_ids, etc.)
 * to the backend coverage map endpoint.
 */
export async function GET(request: NextRequest) {
  const qs = request.nextUrl.searchParams.toString();
  const path = `/dashboard/coverage-map.json${qs ? `?${qs}` : ""}`;

  const { data, status, requestId } = await proxyFetch(path, undefined, {
    prefix: "cm",
    timeoutMs: 15000, // coverage map query can take ~5s
  });

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
