import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/world-cup-2026/overview.json
 * Returns World Cup 2026 overview (status, summary, alerts, upcoming)
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/football/world-cup-2026/overview.json",
    undefined,
    { prefix: "fb-wc2026-overview" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
