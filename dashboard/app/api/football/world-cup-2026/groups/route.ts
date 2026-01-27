import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/world-cup-2026/groups.json
 * Returns World Cup 2026 groups list
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/football/world-cup-2026/groups.json",
    undefined,
    { prefix: "fb-wc2026-groups" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
