import { NextResponse, NextRequest } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/world-cup-2026/group/{group}.json
 * Returns World Cup 2026 group detail (standings, matches)
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ group: string }> }
) {
  const { group } = await params;

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/football/world-cup-2026/group/${encodeURIComponent(group)}.json`,
    undefined,
    { prefix: "fb-wc2026-group" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
