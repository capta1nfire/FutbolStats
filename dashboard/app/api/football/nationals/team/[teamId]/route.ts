import { NextResponse, NextRequest } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/nationals/team/{team_id}.json
 * Returns national team detail
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ teamId: string }> }
) {
  const { teamId } = await params;

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/football/nationals/team/${encodeURIComponent(teamId)}.json`,
    undefined,
    { prefix: "fb-nationals-team" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
