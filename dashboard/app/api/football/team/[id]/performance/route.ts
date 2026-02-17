import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/team/{id}/performance.json
 * Returns match-by-match performance data for charts.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!id || isNaN(Number(id))) {
    return NextResponse.json({ error: "Valid team ID required" }, { status: 400 });
  }

  const sp = request.nextUrl.searchParams;
  const queryParams = new URLSearchParams();

  const leagueId = sp.get("league_id");
  if (leagueId) queryParams.set("league_id", leagueId);

  const season = sp.get("season");
  if (season) queryParams.set("season", season);

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/admin/team/${id}/performance.json`,
    queryParams.toString() ? queryParams : undefined,
    { prefix: "fb-team-performance" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
