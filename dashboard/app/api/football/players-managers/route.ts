import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/players-managers.json
 * Returns injuries or managers global view.
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const view = searchParams.get("view") || "injuries";
  const leagueId = searchParams.get("league_id");
  const limit = searchParams.get("limit");

  const params = new URLSearchParams({ view });
  if (leagueId) params.set("league_id", leagueId);
  if (limit) params.set("limit", limit);

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/admin/players-managers.json?${params.toString()}`,
    undefined,
    { prefix: "fb-players-managers" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
