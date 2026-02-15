import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/team/{id}/squad-stats.json
 * Returns per-player seasonal aggregates from match_player_stats.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!id || isNaN(Number(id))) {
    return NextResponse.json({ error: "Valid team ID required" }, { status: 400 });
  }

  const season = request.nextUrl.searchParams.get("season");
  const queryParams = season ? new URLSearchParams({ season }) : undefined;

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/admin/team/${id}/squad-stats.json`,
    queryParams,
    { prefix: "fb-team-squad-stats" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}

