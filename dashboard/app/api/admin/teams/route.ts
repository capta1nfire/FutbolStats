import { NextResponse, NextRequest } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/teams.json
 * Forwards query params: type, country, search, limit, offset
 */
export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const { data, status, requestId } = await proxyFetch(
    "/dashboard/admin/teams.json",
    searchParams.size > 0 ? searchParams : undefined,
    { prefix: "admin-teams" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
