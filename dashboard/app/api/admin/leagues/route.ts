import { NextResponse, NextRequest } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/leagues.json
 * Forwards query params: search, country, kind, is_active, source
 */
export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const { data, status, requestId } = await proxyFetch(
    "/dashboard/admin/leagues.json",
    searchParams.size > 0 ? searchParams : undefined,
    { prefix: "admin-leagues" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
