import { NextResponse, NextRequest } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/audit.json
 * Forwards query params: entity_type, entity_id, limit, offset
 */
export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const { data, status, requestId } = await proxyFetch(
    "/dashboard/admin/audit.json",
    searchParams.size > 0 ? searchParams : undefined,
    { prefix: "admin-audit" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
