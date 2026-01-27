import { NextResponse, NextRequest } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/league-group/{id}.json
 */
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/admin/league-group/${encodeURIComponent(id)}.json`,
    undefined,
    { prefix: "admin-league-group" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
