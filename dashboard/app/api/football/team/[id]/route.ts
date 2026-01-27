import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/team/{id}.json
 * Returns Team 360 detail (temporary endpoint for P0)
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!id || isNaN(Number(id))) {
    return NextResponse.json(
      { error: "Valid team ID required" },
      { status: 400 }
    );
  }

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/admin/team/${id}.json`,
    undefined,
    { prefix: "fb-team" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
