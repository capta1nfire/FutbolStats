import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/league/{id}.json
 * Returns detail for a specific league
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!id || isNaN(Number(id))) {
    return NextResponse.json(
      { error: "Valid league ID required" },
      { status: 400 }
    );
  }

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/football/league/${id}.json`,
    undefined,
    { prefix: "fb-league" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
