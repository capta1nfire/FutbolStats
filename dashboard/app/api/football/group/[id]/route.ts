import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/group/{id}.json
 * Returns detail for a specific league group
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!id || isNaN(Number(id))) {
    return NextResponse.json(
      { error: "Valid group ID required" },
      { status: 400 }
    );
  }

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/football/group/${id}.json`,
    undefined,
    { prefix: "fb-group" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
