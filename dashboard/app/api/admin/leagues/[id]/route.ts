import { NextResponse, NextRequest } from "next/server";
import { proxyFetchMutation, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for PATCH /dashboard/admin/leagues/{id}.json
 * Forwards JSON body to backend for league mutations
 */
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body" },
      { status: 400 }
    );
  }

  const { data, status, requestId } = await proxyFetchMutation(
    `/dashboard/admin/leagues/${encodeURIComponent(id)}.json`,
    "PATCH",
    body,
    { prefix: "admin-league-patch" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
