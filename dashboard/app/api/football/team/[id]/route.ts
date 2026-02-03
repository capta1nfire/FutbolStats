import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, proxyFetchMutation, standardHeaders } from "@/lib/api/proxy-utils";

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

/**
 * PATCH team wiki fields
 *
 * Proxies to backend PATCH /dashboard/admin/team/{id}.json
 * Tolerant of backend not supporting this yet (returns 501/404/405).
 */
export async function PATCH(
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
    `/dashboard/admin/team/${id}.json`,
    "PATCH",
    body,
    { prefix: "fb-team-patch" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
