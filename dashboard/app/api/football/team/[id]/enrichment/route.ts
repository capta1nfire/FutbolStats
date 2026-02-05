import { NextRequest, NextResponse } from "next/server";
import { proxyFetchMutation, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * PUT team enrichment override
 *
 * Proxies to backend PUT /dashboard/admin/team/{id}/enrichment
 * Upserts manual override values for team enrichment.
 */
export async function PUT(
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
    `/dashboard/admin/team/${id}/enrichment`,
    "PUT",
    body,
    { prefix: "fb-team-enrichment-put" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}

/**
 * DELETE team enrichment override
 *
 * Proxies to backend DELETE /dashboard/admin/team/{id}/enrichment
 * Removes all override values, reverting to Wikidata source.
 */
export async function DELETE(
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

  const { data, status, requestId } = await proxyFetchMutation(
    `/dashboard/admin/team/${id}/enrichment`,
    "DELETE",
    undefined,
    { prefix: "fb-team-enrichment-del" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
