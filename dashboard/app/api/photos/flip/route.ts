import { NextRequest, NextResponse } from "next/server";
import { proxyFetchMutation, standardHeaders } from "@/lib/api/proxy-utils";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { player_external_id } = body;

  if (!player_external_id) {
    return NextResponse.json(
      { error: "player_external_id required" },
      { status: 400 }
    );
  }

  const { data, status, requestId } = await proxyFetchMutation(
    `/dashboard/photos/flip/${player_external_id}`,
    "POST",
    {},
    { prefix: "photos-flip", timeoutMs: 30000 }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
