import { NextRequest, NextResponse } from "next/server";
import { proxyFetchMutation, standardHeaders } from "@/lib/api/proxy-utils";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { player_external_id, image_url, image_base64 } = body;

  if (!player_external_id || (!image_url && !image_base64)) {
    return NextResponse.json(
      { error: "player_external_id and (image_url or image_base64) required" },
      { status: 400 }
    );
  }

  const payload: Record<string, unknown> = { player_external_id };
  if (image_base64) payload.image_base64 = image_base64;
  if (image_url) payload.image_url = image_url;

  const { data, status, requestId } = await proxyFetchMutation(
    "/dashboard/photos/create-candidate",
    "POST",
    payload,
    { prefix: "photos-add-candidate", timeoutMs: 30000 }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
