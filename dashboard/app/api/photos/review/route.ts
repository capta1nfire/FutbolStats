import { NextRequest, NextResponse } from "next/server";
import {
  proxyFetch,
  proxyFetchMutation,
  standardHeaders,
} from "@/lib/api/proxy-utils";

export async function GET(request: NextRequest) {
  const status = request.nextUrl.searchParams.get("status") || "pending_review";
  const params = new URLSearchParams({ status });
  const teamId = request.nextUrl.searchParams.get("team_id");
  if (teamId) params.set("team_id", teamId);
  const { data, status: httpStatus, requestId } = await proxyFetch(
    "/dashboard/photos/candidates.json",
    params,
    { prefix: "photos", timeoutMs: 10000 }
  );

  return NextResponse.json(data, {
    status: httpStatus,
    headers: standardHeaders(requestId),
  });
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { id, action, manual_crop, face_detect } = body;

  if (!id || !action) {
    return NextResponse.json({ error: "id and action required" }, { status: 400 });
  }

  const payload: Record<string, unknown> = { action };
  if (manual_crop) payload.manual_crop = manual_crop;
  if (face_detect) payload.face_detect = face_detect;

  const { data, status, requestId } = await proxyFetchMutation(
    `/dashboard/photos/review/${id}`,
    "POST",
    payload,
    { prefix: "photos-review", timeoutMs: 30000 }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
