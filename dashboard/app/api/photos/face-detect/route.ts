import { NextRequest, NextResponse } from "next/server";
import { fetchWithTimeout, generateRequestId } from "@/lib/api/proxy-utils";

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;

/**
 * GET /api/photos/face-detect?id=123
 * Returns face detection keypoints (normalized 0-1) for crop tool overlay.
 */
export async function GET(request: NextRequest) {
  const candidateId = request.nextUrl.searchParams.get("id");
  if (!candidateId) {
    return NextResponse.json({ error: "id required" }, { status: 400 });
  }

  const requestId = generateRequestId("face-detect");

  try {
    const url = `${BACKEND_BASE_URL}/dashboard/photos/face-detect/${candidateId}`;
    const resp = await fetchWithTimeout(
      url,
      {
        headers: {
          [AUTH_HEADER_NAME]: AUTH_HEADER_VALUE || "",
          "x-request-id": requestId,
        },
      },
      15000
    );

    if (!resp.ok) {
      return NextResponse.json(
        { error: `Backend error ${resp.status}` },
        { status: resp.status }
      );
    }

    const data = await resp.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "public, max-age=3600",
        "x-request-id": requestId,
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Face detection failed" },
      { status: 502 }
    );
  }
}
