import { NextRequest, NextResponse } from "next/server";
import { fetchWithTimeout, generateRequestId } from "@/lib/api/proxy-utils";

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;

/**
 * GET /api/photos/preview?id=123
 * Proxies to backend /dashboard/photos/preview/{id} which returns a face-crop PNG.
 */
export async function GET(request: NextRequest) {
  const candidateId = request.nextUrl.searchParams.get("id");
  if (!candidateId) {
    return NextResponse.json({ error: "id required" }, { status: 400 });
  }

  const requestId = generateRequestId("photo-preview");

  // Forward optional manual crop params
  const cropParams = new URLSearchParams();
  for (const key of ["cx", "cy", "cs", "sw", "sh", "clean", "rot"]) {
    const val = request.nextUrl.searchParams.get(key);
    if (val) cropParams.set(key, val);
  }
  const qs = cropParams.toString();

  try {
    const url = `${BACKEND_BASE_URL}/dashboard/photos/preview/${candidateId}${qs ? `?${qs}` : ""}`;
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

    const imageBuffer = await resp.arrayBuffer();
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "public, max-age=3600",
        "x-request-id": requestId,
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Preview generation failed" },
      { status: 502 }
    );
  }
}
