import { NextRequest, NextResponse } from "next/server";

/**
 * Proxy route handler for /dashboard/matches/{matchId}/market-snapshot.json
 *
 * Returns cross-sectional market snapshot: all bookmaker odds for a single match.
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME =
  process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  if (!BACKEND_BASE_URL) {
    return NextResponse.json(
      { error: "Backend not configured" },
      { status: 503, headers: { "Cache-Control": "no-store" } }
    );
  }

  const { matchId } = await params;
  const url = `${BACKEND_BASE_URL}/dashboard/matches/${matchId}/market-snapshot.json`;

  const headers: HeadersInit = { Accept: "application/json" };
  if (AUTH_HEADER_NAME && AUTH_HEADER_VALUE) {
    headers[AUTH_HEADER_NAME] = AUTH_HEADER_VALUE;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: "GET",
      headers,
      signal: controller.signal,
      cache: "no-store",
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend returned ${response.status}` },
        { status: response.status >= 500 ? 502 : response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, {
      status: 200,
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    clearTimeout(timeoutId);
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return NextResponse.json(
      { error: isTimeout ? "Backend timeout" : "Backend unreachable" },
      { status: 504 }
    );
  }
}
