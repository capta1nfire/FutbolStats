import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /standings/{league_id}
 *
 * Proxies to backend API with authentication.
 * Uses X-API-Key header (protected endpoint).
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const API_KEY = process.env.FUTBOLSTATS_API_KEY;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

function generateRequestId(): string {
  try {
    return `standings-${randomUUID()}`;
  } catch {
    return `standings-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ leagueId: string }> }
) {
  const { leagueId } = await params;
  const requestId = generateRequestId();

  if (!BACKEND_BASE_URL) {
    return NextResponse.json(
      { error: "Backend not configured", requestId },
      { status: 503, headers: { "Cache-Control": "no-store", "x-request-id": requestId } }
    );
  }

  if (!API_KEY) {
    return NextResponse.json(
      { error: "API key not configured", requestId },
      { status: 503, headers: { "Cache-Control": "no-store", "x-request-id": requestId } }
    );
  }

  // Forward query params (e.g., ?season=2024)
  const searchParams = request.nextUrl.searchParams;
  const queryString = searchParams.toString();
  const url = `${BACKEND_BASE_URL}/standings/${leagueId}${queryString ? `?${queryString}` : ""}`;

  const headers: HeadersInit = {
    "x-request-id": requestId,
    "X-API-Key": API_KEY,
    Accept: "application/json",
  };

  const fetchOptions: RequestInit = {
    method: "GET",
    headers,
    cache: "no-store",
  };

  try {
    const response = await fetchWithTimeout(url, fetchOptions, TIMEOUT_MS);

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data, {
        status: 200,
        headers: {
          // Cache for 5 minutes (standings don't change frequently)
          "Cache-Control": "public, max-age=300, stale-while-revalidate=60",
          "x-request-id": requestId,
        },
      });
    }

    if (response.status === 404) {
      return NextResponse.json(
        { error: "Standings not found", requestId },
        { status: 404, headers: { "Cache-Control": "no-store", "x-request-id": requestId } }
      );
    }

    return NextResponse.json(
      { error: `Backend returned ${response.status}`, requestId },
      { status: response.status >= 500 ? 502 : response.status, headers: { "Cache-Control": "no-store", "x-request-id": requestId } }
    );
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return NextResponse.json(
      { error: isTimeout ? "Backend timeout" : "Backend unreachable", requestId },
      { status: 504, headers: { "Cache-Control": "no-store", "x-request-id": requestId } }
    );
  }
}
