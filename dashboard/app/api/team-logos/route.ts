import { NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/team_logos.json
 *
 * Returns a map of team_name -> logo_url for rendering shields.
 * Uses same auth pattern as ops.json proxy.
 *
 * Keying: Currently uses team names as keys. Risk of collisions for
 * common names like "United". Future: add provider_team_id keying.
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = 10000; // 10s - larger payload

/**
 * Generate request ID for tracing
 */
function generateRequestId(): string {
  try {
    return `logos-${randomUUID()}`;
  } catch {
    return `logos-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

/**
 * Fetch with timeout using AbortController
 */
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

export async function GET() {
  const requestId = generateRequestId();

  // Validate configuration
  if (!BACKEND_BASE_URL || !AUTH_HEADER_VALUE) {
    console.error(`[team-logos] [${requestId}] Missing configuration`);
    return NextResponse.json(
      { error: "Service unavailable", teams: {} },
      {
        status: 503,
        headers: { "x-request-id": requestId },
      }
    );
  }

  const url = `${BACKEND_BASE_URL}/dashboard/team_logos.json`;

  try {
    const response = await fetchWithTimeout(
      url,
      {
        method: "GET",
        headers: {
          [AUTH_HEADER_NAME]: AUTH_HEADER_VALUE,
          Accept: "application/json",
        },
        // Use Next.js cache with revalidation (logos don't change often)
        next: { revalidate: 3600 }, // 1 hour
      },
      TIMEOUT_MS
    );

    // Handle auth errors - don't cache, don't leak details
    if (response.status === 401 || response.status === 403) {
      console.error(`[team-logos] [${requestId}] Auth failed: ${response.status}`);
      return NextResponse.json(
        { error: "Unauthorized", teams: {} },
        {
          status: response.status,
          headers: {
            "x-request-id": requestId,
            "Cache-Control": "no-store",
          },
        }
      );
    }

    // Handle other errors - don't cache 5xx
    if (!response.ok) {
      console.error(`[team-logos] [${requestId}] Backend returned ${response.status}`);
      return NextResponse.json(
        { error: "Backend error", teams: {} },
        {
          status: response.status,
          headers: {
            "x-request-id": requestId,
            "Cache-Control": response.status >= 500 ? "no-store" : "public, max-age=60",
          },
        }
      );
    }

    const data = await response.json();

    // Pass through the response with cache headers
    return NextResponse.json(data, {
      headers: {
        "x-request-id": requestId,
        "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=7200",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error(`[team-logos] [${requestId}] Fetch failed: ${message}`);

    return NextResponse.json(
      { error: "Service unavailable", teams: {} },
      {
        status: 503,
        headers: {
          "x-request-id": requestId,
          "Cache-Control": "no-store",
        },
      }
    );
  }
}
