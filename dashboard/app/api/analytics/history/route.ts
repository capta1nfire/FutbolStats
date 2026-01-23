import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/ops/history.json
 *
 * Enterprise-safe architecture:
 * - Same-origin proxy avoids CORS issues
 * - Server-side only - secrets never exposed to client
 * - Configurable auth header via env vars (same as /api/ops)
 * - Timeout + single retry on network/5xx errors
 * - Passes through query params (days)
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME =
  process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

function generateRequestId(): string {
  try {
    return `analytics-history-${randomUUID()}`;
  } catch {
    return `analytics-history-${Date.now()}-${process.pid}`;
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

function isRetryableError(error: unknown, response?: Response): boolean {
  if (error instanceof Error) {
    if (error.name === "AbortError") return true;
    if (error.message.includes("fetch")) return true;
  }
  if (response && response.status >= 500) return true;
  return false;
}

export async function GET(request: NextRequest) {
  const requestId = generateRequestId();

  if (!BACKEND_BASE_URL) {
    return NextResponse.json(
      { error: "Backend not configured", requestId },
      {
        status: 503,
        headers: {
          "Cache-Control": "no-store",
          "x-request-id": requestId,
        },
      }
    );
  }

  // Build query params
  const searchParams = request.nextUrl.searchParams;
  const backendParams = new URLSearchParams();

  const days = searchParams.get("days");
  if (days) backendParams.set("days", days);

  const queryString = backendParams.toString();
  const url = `${BACKEND_BASE_URL}/dashboard/ops/history.json${queryString ? `?${queryString}` : ""}`;

  const headers: HeadersInit = {
    "x-request-id": requestId,
    Accept: "application/json",
  };

  if (AUTH_HEADER_NAME && AUTH_HEADER_VALUE) {
    headers[AUTH_HEADER_NAME] = AUTH_HEADER_VALUE;
  }

  const fetchOptions: RequestInit = {
    method: "GET",
    headers,
    cache: "no-store",
  };

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetchWithTimeout(url, fetchOptions, TIMEOUT_MS);

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data, {
          status: 200,
          headers: {
            "Cache-Control": "no-store",
            "x-request-id": requestId,
          },
        });
      }

      if (response.status >= 400 && response.status < 500) {
        return NextResponse.json(
          { error: `Backend returned ${response.status}`, requestId },
          {
            status: response.status,
            headers: {
              "Cache-Control": "no-store",
              "x-request-id": requestId,
            },
          }
        );
      }

      if (attempt === 0 && isRetryableError(null, response)) {
        continue;
      }

      return NextResponse.json(
        { error: `Backend returned ${response.status}`, requestId },
        {
          status: 502,
          headers: {
            "Cache-Control": "no-store",
            "x-request-id": requestId,
          },
        }
      );
    } catch (error) {
      if (attempt === 0 && isRetryableError(error)) {
        continue;
      }

      const isTimeout = error instanceof Error && error.name === "AbortError";

      return NextResponse.json(
        {
          error: isTimeout ? "Backend timeout" : "Backend unreachable",
          requestId,
        },
        {
          status: 504,
          headers: {
            "Cache-Control": "no-store",
            "x-request-id": requestId,
          },
        }
      );
    }
  }

  return NextResponse.json(
    { error: "Failed after retries", requestId },
    {
      status: 502,
      headers: {
        "Cache-Control": "no-store",
        "x-request-id": requestId,
      },
    }
  );
}
