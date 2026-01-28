import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/settings/ia-features/call-history.json
 *
 * GET: Fetch recent LLM narrative generation calls
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

function generateRequestId(): string {
  try {
    return `call-hist-${randomUUID()}`;
  } catch {
    return `call-hist-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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

function buildHeaders(requestId: string): HeadersInit {
  const headers: HeadersInit = {
    "x-request-id": requestId,
    Accept: "application/json",
  };

  if (AUTH_HEADER_NAME && AUTH_HEADER_VALUE) {
    headers[AUTH_HEADER_NAME] = AUTH_HEADER_VALUE;
  }

  return headers;
}

function errorResponse(
  error: string,
  requestId: string,
  status: number
): NextResponse {
  return NextResponse.json(
    { error, requestId },
    {
      status,
      headers: {
        "Cache-Control": "no-store",
        "x-request-id": requestId,
      },
    }
  );
}

/**
 * GET /api/settings/ia-features/call-history
 *
 * Fetch recent LLM narrative generation calls.
 * Query params:
 * - limit: Max items to return (default 20, max 100)
 */
export async function GET(request: NextRequest) {
  const requestId = generateRequestId();

  if (!BACKEND_BASE_URL) {
    return errorResponse("Backend not configured", requestId, 503);
  }

  // Parse query params
  const searchParams = request.nextUrl.searchParams;
  const limit = Math.min(Math.max(parseInt(searchParams.get("limit") || "20", 10), 1), 100);

  const url = `${BACKEND_BASE_URL}/dashboard/settings/ia-features/call-history.json?limit=${limit}`;
  const fetchOptions: RequestInit = {
    method: "GET",
    headers: buildHeaders(requestId),
    cache: "no-store",
  };

  try {
    const response = await fetchWithTimeout(url, fetchOptions, TIMEOUT_MS);

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data, {
        status: 200,
        headers: {
          "Cache-Control": "no-store", // Always fresh for history
          "x-request-id": requestId,
        },
      });
    }

    return errorResponse(`Backend returned ${response.status}`, requestId, response.status);
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return errorResponse(
      isTimeout ? "Backend timeout" : "Backend unreachable",
      requestId,
      504
    );
  }
}
