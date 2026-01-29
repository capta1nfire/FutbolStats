import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Catch-all proxy route for /dashboard/logos/* endpoints
 *
 * Proxies all requests to the backend logo generation API.
 * Handles auth via X-Dashboard-Token header.
 *
 * Supported endpoints:
 * - GET  /logos/leagues - List leagues for generation
 * - POST /logos/generate/league/{id} - Start batch job
 * - GET  /logos/batch/{id} - Get batch status
 * - POST /logos/batch/{id}/pause - Pause batch
 * - POST /logos/batch/{id}/resume - Resume batch
 * - POST /logos/batch/{id}/cancel - Cancel batch
 * - POST /logos/generate/batch/{id}/process - Process next batch
 * - GET  /logos/review/league/{id} - Get teams for review
 * - POST /logos/review/team/{id} - Review single team
 * - POST /logos/review/league/{id}/approve - Bulk approve/reject
 * - GET  /logos/teams/{id}/status - Get team status
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "30000", 10); // 30s for IA calls

function generateRequestId(prefix: string = "logos"): string {
  try {
    return `${prefix}-${randomUUID()}`;
  } catch {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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
  status: number,
  detail?: string
): NextResponse {
  return NextResponse.json(
    { error, detail, requestId },
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
 * GET /api/logos/[...path]
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const requestId = generateRequestId("logos-get");

  if (!BACKEND_BASE_URL) {
    return errorResponse("Backend not configured", requestId, 503);
  }

  const pathStr = path.join("/");
  const searchParams = request.nextUrl.search;
  const url = `${BACKEND_BASE_URL}/dashboard/logos/${pathStr}${searchParams}`;

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
          "Cache-Control": "no-store",
          "x-request-id": requestId,
        },
      });
    }

    // Try to get error details from response
    const errorData = await response.json().catch(() => null);
    const detail = errorData?.detail || `Backend returned ${response.status}`;

    return errorResponse(
      `Request failed`,
      requestId,
      response.status,
      detail
    );
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return errorResponse(
      isTimeout ? "Backend timeout" : "Backend unreachable",
      requestId,
      504
    );
  }
}

/**
 * POST /api/logos/[...path]
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const requestId = generateRequestId("logos-post");

  if (!BACKEND_BASE_URL) {
    return errorResponse("Backend not configured", requestId, 503);
  }

  // Parse request body (may be empty for actions like pause/resume)
  let body: unknown = {};
  try {
    const text = await request.text();
    if (text) {
      body = JSON.parse(text);
    }
  } catch {
    return errorResponse("Invalid JSON body", requestId, 400);
  }

  const pathStr = path.join("/");
  const url = `${BACKEND_BASE_URL}/dashboard/logos/${pathStr}`;

  const fetchOptions: RequestInit = {
    method: "POST",
    headers: {
      ...buildHeaders(requestId),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  };

  try {
    const response = await fetchWithTimeout(url, fetchOptions, TIMEOUT_MS);

    const data = await response.json().catch(() => null);

    if (response.ok) {
      return NextResponse.json(data || { success: true }, {
        status: response.status,
        headers: {
          "Cache-Control": "no-store",
          "x-request-id": requestId,
        },
      });
    }

    // Forward backend error details
    const detail = data?.detail || `Backend returned ${response.status}`;
    return NextResponse.json(
      { error: "Request failed", detail, requestId },
      {
        status: response.status,
        headers: {
          "Cache-Control": "no-store",
          "x-request-id": requestId,
        },
      }
    );
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return errorResponse(
      isTimeout ? "Backend timeout" : "Backend unreachable",
      requestId,
      504
    );
  }
}
