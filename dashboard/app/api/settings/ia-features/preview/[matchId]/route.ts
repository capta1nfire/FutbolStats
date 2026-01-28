import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/settings/ia-features/preview/{match_id}.json
 *
 * GET: Preview LLM payload for a specific match (without calling the LLM)
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "10000", 10); // Slightly longer for preview

function generateRequestId(): string {
  try {
    return `preview-${randomUUID()}`;
  } catch {
    return `preview-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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

interface RouteParams {
  params: Promise<{ matchId: string }>;
}

/**
 * GET /api/settings/ia-features/preview/[matchId]
 *
 * Preview LLM payload for a specific match.
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
  const requestId = generateRequestId();
  const { matchId } = await params;

  // Validate matchId is a number
  const matchIdNum = parseInt(matchId, 10);
  if (isNaN(matchIdNum) || matchIdNum <= 0) {
    return errorResponse("Invalid match ID", requestId, 400);
  }

  if (!BACKEND_BASE_URL) {
    return errorResponse("Backend not configured", requestId, 503);
  }

  const url = `${BACKEND_BASE_URL}/dashboard/settings/ia-features/preview/${matchIdNum}.json`;
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
          "Cache-Control": "no-store", // Always fresh for preview
          "x-request-id": requestId,
        },
      });
    }

    // Forward 404 for match not found
    if (response.status === 404) {
      return errorResponse("Match not found", requestId, 404);
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
