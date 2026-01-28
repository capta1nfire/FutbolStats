import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/settings/ia-features/playground
 *
 * POST: Generate narrative with custom parameters (calls LLM, incurs costs)
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "30000", 10); // Longer timeout for LLM

function generateRequestId(): string {
  try {
    return `playground-${randomUUID()}`;
  } catch {
    return `playground-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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
    "Content-Type": "application/json",
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
  extra?: Record<string, unknown>
): NextResponse {
  return NextResponse.json(
    { error, requestId, ...extra },
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
 * POST /api/settings/ia-features/playground
 *
 * Generate narrative with custom parameters.
 * Body: { match_id, temperature?, max_tokens?, model? }
 */
export async function POST(request: NextRequest) {
  const requestId = generateRequestId();

  if (!BACKEND_BASE_URL) {
    return errorResponse("Backend not configured", requestId, 503);
  }

  // Parse request body
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return errorResponse("Invalid JSON body", requestId, 400);
  }

  const url = `${BACKEND_BASE_URL}/dashboard/settings/ia-features/playground`;
  const fetchOptions: RequestInit = {
    method: "POST",
    headers: buildHeaders(requestId),
    body: JSON.stringify(body),
    cache: "no-store",
  };

  try {
    const response = await fetchWithTimeout(url, fetchOptions, TIMEOUT_MS);

    const data = await response.json().catch(() => null);

    if (response.ok) {
      return NextResponse.json(data, {
        status: 200,
        headers: {
          "Cache-Control": "no-store",
          "x-request-id": requestId,
        },
      });
    }

    // Handle rate limit specially
    if (response.status === 429) {
      return NextResponse.json(
        {
          error: data?.error || "Rate limit exceeded",
          requestId,
          rate_limit: data?.rate_limit,
        },
        {
          status: 429,
          headers: {
            "Cache-Control": "no-store",
            "x-request-id": requestId,
          },
        }
      );
    }

    // Forward other errors
    const errorMessage = data?.detail || data?.error || `Backend returned ${response.status}`;
    return errorResponse(errorMessage, requestId, response.status);
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return errorResponse(
      isTimeout ? "LLM generation timed out" : "Backend unreachable",
      requestId,
      504
    );
  }
}
