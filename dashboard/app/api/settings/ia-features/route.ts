import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/settings/ia-features.json
 *
 * Supports:
 * - GET: Fetch current IA Features config
 * - PATCH: Update IA Features config (narratives_enabled, primary_model, temperature, max_tokens)
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

function generateRequestId(prefix: string = "ia-features"): string {
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

function isRetryableError(error: unknown, response?: Response): boolean {
  if (error instanceof Error) {
    if (error.name === "AbortError") return true;
    if (error.message.includes("fetch")) return true;
  }
  if (response && response.status >= 500) return true;
  return false;
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
 * GET /api/settings/ia-features
 *
 * Fetch current IA Features configuration.
 */
export async function GET() {
  const requestId = generateRequestId("ia-get");

  if (!BACKEND_BASE_URL) {
    return errorResponse("Backend not configured", requestId, 503);
  }

  const url = `${BACKEND_BASE_URL}/dashboard/settings/ia-features.json`;
  const fetchOptions: RequestInit = {
    method: "GET",
    headers: buildHeaders(requestId),
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
        return errorResponse(`Backend returned ${response.status}`, requestId, response.status);
      }

      if (attempt === 0 && isRetryableError(null, response)) {
        continue;
      }

      return errorResponse(`Backend returned ${response.status}`, requestId, 502);
    } catch (error) {
      if (attempt === 0 && isRetryableError(error)) {
        continue;
      }

      const isTimeout = error instanceof Error && error.name === "AbortError";
      return errorResponse(
        isTimeout ? "Backend timeout" : "Backend unreachable",
        requestId,
        504
      );
    }
  }

  return errorResponse("Failed after retries", requestId, 502);
}

/**
 * PATCH /api/settings/ia-features
 *
 * Update IA Features configuration.
 * Body: { narratives_enabled?, primary_model?, temperature?, max_tokens? }
 */
export async function PATCH(request: NextRequest) {
  const requestId = generateRequestId("ia-patch");

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

  const url = `${BACKEND_BASE_URL}/dashboard/settings/ia-features.json`;
  const fetchOptions: RequestInit = {
    method: "PATCH",
    headers: {
      ...buildHeaders(requestId),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  };

  // No retry for mutations (not idempotent)
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

    // Forward backend error details if available
    const errorMessage = data?.detail || `Backend returned ${response.status}`;
    return NextResponse.json(
      { error: errorMessage, requestId },
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
