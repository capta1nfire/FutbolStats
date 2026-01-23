import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/audit_logs.json
 *
 * Enterprise-safe architecture:
 * - Same-origin proxy avoids CORS issues
 * - Server-side only - secrets never exposed to client
 * - Configurable auth header via env vars (X-Dashboard-Token)
 * - Timeout + single retry on network/5xx errors
 * - Passes through query params for filtering/pagination
 *
 * Supported filters (multi-value where noted):
 * - type (multi): job_live_tick, job_fastpath, job_global_sync, etc.
 * - severity (multi): info, warning, error
 * - actor_kind (multi): system, user
 * - q: search term
 * - range: 1h, 24h, 7d, 30d
 * - page, limit (max 100)
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME =
  process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

/**
 * Generate a cryptographically secure request ID for tracing
 */
function generateRequestId(): string {
  try {
    return `audit-logs-${randomUUID()}`;
  } catch {
    return `audit-logs-${Date.now()}-${process.pid}`;
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

/**
 * Check if error is retryable (network/timeout/5xx)
 */
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

  // Check if backend URL is configured
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

  // Build query params from request
  const searchParams = request.nextUrl.searchParams;
  const backendParams = new URLSearchParams();

  // Pass through supported params (multi-value for type, severity, actor_kind)
  const types = searchParams.getAll("type");
  for (const t of types) {
    backendParams.append("type", t);
  }

  const severities = searchParams.getAll("severity");
  for (const s of severities) {
    backendParams.append("severity", s);
  }

  const actorKinds = searchParams.getAll("actor_kind");
  for (const a of actorKinds) {
    backendParams.append("actor_kind", a);
  }

  const q = searchParams.get("q");
  if (q) backendParams.set("q", q);

  const range = searchParams.get("range");
  if (range) backendParams.set("range", range);

  const page = searchParams.get("page");
  if (page) backendParams.set("page", page);

  const limit = searchParams.get("limit");
  if (limit) backendParams.set("limit", limit);

  const queryString = backendParams.toString();
  const url = `${BACKEND_BASE_URL}/dashboard/audit_logs.json${queryString ? `?${queryString}` : ""}`;

  // Build headers
  const headers: HeadersInit = {
    "x-request-id": requestId,
    Accept: "application/json",
  };

  // Add auth header if configured
  if (AUTH_HEADER_NAME && AUTH_HEADER_VALUE) {
    headers[AUTH_HEADER_NAME] = AUTH_HEADER_VALUE;
  }

  const fetchOptions: RequestInit = {
    method: "GET",
    headers,
    cache: "no-store",
  };

  // Attempt fetch with 1 retry on retryable errors
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

      // 4xx errors - don't retry, don't leak backend details
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

      // 5xx - may retry
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

  // Fallback (shouldn't reach here)
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
