import { NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for /dashboard/ops.json
 *
 * Enterprise-safe architecture:
 * - Same-origin proxy avoids CORS issues
 * - Server-side only - secrets never exposed to client
 * - Configurable auth header via env vars
 * - Timeout + single retry on network/5xx errors
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

/**
 * Generate a cryptographically secure request ID for tracing
 */
function generateRequestId(): string {
  // Prefer Node's crypto.randomUUID (stable in server runtime).
  // Fall back to a timestamp-based ID if unavailable for any reason.
  try {
    return `ops-${randomUUID()}`;
  } catch {
    return `ops-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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
  // Network errors or timeouts
  if (error instanceof Error) {
    if (error.name === "AbortError") return true;
    if (error.message.includes("fetch")) return true;
  }
  // 5xx server errors
  if (response && response.status >= 500) return true;
  return false;
}

/**
 * Remove PII fields from ops response before sending to client.
 * This ensures sensitive data from backend never reaches the browser.
 *
 * Response structure: { data: { budget: { account_email: "..." } } }
 */
function scrubPii(response: unknown): unknown {
  if (typeof response !== "object" || response === null) {
    return response;
  }

  const result = { ...response } as Record<string, unknown>;

  // Handle nested data.budget.account_email
  if (typeof result.data === "object" && result.data !== null) {
    const data = { ...result.data } as Record<string, unknown>;

    if (typeof data.budget === "object" && data.budget !== null) {
      const budget = { ...data.budget } as Record<string, unknown>;
      delete budget.account_email;
      data.budget = budget;
    }

    result.data = data;
  }

  // Also handle root-level budget (in case structure changes)
  if (
    typeof result.budget === "object" &&
    result.budget !== null &&
    "account_email" in result.budget
  ) {
    const budget = { ...result.budget } as Record<string, unknown>;
    delete budget.account_email;
    result.budget = budget;
  }

  return result;
}

export async function GET() {
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

  const url = `${BACKEND_BASE_URL}/dashboard/ops.json`;

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
    // Disable Next.js cache for this dynamic data
    cache: "no-store",
  };

  // Attempt fetch with 1 retry on retryable errors
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetchWithTimeout(url, fetchOptions, TIMEOUT_MS);

      // Success - scrub PII before responding
      if (response.ok) {
        const data = await response.json();
        const safeData = scrubPii(data);
        return NextResponse.json(safeData, {
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

      // Final attempt failed
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
      // Retry on retryable errors (first attempt only)
      if (attempt === 0 && isRetryableError(error)) {
        continue;
      }

      // Final attempt - return error (no internal details exposed)
      const isTimeout =
        error instanceof Error && error.name === "AbortError";

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
    {
      error: "Failed after retries",
      requestId,
    },
    {
      status: 502,
      headers: {
        "Cache-Control": "no-store",
        "x-request-id": requestId,
      },
    }
  );
}
