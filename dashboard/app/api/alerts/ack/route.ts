import { NextRequest, NextResponse } from "next/server";
import { generateRequestId, buildHeaders, fetchWithTimeout, isRetryableError, standardHeaders, getBackendBaseUrl } from "@/lib/api/proxy-utils";

/**
 * Proxy route handler for POST /dashboard/ops/alerts/ack
 *
 * Body options:
 * - { ids: number[] } - Mark specific alerts as read
 * - { ack_all: true } - Mark all alerts as read
 *
 * Returns: { status: "ok", updated: number }
 */
export async function POST(request: NextRequest) {
  const requestId = generateRequestId("alerts-ack");
  const baseUrl = getBackendBaseUrl();

  if (!baseUrl) {
    return NextResponse.json(
      { error: "Backend not configured", requestId },
      { status: 503, headers: standardHeaders(requestId) }
    );
  }

  // Parse and validate body
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body", requestId },
      { status: 400, headers: standardHeaders(requestId) }
    );
  }

  // Validate body shape
  if (!isValidAckBody(body)) {
    return NextResponse.json(
      { error: "Invalid body: must have 'ids' array or 'ack_all: true'", requestId },
      { status: 400, headers: standardHeaders(requestId) }
    );
  }

  const url = `${baseUrl}/dashboard/ops/alerts/ack`;
  const baseHeaders = buildHeaders(requestId);

  const fetchOptions: RequestInit = {
    method: "POST",
    headers: {
      ...baseHeaders,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  };

  // Attempt fetch with 1 retry on retryable errors
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetchWithTimeout(url, fetchOptions, 8000);

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data, {
          status: 200,
          headers: standardHeaders(requestId),
        });
      }

      // 4xx errors - don't retry
      if (response.status >= 400 && response.status < 500) {
        return NextResponse.json(
          { error: `Backend returned ${response.status}`, requestId },
          { status: response.status, headers: standardHeaders(requestId) }
        );
      }

      // 5xx - may retry
      if (attempt === 0 && isRetryableError(null, response)) {
        continue;
      }

      return NextResponse.json(
        { error: `Backend returned ${response.status}`, requestId },
        { status: 502, headers: standardHeaders(requestId) }
      );
    } catch (error) {
      if (attempt === 0 && isRetryableError(error)) {
        continue;
      }

      const isTimeout = error instanceof Error && error.name === "AbortError";
      return NextResponse.json(
        { error: isTimeout ? "Backend timeout" : "Backend unreachable", requestId },
        { status: 504, headers: standardHeaders(requestId) }
      );
    }
  }

  return NextResponse.json(
    { error: "Failed after retries", requestId },
    { status: 502, headers: standardHeaders(requestId) }
  );
}

/**
 * Validate ack body shape
 */
function isValidAckBody(body: unknown): body is { ids: number[] } | { ack_all: true } {
  if (typeof body !== "object" || body === null) return false;

  const obj = body as Record<string, unknown>;

  // Option 1: { ids: number[] }
  if (Array.isArray(obj.ids)) {
    return obj.ids.every((id) => typeof id === "number" && id > 0);
  }

  // Option 2: { ack_all: true }
  if (obj.ack_all === true) {
    return true;
  }

  return false;
}
