/**
 * Shared utilities for API proxy routes
 *
 * Enterprise-safe architecture:
 * - Timeout + single retry on network/5xx errors
 * - x-request-id for tracing
 * - No internal details leaked to client
 */

import { randomUUID } from "crypto";

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const DEFAULT_TIMEOUT_MS = 8000;

/**
 * Generate a cryptographically secure request ID for tracing
 */
export function generateRequestId(prefix: string = "req"): string {
  try {
    return `${prefix}-${randomUUID()}`;
  } catch {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

/**
 * Fetch with timeout using AbortController
 */
export async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
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
export function isRetryableError(error: unknown, response?: Response): boolean {
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
 * Build standard headers for backend requests
 */
export function buildHeaders(requestId: string): HeadersInit {
  const headers: HeadersInit = {
    "x-request-id": requestId,
    Accept: "application/json",
  };

  if (AUTH_HEADER_NAME && AUTH_HEADER_VALUE) {
    headers[AUTH_HEADER_NAME] = AUTH_HEADER_VALUE;
  }

  return headers;
}

/**
 * Get backend base URL (throws if not configured)
 */
export function getBackendBaseUrl(): string | null {
  return BACKEND_BASE_URL || null;
}

/**
 * Standard response headers
 */
export function standardHeaders(requestId: string): Record<string, string> {
  return {
    "Cache-Control": "no-store",
    "x-request-id": requestId,
  };
}

export interface ProxyOptions {
  /** Request ID prefix for tracing */
  prefix?: string;
  /** Timeout in milliseconds */
  timeoutMs?: number;
  /** Optional scrubbing function */
  scrubFn?: (data: unknown) => unknown;
}

/**
 * Generic proxy fetch with hardening
 *
 * - Timeout + 1 retry on 5xx/network errors
 * - x-request-id tracking
 * - No internal details leaked
 */
export async function proxyFetch(
  path: string,
  queryParams?: URLSearchParams,
  options: ProxyOptions = {}
): Promise<{ data: unknown; status: number; requestId: string }> {
  const { prefix = "req", timeoutMs = DEFAULT_TIMEOUT_MS, scrubFn } = options;
  const requestId = generateRequestId(prefix);
  const baseUrl = getBackendBaseUrl();

  if (!baseUrl) {
    return { data: { error: "Backend not configured" }, status: 503, requestId };
  }

  const url = queryParams
    ? `${baseUrl}${path}?${queryParams.toString()}`
    : `${baseUrl}${path}`;

  const headers = buildHeaders(requestId);
  const fetchOptions: RequestInit = {
    method: "GET",
    headers,
    cache: "no-store",
  };

  // Attempt fetch with 1 retry on retryable errors
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetchWithTimeout(url, fetchOptions, timeoutMs);

      // Success
      if (response.ok) {
        const data = await response.json();
        const safeData = scrubFn ? scrubFn(data) : data;
        return { data: safeData, status: 200, requestId };
      }

      // 4xx errors - don't retry, don't leak backend details
      if (response.status >= 400 && response.status < 500) {
        return {
          data: { error: `Backend returned ${response.status}` },
          status: response.status,
          requestId,
        };
      }

      // 5xx - may retry
      if (attempt === 0 && isRetryableError(null, response)) {
        continue;
      }

      // Final attempt failed
      return {
        data: { error: `Backend returned ${response.status}` },
        status: 502,
        requestId,
      };
    } catch (error) {
      // Retry on retryable errors (first attempt only)
      if (attempt === 0 && isRetryableError(error)) {
        continue;
      }

      // Final attempt - return error (no internal details exposed)
      const isTimeout = error instanceof Error && error.name === "AbortError";
      return {
        data: { error: isTimeout ? "Backend timeout" : "Backend unreachable" },
        status: 504,
        requestId,
      };
    }
  }

  // Fallback (shouldn't reach here)
  return { data: { error: "Failed after retries" }, status: 502, requestId };
}

/**
 * Generic proxy fetch for mutations (PATCH/POST/PUT)
 *
 * - NO retry (mutations are not idempotent)
 * - Timeout + abort
 * - x-request-id tracking
 * - Content-Type: application/json
 */
export async function proxyFetchMutation(
  path: string,
  method: "PATCH" | "POST" | "PUT" | "DELETE",
  body: unknown,
  options: ProxyOptions = {}
): Promise<{ data: unknown; status: number; requestId: string }> {
  const { prefix = "mut", timeoutMs = DEFAULT_TIMEOUT_MS } = options;
  const requestId = generateRequestId(prefix);
  const baseUrl = getBackendBaseUrl();

  if (!baseUrl) {
    return { data: { error: "Backend not configured" }, status: 503, requestId };
  }

  const url = `${baseUrl}${path}`;
  const headers = {
    ...buildHeaders(requestId),
    "Content-Type": "application/json",
  };

  const fetchOptions: RequestInit = {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    cache: "no-store",
  };

  try {
    const response = await fetchWithTimeout(url, fetchOptions, timeoutMs);

    const data = await response.json().catch(() => null);

    if (response.ok) {
      return { data, status: response.status, requestId };
    }

    return {
      data: data || { error: `Backend returned ${response.status}` },
      status: response.status,
      requestId,
    };
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return {
      data: { error: isTimeout ? "Backend timeout" : "Backend unreachable" },
      status: 504,
      requestId,
    };
  }
}
