import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Proxy route handler for PATCH /dashboard/incidents/{id}
 *
 * Forwards acknowledge/resolve actions to backend.
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME =
  process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = parseInt(process.env.OPS_TIMEOUT_MS || "8000", 10);

function generateRequestId(): string {
  try {
    return `incident-patch-${randomUUID()}`;
  } catch {
    return `incident-patch-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const requestId = generateRequestId();
  const { id } = await params;

  if (!BACKEND_BASE_URL) {
    return NextResponse.json(
      { error: "Backend not configured", requestId },
      { status: 503, headers: { "x-request-id": requestId } }
    );
  }

  const url = `${BACKEND_BASE_URL}/dashboard/incidents/${id}`;

  const headers: HeadersInit = {
    "x-request-id": requestId,
    "Content-Type": "application/json",
    Accept: "application/json",
  };

  if (AUTH_HEADER_NAME && AUTH_HEADER_VALUE) {
    headers[AUTH_HEADER_NAME] = AUTH_HEADER_VALUE;
  }

  try {
    const body = await request.text();
    const response = await fetchWithTimeout(
      url,
      { method: "PATCH", headers, body, cache: "no-store" },
      TIMEOUT_MS
    );

    const data = await response.json();
    return NextResponse.json(data, {
      status: response.status,
      headers: { "x-request-id": requestId },
    });
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return NextResponse.json(
      { error: isTimeout ? "Backend timeout" : "Backend unreachable", requestId },
      { status: 504, headers: { "x-request-id": requestId } }
    );
  }
}
