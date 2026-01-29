import { NextRequest, NextResponse } from "next/server";
import { randomUUID } from "crypto";

/**
 * Upload route for team logos (multipart/form-data)
 *
 * Proxies file uploads to backend POST /dashboard/logos/teams/{team_id}/upload
 */

const BACKEND_BASE_URL = process.env.BACKEND_BASE_URL;
const AUTH_HEADER_NAME = process.env.OPS_AUTH_HEADER_NAME || "X-Dashboard-Token";
const AUTH_HEADER_VALUE = process.env.OPS_AUTH_HEADER_VALUE;
const TIMEOUT_MS = 60000; // 60s for uploads

function generateRequestId(): string {
  try {
    return `logos-upload-${randomUUID()}`;
  } catch {
    return `logos-upload-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ teamId: string }> }
) {
  const { teamId } = await params;
  const requestId = generateRequestId();

  if (!BACKEND_BASE_URL) {
    return NextResponse.json(
      { error: "Backend not configured", requestId },
      { status: 503 }
    );
  }

  // Validate teamId is a number
  const teamIdNum = parseInt(teamId, 10);
  if (isNaN(teamIdNum)) {
    return NextResponse.json(
      { error: "Invalid team ID", requestId },
      { status: 400 }
    );
  }

  try {
    // Get the form data from the request
    const formData = await request.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof File)) {
      return NextResponse.json(
        { error: "No file provided", requestId },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ["image/png", "image/webp", "image/svg+xml"];
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json(
        {
          error: "Invalid file type",
          detail: `Allowed types: PNG, WebP, SVG. Got: ${file.type}`,
          requestId,
        },
        { status: 400 }
      );
    }

    // Validate file size (max 5MB)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
      return NextResponse.json(
        {
          error: "File too large",
          detail: `Max size: 5MB. Got: ${(file.size / 1024 / 1024).toFixed(2)}MB`,
          requestId,
        },
        { status: 400 }
      );
    }

    // Create new FormData to send to backend
    const backendFormData = new FormData();
    backendFormData.append("file", file);

    // Build headers (without Content-Type - let fetch set it for multipart)
    const headers: HeadersInit = {
      "x-request-id": requestId,
      Accept: "application/json",
    };

    if (AUTH_HEADER_NAME && AUTH_HEADER_VALUE) {
      headers[AUTH_HEADER_NAME] = AUTH_HEADER_VALUE;
    }

    // Forward to backend
    const url = `${BACKEND_BASE_URL}/dashboard/logos/teams/${teamIdNum}/upload`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: backendFormData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

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

      return NextResponse.json(
        {
          error: "Upload failed",
          detail: data?.detail || `Backend returned ${response.status}`,
          requestId,
        },
        {
          status: response.status,
          headers: {
            "Cache-Control": "no-store",
            "x-request-id": requestId,
          },
        }
      );
    } finally {
      clearTimeout(timeoutId);
    }
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === "AbortError";
    return NextResponse.json(
      {
        error: isTimeout ? "Upload timeout" : "Upload failed",
        detail: error instanceof Error ? error.message : "Unknown error",
        requestId,
      },
      { status: isTimeout ? 504 : 500 }
    );
  }
}
