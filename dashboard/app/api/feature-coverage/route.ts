import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy route handler for /dashboard/feature-coverage.json
 *
 * Returns feature coverage matrix data for the SOTA section.
 * Data changes infrequently - suitable for longer cache times client-side.
 */

export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/feature-coverage.json",
    undefined,
    { prefix: "fc" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
