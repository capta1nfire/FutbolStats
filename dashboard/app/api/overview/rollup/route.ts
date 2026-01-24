import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy route handler for /dashboard/overview/rollup.json
 *
 * Returns aggregated overview metrics for the dashboard.
 * Hardened with timeout, retry, and no internal detail leakage.
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/overview/rollup.json",
    undefined,
    { prefix: "rollup" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
