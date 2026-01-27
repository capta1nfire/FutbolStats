import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/nav.json
 * Returns navigation categories for Football section
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/football/nav.json",
    undefined,
    { prefix: "fb-nav" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
