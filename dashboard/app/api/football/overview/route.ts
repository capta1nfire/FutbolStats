import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/overview.json
 * Returns football overview: summary, upcoming, leagues, alerts, titan
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/football/overview.json",
    undefined,
    { prefix: "fb-overview" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
