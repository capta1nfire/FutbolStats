import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/leagues/countries.json
 * Returns list of countries with active leagues
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/football/leagues/countries.json",
    undefined,
    { prefix: "fb-countries" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
