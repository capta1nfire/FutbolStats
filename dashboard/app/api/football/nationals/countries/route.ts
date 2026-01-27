import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/nationals/countries.json
 * Returns list of countries with national teams
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/football/nationals/countries.json",
    undefined,
    { prefix: "fb-nationals-countries" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
