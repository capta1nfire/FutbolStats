import { NextResponse, NextRequest } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/nationals/country/{country}.json
 * Returns national teams detail for a country
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ country: string }> }
) {
  const { country } = await params;

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/football/nationals/country/${encodeURIComponent(country)}.json`,
    undefined,
    { prefix: "fb-nationals-country" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
