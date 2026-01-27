import { NextRequest, NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/leagues/country/{country}.json
 * Returns competitions for a specific country
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ country: string }> }
) {
  const { country } = await params;

  if (!country) {
    return NextResponse.json(
      { error: "Country parameter required" },
      { status: 400 }
    );
  }

  const { data, status, requestId } = await proxyFetch(
    `/dashboard/football/leagues/country/${encodeURIComponent(country)}.json`,
    undefined,
    { prefix: "fb-country" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
