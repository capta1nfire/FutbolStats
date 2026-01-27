import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/football/tournaments.json
 * Returns list of tournaments and cups
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/football/tournaments.json",
    undefined,
    { prefix: "fb-tournaments" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
