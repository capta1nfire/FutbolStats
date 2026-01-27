import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/league-groups.json
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/admin/league-groups.json",
    undefined,
    { prefix: "admin-league-groups" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
