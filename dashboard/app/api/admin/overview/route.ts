import { NextResponse } from "next/server";
import { proxyFetch, standardHeaders } from "@/lib/api/proxy-utils";

/**
 * Proxy for /dashboard/admin/overview.json
 */
export async function GET() {
  const { data, status, requestId } = await proxyFetch(
    "/dashboard/admin/overview.json",
    undefined,
    { prefix: "admin-overview" }
  );

  return NextResponse.json(data, {
    status,
    headers: standardHeaders(requestId),
  });
}
