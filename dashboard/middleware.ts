import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Edge middleware for ops.futbolstat.com
 *
 * Security headers for internal admin surface:
 * - X-Robots-Tag: noindex — belt-and-suspenders with robots.txt
 * - X-Frame-Options: DENY — prevent iframe embedding
 * - X-Content-Type-Options: nosniff
 */
export function middleware(request: NextRequest) {
  const response = NextResponse.next();

  response.headers.set("X-Robots-Tag", "noindex, nofollow");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("X-Content-Type-Options", "nosniff");

  return response;
}

export const config = {
  // Apply to all routes except static assets and Next.js internals
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
