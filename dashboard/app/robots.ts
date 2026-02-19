import type { MetadataRoute } from "next";

/**
 * Robots.txt — block all crawlers.
 * ops.futbolstat.com is an internal admin surface, not consumer-facing.
 */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      disallow: "/",
    },
  };
}
