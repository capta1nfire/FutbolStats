import { redirect } from "next/navigation";

interface PageProps {
  params: Promise<{ reportId: string }>;
}

/**
 * Deep-link redirect for analytics report details
 *
 * /analytics/123 → redirects to /analytics?id=123
 *
 * This allows shareable URLs while keeping the canonical URL pattern
 * consistent with the query parameter approach.
 */
export default async function AnalyticsReportDeepLinkPage({ params }: PageProps) {
  const { reportId } = await params;
  redirect(`/analytics?id=${reportId}`);
}
