import { redirect } from "next/navigation";

interface PageProps {
  params: Promise<{ checkId: string }>;
}

/**
 * Deep-link redirect for data quality check details
 *
 * /data-quality/123 → redirects to /data-quality?id=123
 *
 * This allows shareable URLs while keeping the canonical URL pattern
 * consistent with the query parameter approach.
 */
export default async function DataQualityCheckDeepLinkPage({ params }: PageProps) {
  const { checkId } = await params;
  redirect(`/data-quality?id=${checkId}`);
}
