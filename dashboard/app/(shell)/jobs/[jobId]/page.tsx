import { redirect } from "next/navigation";

interface PageProps {
  params: Promise<{ jobId: string }>;
}

/**
 * Deep-link redirect for job details
 *
 * /jobs/123 → redirects to /jobs?id=123
 *
 * This allows shareable URLs while keeping the canonical URL pattern
 * consistent with the query parameter approach.
 */
export default async function JobDeepLinkPage({ params }: PageProps) {
  const { jobId } = await params;
  redirect(`/jobs?id=${jobId}`);
}
