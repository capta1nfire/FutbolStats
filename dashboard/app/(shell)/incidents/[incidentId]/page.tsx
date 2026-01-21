import { redirect } from "next/navigation";

interface PageProps {
  params: Promise<{ incidentId: string }>;
}

/**
 * Deep-link redirect for incident details
 *
 * /incidents/123 → redirects to /incidents?id=123
 *
 * This allows shareable URLs while keeping the canonical URL pattern
 * consistent with the query parameter approach.
 */
export default async function IncidentDeepLinkPage({ params }: PageProps) {
  const { incidentId } = await params;
  redirect(`/incidents?id=${incidentId}`);
}
