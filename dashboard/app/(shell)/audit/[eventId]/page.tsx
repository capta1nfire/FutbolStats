import { redirect } from "next/navigation";

interface PageProps {
  params: Promise<{ eventId: string }>;
}

/**
 * Deep-link redirect for audit event details
 *
 * /audit/123 → redirects to /audit?id=123
 *
 * This allows shareable URLs while keeping the canonical URL pattern
 * consistent with the query parameter approach.
 */
export default async function AuditEventDeepLinkPage({ params }: PageProps) {
  const { eventId } = await params;
  redirect(`/audit?id=${eventId}`);
}
