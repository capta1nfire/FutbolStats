import { redirect } from "next/navigation";

interface PageProps {
  params: Promise<{ matchId: string }>;
}

/**
 * Deep-link handler: /matches/[matchId]
 *
 * Redirects to canonical URL: /matches?id=[matchId]
 * Server-side redirect (307)
 */
export default async function MatchDeepLinkPage({ params }: PageProps) {
  const { matchId } = await params;
  redirect(`/matches?id=${matchId}`);
}
