import { redirect } from "next/navigation";

interface PageProps {
  params: Promise<{ predictionId: string }>;
}

/**
 * Deep-link redirect for prediction details
 *
 * /predictions/123 → redirects to /predictions?id=123
 *
 * This allows shareable URLs while keeping the canonical URL pattern
 * consistent with the query parameter approach.
 */
export default async function PredictionDeepLinkPage({ params }: PageProps) {
  const { predictionId } = await params;
  redirect(`/predictions?id=${predictionId}`);
}
