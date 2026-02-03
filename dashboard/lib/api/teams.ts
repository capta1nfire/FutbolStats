/**
 * Teams API Client
 *
 * Mutations for updating team data (Wikipedia/Wikidata).
 * Uses Next.js API routes for auth handling.
 */

import type { TeamWikiInfo } from "@/lib/types/football";

// =============================================================================
// Types
// =============================================================================

export interface TeamPatchRequest {
  wiki_url?: string | null;
  wikidata_id?: string | null;
}

export interface TeamPatchResponse {
  team_id: number;
  updated_fields: string[];
  wiki?: TeamWikiInfo;
  error?: string;
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * PATCH team with wiki fields
 *
 * Tolerant of backend not supporting this yet (405/501 only).
 * 404 is treated as a real error (team not found).
 * Returns structured response or throws on unexpected errors.
 */
export async function patchTeam(
  teamId: number,
  data: TeamPatchRequest
): Promise<TeamPatchResponse> {
  const res = await fetch(`/api/football/team/${teamId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  const json = await res.json().catch(() => ({}));

  // Handle backend not supporting PATCH method yet (405 Method Not Allowed, 501 Not Implemented)
  // Note: 404 is NOT treated as "not supported" - it means the team wasn't found (real error)
  if (res.status === 405 || res.status === 501) {
    const error = new Error("Backend aún no soporta campos wiki");
    (error as Error & { isNotSupported: boolean }).isNotSupported = true;
    throw error;
  }

  if (!res.ok) {
    throw new Error(json.error || json.detail || `PATCH failed: ${res.status}`);
  }

  return json;
}
