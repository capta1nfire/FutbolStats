/**
 * Team Enrichment API Client
 *
 * Mutations for updating team enrichment overrides.
 * Uses Next.js API routes for auth handling.
 */

// =============================================================================
// Types
// =============================================================================

export interface TeamEnrichmentPutRequest {
  full_name?: string | null;
  short_name?: string | null;
  stadium_name?: string | null;
  stadium_capacity?: number | null;
  stadium_wikidata_id?: string | null;
  city?: string | null;
  website?: string | null;
  twitter_handle?: string | null;
  instagram_handle?: string | null;
  source?: string;
  notes?: string | null;
}

export interface TeamEnrichmentPutResponse {
  status: string;
  team_id: number;
  action: "upserted" | "deleted";
}

export interface TeamEnrichmentDeleteResponse {
  status: string;
  team_id: number;
  deleted: boolean;
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * PUT team enrichment override
 *
 * Upserts override values. Empty strings are normalized to NULL.
 * If all fields end up NULL, the override row is deleted.
 */
export async function putTeamEnrichment(
  teamId: number,
  data: TeamEnrichmentPutRequest
): Promise<TeamEnrichmentPutResponse> {
  const res = await fetch(`/api/football/team/${teamId}/enrichment`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  const json = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(json.error || json.detail || `PUT failed: ${res.status}`);
  }

  return json;
}

/**
 * DELETE team enrichment override
 *
 * Removes all override values, reverting to Wikidata source.
 */
export async function deleteTeamEnrichment(
  teamId: number
): Promise<TeamEnrichmentDeleteResponse> {
  const res = await fetch(`/api/football/team/${teamId}/enrichment`, {
    method: "DELETE",
  });

  const json = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(json.error || json.detail || `DELETE failed: ${res.status}`);
  }

  return json;
}
