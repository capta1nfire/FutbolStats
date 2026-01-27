/**
 * Feature Coverage Detail Drawer types
 *
 * Supports URL state ?cell=feature_key:league_id for shareable drawer links.
 */

/**
 * Parsed cell selection from URL param ?cell=feature_key:league_id
 */
export interface FeatureCoverageCellSelection {
  featureKey: string;
  leagueId: number;
}

/**
 * Parse cell param "home_goals_scored_avg:39" -> {featureKey, leagueId}
 *
 * Uses lastIndexOf(":") since feature keys use underscores, never colons.
 * Returns null for invalid/missing params.
 */
export function parseCellParam(
  param: string | null
): FeatureCoverageCellSelection | null {
  if (!param) return null;
  const idx = param.lastIndexOf(":");
  if (idx <= 0) return null;
  const featureKey = param.slice(0, idx);
  const leagueId = parseInt(param.slice(idx + 1), 10);
  if (isNaN(leagueId) || leagueId <= 0) return null;
  return { featureKey, leagueId };
}

/**
 * Build cell param string for URL: "home_goals_scored_avg:39"
 */
export function buildCellParam(
  featureKey: string,
  leagueId: number
): string {
  return `${featureKey}:${leagueId}`;
}
