/**
 * Types for v2 drawer endpoints
 *
 * These types define the expected response shapes from:
 * - /dashboard/overview/rollup.json
 * - /dashboard/sentry/issues.json
 * - /dashboard/predictions/missing.json
 */

/**
 * Sentry issue from /dashboard/sentry/issues.json
 *
 * Note: Backend returns a subset of Sentry API fields with snake_case
 */
export interface SentryIssue {
  id: string;
  shortId?: string;
  title: string;
  culprit?: string;
  level: "error" | "warning" | "info" | "debug";
  status?: "unresolved" | "resolved" | "ignored";
  count: number;
  userCount?: number;
  firstSeen?: string;
  lastSeen: string;
  permalink?: string;
}

/**
 * Raw issue from backend
 */
interface SentryIssueRaw {
  id: string;
  title: string;
  level: string;
  count: number;
  first_seen_at: string;
  last_seen_at: string;
  issue_url?: string;
}

/**
 * Raw response from /dashboard/sentry/issues.json
 */
export interface SentryIssuesRawResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number;
  data: {
    issues: SentryIssueRaw[];
    total: number;
    page: number;
    limit: number;
    pages: number;
    status: string;
  };
}

/**
 * Normalize backend issue to frontend shape
 */
export function normalizeSentryIssue(raw: SentryIssueRaw): SentryIssue {
  return {
    id: raw.id,
    title: raw.title,
    level: raw.level as SentryIssue["level"],
    count: raw.count,
    firstSeen: raw.first_seen_at,
    lastSeen: raw.last_seen_at,
    permalink: raw.issue_url,
  };
}

/**
 * Normalized paginated response for Sentry issues
 */
export interface SentryIssuesResponse {
  issues: SentryIssue[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
  range: "1h" | "24h" | "7d";
}

/**
 * Match missing prediction from /dashboard/predictions/missing.json
 */
export interface MissingPredictionMatch {
  fixture_id: number;
  home_team: string;
  away_team: string;
  league_name: string;
  league_id: number;
  kickoff_utc: string;
  status: string;
  hours_until_kickoff: number;
}

/**
 * Raw response from /dashboard/predictions/missing.json
 */
export interface MissingPredictionsRawResponse {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number;
  data: {
    missing: MissingPredictionMatch[];
    missing_total: number;
    matches_total: number;
    coverage_pct: number;
    total: number;
    page: number;
    limit: number;
    pages: number;
    status: string;
  };
}

/**
 * Paginated response wrapper for missing predictions
 */
export interface MissingPredictionsResponse {
  matches: MissingPredictionMatch[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
  hours: number;
}

/**
 * Overview rollup response from /dashboard/overview/rollup.json
 *
 * This is a summary/rollup of key metrics for the overview page.
 * Shape TBD based on backend implementation.
 */
export interface OverviewRollupResponse {
  generated_at: string;
  // Add fields as backend defines them
  [key: string]: unknown;
}
