/**
 * Admin Panel Types
 *
 * Types matching the REAL backend contract:
 * - /dashboard/admin/overview.json
 * - /dashboard/admin/leagues.json
 * - /dashboard/admin/league/{id}.json
 * - /dashboard/admin/leagues/{id}.json (PATCH)
 * - /dashboard/admin/audit.json
 * - /dashboard/admin/league-groups.json
 * - /dashboard/admin/league-group/{id}.json
 *
 * Wrapper format: { generated_at, cached, cache_age_seconds, data }
 */

import type { FootballApiResponse } from "./football";

// =============================================================================
// Overview — data.counts, data.coverage_summary, data.top_leagues_by_matches
// =============================================================================

export interface AdminOverviewCounts {
  leagues_total: number;
  leagues_active: number;
  leagues_seed: number;
  leagues_observed: number;
  leagues_in_matches_30d: number;
  teams_total: number;
  teams_clubs: number;
  teams_national: number;
  matches_total: number;
  matches_25_26: number;
  predictions_total: number;
}

export interface AdminOverviewCoverage {
  matches_with_stats_pct: number;
  titan_tier1_25_26_pct: number;
}

export interface AdminOverviewTopLeague {
  league_id: number;
  name: string;
  matches: number;
}

export interface AdminOverviewData {
  counts: AdminOverviewCounts;
  coverage_summary: AdminOverviewCoverage;
  top_leagues_by_matches: AdminOverviewTopLeague[];
}

export type AdminOverviewResponse = FootballApiResponse<AdminOverviewData>;

// =============================================================================
// Leagues List — data.leagues[], data.totals, data.groups[], data.unmapped_in_matches[]
// =============================================================================

export interface AdminLeagueStats {
  total_matches: number;
  finished_matches: number;
  matches_25_26: number;
  with_stats_pct: number;
  unique_teams: number;
  seasons_range: number[];
  last_match: string | null;
}

export interface AdminLeagueListItem {
  league_id: number;
  name: string;
  country: string;
  kind: string;
  is_active: boolean;
  configured: boolean;
  source: string;
  priority: string;
  match_type: string;
  match_weight: number | null;
  observed: boolean;
  stats: AdminLeagueStats;
}

export interface AdminLeaguesTotals {
  total_in_db: number;
  active: number;
  seed: number;
  observed: number;
  in_matches: number;
  with_titan_data: number;
}

export interface AdminLeaguesList {
  leagues: AdminLeagueListItem[];
  totals: AdminLeaguesTotals;
  unmapped_in_matches: unknown[];
  groups: unknown[];
}

export type AdminLeaguesListResponse = FootballApiResponse<AdminLeaguesList>;

// =============================================================================
// League Detail — data.league, data.stats_by_season[], data.teams[], ...
// =============================================================================

export interface AdminLeagueDetailCore {
  league_id: number;
  name: string;
  country: string;
  kind: string;
  is_active: boolean;
  configured: boolean;
  source: string;
  priority: string;
  match_type: string;
  match_weight: number | null;
  observed: boolean;
}

export interface AdminLeagueSeasonStats {
  season: number;
  total_matches: number;
  finished: number;
  with_stats_pct: number;
  with_odds_pct: number;
}

export interface AdminLeagueDetailData {
  league: AdminLeagueDetailCore;
  stats_by_season: AdminLeagueSeasonStats[];
  teams: unknown[];
  titan_coverage: unknown;
  recent_matches: unknown[];
}

export type AdminLeagueDetailResponse = FootballApiResponse<AdminLeagueDetailData>;

// =============================================================================
// League PATCH Response
// =============================================================================

export interface AdminLeagueChangeEntry {
  old: unknown;
  new: unknown;
}

export interface AdminLeaguePatchResponse {
  league: AdminLeagueDetailCore;
  audit_id: number;
  changes_applied: Record<string, AdminLeagueChangeEntry>;
}

// =============================================================================
// League Groups List — data.groups[], data.total
// =============================================================================

export interface AdminLeagueGroupMember {
  league_id: number;
  name: string;
  is_active: boolean;
}

export interface AdminLeagueGroupStats {
  total_matches: number;
  matches_25_26: number;
  last_match: string | null;
  seasons_range: number[];
  with_stats_pct: number;
  with_odds_pct: number;
}

export interface AdminLeagueGroupListItem {
  group_id: number;
  group_key: string;
  name: string;
  country: string;
  leagues: AdminLeagueGroupMember[];
  is_active_any: boolean;
  is_active_all: boolean;
  stats: AdminLeagueGroupStats;
}

export interface AdminLeagueGroupsList {
  groups: AdminLeagueGroupListItem[];
  total: number;
}

export type AdminLeagueGroupsListResponse = FootballApiResponse<AdminLeagueGroupsList>;

// =============================================================================
// League Group Detail (same shape as list item for now)
// =============================================================================

export type AdminLeagueGroupDetail = AdminLeagueGroupListItem;
export type AdminLeagueGroupDetailResponse = FootballApiResponse<AdminLeagueGroupDetail>;

// =============================================================================
// Audit — data.entries[], data.pagination, data.filters
// =============================================================================

export interface AdminAuditEntry {
  id: number;
  entity_type: string;
  entity_id: string;
  action: string;
  actor: string;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  created_at: string;
}

export interface AdminAuditPagination {
  total: number;
  limit: number;
  offset: number;
}

export interface AdminAuditList {
  entries: AdminAuditEntry[];
  pagination: AdminAuditPagination;
  filters: Record<string, string>;
}

export type AdminAuditListResponse = FootballApiResponse<AdminAuditList>;

// =============================================================================
// Filters (used by hooks)
// =============================================================================

export interface AdminLeaguesFilters {
  search?: string;
  country?: string;
  kind?: string;
  is_active?: string;
  source?: string;
}

export interface AdminAuditFilters {
  entity_type?: string;
  entity_id?: string;
  limit?: number;
  offset?: number;
}
