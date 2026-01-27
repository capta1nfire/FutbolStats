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

export interface AdminLeagueGroupSummary {
  key: string;
  name: string;
  country: string;
}

export interface AdminLeaguesList {
  leagues: AdminLeagueListItem[];
  totals: AdminLeaguesTotals;
  unmapped_in_matches: number[];
  groups: AdminLeagueGroupSummary[];
}

export type AdminLeaguesListResponse = FootballApiResponse<AdminLeaguesList>;

// =============================================================================
// League Detail — data.league, data.stats_by_season[], data.teams[], ...
// =============================================================================

export interface AdminLeagueGroupRef {
  key: string;
  name: string;
  country: string;
  paired_handling?: string;
}

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
  group?: AdminLeagueGroupRef;
}

export interface AdminLeagueSeasonStats {
  season: number;
  total_matches: number;
  finished: number;
  with_stats_pct: number;
  with_odds_pct: number;
}

export interface AdminLeagueTeam {
  team_id: number;
  external_id: number;
  name: string;
  matches_in_league: number;
}

export interface AdminLeagueTitanCoverage {
  total: number;
  tier1: number;
  tier1b: number;
  tier1_pct: number;
  tier1b_pct: number;
}

export interface AdminLeagueRecentMatch {
  match_id: number;
  date: string;
  home: string;
  away: string;
  status: string;
  has_stats: boolean;
  has_prediction: boolean;
}

export interface AdminLeagueDetailData {
  league: AdminLeagueDetailCore;
  stats_by_season: AdminLeagueSeasonStats[];
  teams: AdminLeagueTeam[];
  titan_coverage: AdminLeagueTitanCoverage | null;
  recent_matches: AdminLeagueRecentMatch[];
}

export type AdminLeagueDetailResponse = FootballApiResponse<AdminLeagueDetailData>;

// =============================================================================
// League PATCH Response
// =============================================================================

export interface AdminLeaguePatchResponse {
  league: AdminLeagueDetailCore;
  audit_id: number;
  changes_applied: string[];
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
// League Group Detail — full response from /dashboard/admin/league-group/{id}.json
// =============================================================================

export interface AdminLeagueGroupMemberFull {
  league_id: number;
  name: string;
  kind: string;
  is_active: boolean;
  stats?: AdminLeagueStats;
}

export interface AdminLeagueGroupSeasonStats {
  season: number;
  total_matches: number;
  finished: number;
  with_stats_pct: number;
}

export interface AdminLeagueGroupTeam {
  team_id: number;
  name: string;
  country?: string;
}

export interface AdminLeagueGroupMatch {
  match_id: number;
  date: string;
  home_team: string;
  away_team: string;
  score?: string;
}

export interface AdminLeagueGroupDetailFull {
  group: {
    group_id: number;
    group_key: string;
    name: string;
    country: string;
    tags?: Record<string, unknown>;
  };
  member_leagues: AdminLeagueGroupMemberFull[];
  is_active_any: boolean;
  is_active_all: boolean;
  stats_by_season: AdminLeagueGroupSeasonStats[];
  teams_count: number;
  teams: AdminLeagueGroupTeam[];
  recent_matches: AdminLeagueGroupMatch[];
}

export type AdminLeagueGroupDetailResponse = FootballApiResponse<AdminLeagueGroupDetailFull>;

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
  has_more: boolean;
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
