/**
 * Football Navigation Types
 *
 * Types for Football navigation endpoints:
 * - /dashboard/football/nav.json
 * - /dashboard/football/overview.json
 * - /dashboard/football/leagues/countries.json
 * - /dashboard/football/leagues/country/{country}.json
 * - /dashboard/football/league/{id}.json
 * - /dashboard/football/group/{id}.json
 * - /dashboard/admin/team/{id}.json
 *
 * Wrapper format: { generated_at, cached, cache_age_seconds, data }
 */

// =============================================================================
// Generic API Response Wrapper
// =============================================================================

export interface FootballApiResponse<T> {
  generated_at: string;
  cached: boolean;
  cache_age_seconds: number | null;
  data: T;
}

// =============================================================================
// Navigation Categories (nav.json)
// =============================================================================

export interface NavCategory {
  id: string;
  label: string;
  enabled: boolean;
  note?: string;
  count?: number;
}

export interface FootballNav {
  sport: "football";
  categories: NavCategory[];
}

export type FootballNavResponse = FootballApiResponse<FootballNav>;

// =============================================================================
// Countries List (countries.json)
// =============================================================================

export interface CountryLeagueItem {
  league_id: number;
  name: string;
  kind: string;
  group_id?: number;
}

export interface CountryItem {
  country: string;
  leagues_count: number;
  groups_count: number;
  leagues: CountryLeagueItem[];
}

export interface CountriesList {
  countries: CountryItem[];
  total: number;
}

export type CountriesListResponse = FootballApiResponse<CountriesList>;

// =============================================================================
// Country Detail (country/{country}.json)
// =============================================================================

export interface CompetitionStats {
  total_matches: number;
  seasons_range: [number, number] | null;
  last_match: string | null;
  with_stats_pct: number | null;
  with_odds_pct: number | null;
}

export interface CompetitionTitan {
  total: number;
  tier1: number;
  tier1_pct: number | null;
}

export interface CompetitionMember {
  league_id: number;
  name: string;
  kind: string;
}

export interface CompetitionEntry {
  type: "league" | "group";
  league_id?: number;
  group_id?: number;
  group_key?: string;
  name: string;
  kind?: string;
  priority?: string;
  match_type?: string;
  member_count?: number;
  members?: CompetitionMember[];
  stats: CompetitionStats;
  titan?: CompetitionTitan;
}

export interface CountryDetail {
  country: string;
  competitions: CompetitionEntry[];
  total: number;
}

export type CountryDetailResponse = FootballApiResponse<CountryDetail>;

// =============================================================================
// League Detail (league/{id}.json)
// =============================================================================

export interface LeagueInfo {
  league_id: number;
  name: string;
  country: string;
  kind: string;
  priority: string;
  match_type: string;
  match_weight: number | null;
  rules_json: Record<string, unknown>;
}

export interface LeagueGroupInfo {
  group_id: number;
  key: string;
  name: string;
  country: string;
  paired_handling: string;
}

export interface SeasonStats {
  season: number;
  total_matches: number;
  finished: number;
  with_stats_pct: number | null;
  with_odds_pct: number | null;
}

export interface LeagueTitan {
  total: number;
  tier1: number;
  tier1b: number;
  tier1c: number;
  tier1d: number;
  tier1_pct: number;
}

export interface RecentMatch {
  match_id: number;
  date: string | null;
  status: string;
  home_team: string;
  away_team: string;
  home_team_id?: number;
  away_team_id?: number;
  score: string | null;
  league_id?: number;
}

export interface StandingsPlaceholder {
  status: string;
  note: string;
}

export interface LeagueDetail {
  league: LeagueInfo;
  group: LeagueGroupInfo | null;
  stats_by_season: SeasonStats[];
  titan: LeagueTitan | null;
  recent_matches: RecentMatch[];
  standings: StandingsPlaceholder;
}

export type LeagueDetailResponse = FootballApiResponse<LeagueDetail>;

// =============================================================================
// Group Detail (group/{id}.json)
// =============================================================================

export interface GroupInfo {
  group_id: number;
  group_key: string;
  name: string;
  country: string;
  paired_handling: string;
}

export interface GroupMemberLeague {
  league_id: number;
  name: string;
  kind: string;
  priority: string;
  match_type: string;
}

export interface GroupTitan {
  total: number;
  tier1: number;
  tier1_pct: number;
}

export interface GroupDetail {
  group: GroupInfo;
  member_leagues: GroupMemberLeague[];
  is_active_all: boolean;
  stats_by_season: SeasonStats[];
  titan: GroupTitan | null;
  recent_matches: RecentMatch[];
  standings: StandingsPlaceholder;
}

export type GroupDetailResponse = FootballApiResponse<GroupDetail>;

// =============================================================================
// Overview (overview.json)
// =============================================================================

export interface OverviewSummary {
  leagues_active_count: number;
  countries_active_count: number;
  matches_next_7d_count: number;
  matches_live_count: number;
  matches_finished_24h_count: number;
  teams_active_count: number;
}

export interface FootballUpcomingMatch {
  match_id: number;
  date: string | null;
  league_id: number;
  league_name: string;
  home_team: string;
  away_team: string;
  status: string;
  has_prediction: boolean;
}

export interface TopLeague {
  league_id: number;
  name: string;
  country: string;
  matches_30d: number;
  matches_total: number;
  with_stats_pct: number | null;
  with_odds_pct: number | null;
}

export interface OverviewAlert {
  type: string;
  league_id: number;
  league_name: string;
  message: string;
  value: number;
}

export interface OverviewTitan {
  total: number;
  tier1: number;
  tier1b: number;
  tier1_pct: number;
  tier1b_pct: number;
}

export interface FootballOverview {
  summary: OverviewSummary;
  upcoming: FootballUpcomingMatch[];
  leagues: TopLeague[];
  alerts: OverviewAlert[];
  titan: OverviewTitan | null;
}

export type FootballOverviewResponse = FootballApiResponse<FootballOverview>;

// =============================================================================
// Team Detail (admin/team/{id}.json) - Temporary for P0
// =============================================================================

export interface TeamInfo {
  team_id: number;
  name: string;
  short_name: string | null;
  country: string;
  founded: number | null;
  venue_name: string | null;
  venue_city: string | null;
  logo_url: string | null;
}

export interface TeamStats {
  total_matches: number;
  home_matches: number;
  away_matches: number;
  wins: number;
  draws: number;
  losses: number;
  goals_for: number;
  goals_against: number;
}

export interface TeamLeague {
  league_id: number;
  name: string;
  country: string;
  seasons: number[];
}

export interface TeamFormMatch {
  match_id: number;
  date: string;
  opponent: string;
  result: "W" | "D" | "L";
  score: string;
  home: boolean;
}

export interface TeamDetail {
  team: TeamInfo;
  stats?: TeamStats;
  leagues?: TeamLeague[];
  recent_form?: TeamFormMatch[];
}

export type TeamDetailResponse = FootballApiResponse<TeamDetail>;
