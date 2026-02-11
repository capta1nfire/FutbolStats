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

/**
 * League-level configuration stored in admin_leagues.tags JSONB
 */
export interface LeagueTags {
  use_short_names?: boolean;
  // Extensible for future settings
}

export interface LeagueInfo {
  league_id: number;
  name: string;
  country: string;
  kind: string;
  priority: string;
  match_type: string;
  match_weight: number | null;
  rules_json: Record<string, unknown>;
  tags?: LeagueTags;
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
  home_display_name?: string;  // Short name for use_short_names toggle
  away_display_name?: string;  // Short name for use_short_names toggle
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
  home_display_name?: string;  // Short name for use_short_names toggle
  away_display_name?: string;  // Short name for use_short_names toggle
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

/**
 * Wikipedia/Wikidata info for a team
 * - Editable: wiki_url, wikidata_id (user input)
 * - Read-only: wiki_title, wiki_lang, etc. (backend-derived)
 */
export interface TeamWikiInfo {
  // Editable fields (user input)
  wiki_url?: string | null;
  wikidata_id?: string | null;
  // Read-only fields (derived by backend)
  wiki_title?: string | null;
  wiki_lang?: string | null;
  wiki_url_cached?: string | null;
  wiki_source?: string | null;
  wiki_confidence?: number | null;
  wiki_matched_at?: string | null;
}

/**
 * Source badge info for enrichment display
 */
export interface EnrichmentSourceBadge {
  emoji: string;
  label: string;
  tooltip: string;
  color: string;
}

/**
 * Manual override values (only present when has_override is true)
 */
export interface TeamEnrichmentOverride {
  full_name: string | null;
  short_name: string | null;
  stadium_name: string | null;
  stadium_capacity: number | null;
  city: string | null;
  website: string | null;
  twitter: string | null;
  instagram: string | null;
  source: string | null;
  notes: string | null;
  updated_at: string | null;
}

/**
 * Wikidata enrichment with optional manual overrides
 *
 * The effective values (stadium_name, twitter, etc.) are COALESCE(override, wikidata).
 * Use `override` object to see raw override values for edit form.
 */
export interface TeamWikidataEnrichment {
  // Provenance
  wikidata_id: string | null;
  wikidata_updated_at: string | null;
  // Effective values (COALESCE override > wikidata)
  stadium_name: string | null;
  stadium_wikidata_id: string | null;
  stadium_capacity: number | null;
  stadium_altitude_m: number | null;
  city: string | null;
  lat: number | null;
  lon: number | null;
  full_name: string | null;
  short_name: string | null;
  website: string | null;
  twitter: string | null;
  instagram: string | null;
  // Source tracking
  enrichment_source: string;
  source_badge: EnrichmentSourceBadge;
  // Override state
  has_override: boolean;
  override: TeamEnrichmentOverride | null;
}

export interface TeamInfo {
  team_id: number;
  name: string;
  display_name: string;  // COALESCE(override.short_name, wikidata.short_name, name) - always has value
  short_name: string | null;
  country: string;
  founded: number | null;
  venue_name: string | null;
  venue_city: string | null;
  logo_url: string | null;
  wiki?: TeamWikiInfo;
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
  matches: number;
  seasons_range: [number, number];
}

export interface TeamFormMatch {
  match_id: number;
  date: string;
  opponent: string;
  result: "W" | "D" | "L";
  score: string;
  home: boolean;
}

// Feature Coverage (ATI P0 - Team Detail widget)
export interface FeatureCoverageSource {
  count: number;
  pct: number;
}

export interface FeatureCoverageKillswitch {
  ft_league_matches: number;
  lookback_days: number;
  min_required: number;
  status: "ok" | "warning" | "blocked";
}

export interface FeatureCoverage {
  killswitch: FeatureCoverageKillswitch | null;
  sources: {
    total_matches: number;
    odds: FeatureCoverageSource;
    xg: FeatureCoverageSource;
    lineup: FeatureCoverageSource;
    xi_depth: FeatureCoverageSource;
    form: FeatureCoverageSource;
    h2h: FeatureCoverageSource;
  } | null;
  tiers: Record<string, FeatureCoverageSource> | null;
}

export interface TeamDetail {
  team: TeamInfo;
  wikidata_enrichment?: TeamWikidataEnrichment | null;
  stats?: TeamStats;
  leagues_played?: TeamLeague[];
  recent_form?: TeamFormMatch[];
  feature_coverage?: FeatureCoverage | null;
}

export type TeamDetailResponse = FootballApiResponse<TeamDetail>;

// =============================================================================
// Tournaments & Cups (tournaments.json) - P3.4
// =============================================================================

export interface TournamentStats {
  total_matches: number;
  matches_30d: number;
  seasons_range: [number, number] | null;
  last_match: string | null;
  next_match: string | null;
  with_stats_pct: number | null;
  with_odds_pct: number | null;
  participants_count: number;
}

export interface TournamentEntry {
  league_id: number;
  name: string;
  country: string | null;
  kind: string;
  priority: string;
  stats: TournamentStats;
}

export interface TournamentsTotals {
  tournaments_count: number;
  cups_count: number;
  international_count: number;
  friendly_count: number;
}

export interface TournamentsList {
  tournaments: TournamentEntry[];
  totals: TournamentsTotals;
}

export type TournamentsListResponse = FootballApiResponse<TournamentsList>;

// =============================================================================
// World Cup 2026 (world-cup-2026/*.json) - P3.5
// =============================================================================

export interface WorldCupLeagueInfo {
  league_id: number;
  name?: string;
  season: number;
}

export interface WorldCupSummary {
  groups_count: number;
  teams_count: number;
  matches_total: number;
  matches_played: number;
  matches_upcoming: number;
  next_match_at: string | null;
  standings_source: string;
  standings_captured_at: string | null;
}

export interface WorldCupAlert {
  type: string;
  message: string;
  value: number | null;
}

export interface WorldCupUpcomingMatch {
  match_id: number;
  date: string | null;
  group: string | null;
  home_team: string;
  away_team: string;
  home_display_name?: string;  // Short name for use_short_names toggle
  away_display_name?: string;  // Short name for use_short_names toggle
  home_team_id: number | null;
  away_team_id: number | null;
  status: string;
}

// World Cup status union matches backend: "ok" | "not_ready" | "disabled"
export type WorldCupStatus = "ok" | "not_ready" | "disabled";

export interface WorldCupOverview {
  league: WorldCupLeagueInfo;
  status: WorldCupStatus;
  summary: WorldCupSummary;
  alerts: WorldCupAlert[];
  upcoming: WorldCupUpcomingMatch[];
}

export type WorldCupOverviewResponse = FootballApiResponse<WorldCupOverview>;

// World Cup Groups List
// Backend returns groups with embedded teams array, not summary stats

export interface WorldCupStandingEntry {
  team_id: number | null;
  external_id: number | null;
  name: string;
  logo_url: string | null;
  position: number;
  points: number;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_diff: number;
  form: string | null;
  description: string | null;
}

export interface WorldCupGroupWithTeams {
  group: string;
  teams: WorldCupStandingEntry[];
}

export interface WorldCupGroupsTotals {
  groups_count: number;
  teams_count: number;
}

export interface WorldCupGroups {
  league: WorldCupLeagueInfo;
  status: WorldCupStatus;
  groups: WorldCupGroupWithTeams[];
  totals: WorldCupGroupsTotals;
}

export type WorldCupGroupsResponse = FootballApiResponse<WorldCupGroups>;

// World Cup Group Detail

export interface WorldCupGroupMatch {
  match_id: number;
  date: string | null;
  home_team: string;
  away_team: string;
  home_display_name?: string;  // Short name for use_short_names toggle
  away_display_name?: string;  // Short name for use_short_names toggle
  home_team_id: number | null;
  away_team_id: number | null;
  score: string | null;
  status: string;
}

export interface WorldCupGroupDetail {
  league: WorldCupLeagueInfo;
  status: WorldCupStatus;
  group: string;
  standings: WorldCupStandingEntry[];
  matches: WorldCupGroupMatch[];
}

export type WorldCupGroupDetailResponse = FootballApiResponse<WorldCupGroupDetail>;

// =============================================================================
// National Teams (nationals/*.json) - P3.3
// =============================================================================

export interface NationalsCountryItem {
  country: string;
  teams_count: number;
  total_matches: number;
  competitions_count: number;
  last_match: string | null;
}

export interface NationalsCountriesTotals {
  countries_count: number;
  teams_count: number;
  competitions_count: number;
}

export interface NationalsCountriesList {
  countries: NationalsCountryItem[];
  totals: NationalsCountriesTotals;
}

export type NationalsCountriesListResponse = FootballApiResponse<NationalsCountriesList>;

export interface NationalsTeamItem {
  team_id: number;
  name: string;
  logo_url: string | null;
  total_matches: number;
  matches_25_26: number;
  competitions: string[];
}

export interface NationalsCompetitionItem {
  league_id: number;
  name: string;
  matches_count: number;
}

export interface NationalsRecentMatch {
  match_id: number;
  date: string | null;
  competition_name: string;
  home_team: string;
  away_team: string;
  home_display_name?: string;  // Short name for use_short_names toggle
  away_display_name?: string;  // Short name for use_short_names toggle
  status: string;
  score: string | null;
}

export interface NationalsCountryStats {
  total_matches: number;
  wins: number;
  draws: number;
  losses: number;
}

export interface NationalsCountryDetail {
  country: string;
  teams: NationalsTeamItem[];
  competitions: NationalsCompetitionItem[];
  recent_matches: NationalsRecentMatch[];
  stats: NationalsCountryStats;
}

export type NationalsCountryDetailResponse = FootballApiResponse<NationalsCountryDetail>;
