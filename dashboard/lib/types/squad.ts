/**
 * Types for Players & Managers (Squad) data.
 * ABE P1-1: injury_type is string (not closed union) to handle unknown types.
 */

export interface ManagerInfo {
  external_id?: number;
  name: string;
  nationality: string | null;
  photo_url: string | null;
  start_date: string | null;
  end_date?: string | null;
  tenure_days?: number | null;
}

export interface PlayerInjury {
  player_name: string;
  injury_type: string;
  injury_reason: string | null;
  fixture_date?: string | null;
}

// Match squad (Match Detail > Squad tab)
export interface TeamSquadSide {
  team_id: number;
  team_name: string;
  manager: ManagerInfo | null;
  injuries: PlayerInjury[];
}

export interface MatchSquadData {
  match_id: number;
  home: TeamSquadSide;
  away: TeamSquadSide;
}

// Team squad (TeamDrawer > overview)
export interface TeamSquadData {
  team_id: number;
  team_name: string;
  current_manager: ManagerInfo | null;
  manager_history: ManagerInfo[];
  current_injuries: PlayerInjury[];
}

// Team squad stats (TeamDrawer > Squad tab)
export interface TeamSquadPlayerSeasonStats {
  player_external_id: number;
  player_name: string;
  photo_url?: string | null;
  photo_url_thumb_hq?: string | null;
  photo_url_card_hq?: string | null;
  position: string; // 'G' | 'D' | 'M' | 'F' | 'U'
  jersey_number: number | null;
  appearances: number;
  avg_rating: number | null;
  total_minutes: number;
  goals: number;
  assists: number;
  saves: number;
  yellows: number;
  reds: number;
  key_passes: number;
  tackles: number;
  interceptions: number;
  shots_total: number;
  shots_on_target: number;
  passes_total: number;
  passes_accuracy: number | null;
  blocks: number;
  duels_total: number;
  duels_won: number;
  dribbles_attempts: number;
  dribbles_success: number;
  fouls_drawn: number;
  fouls_committed: number;
  ever_captain: boolean;
  // Bio fields (from players table, may be null until sync)
  firstname?: string | null;
  lastname?: string | null;
  birth_date?: string | null;
  birth_place?: string | null;
  birth_country?: string | null;
  nationality?: string | null;
  height?: string | null;
  weight?: string | null;
}

export interface TeamSquadStatsData {
  team_id: number;
  team_external_id: number | null;
  team_name: string;
  season: number | null;
  available_seasons: number[];
  team_matches_played: number;
  players: TeamSquadPlayerSeasonStats[];
}

// Players view (Football > Players category)
export interface PlayersViewData {
  leagues: Array<{
    league_id: number;
    name: string;
    teams: Array<{
      team_id: number;
      name: string;
      injuries: PlayerInjury[];
    }>;
    absences_count: number;
  }>;
  total_absences: number;
}

// Managers view (Football > Managers category)
export interface ManagerEntry {
  team_id: number;
  team_name: string;
  league_id: number;
  league_name: string;
  manager: ManagerInfo;
  tenure_days: number | null;
  is_new: boolean;
}

export interface ManagersViewData {
  managers: ManagerEntry[];
  total_managers: number;
  new_managers_count: number;
}
