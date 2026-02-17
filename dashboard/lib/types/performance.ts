export interface TeamPerformanceMatch {
  match_id: number;
  match_date: string;
  round: string | null;
  opponent_name: string;
  opponent_logo: string | null;
  is_home: boolean;
  goals_for: number;
  goals_against: number;
  result: "W" | "D" | "L";
  points: number;
  cumulative_points: number;
  xg_for: number | null;
  xg_against: number | null;
  cumulative_xg_for: number | null;
  cumulative_xg_against: number | null;
  avg_team_rating: number | null;
}

export interface TeamPerformanceData {
  team_id: number;
  team_name: string;
  league_id: number;
  season: number | null;
  xg_source: "understat" | "fotmob" | null;
  rating_method: string;
  matches: TeamPerformanceMatch[];
}
