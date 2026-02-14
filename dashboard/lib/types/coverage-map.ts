/** Coverage Map types — matches backend contract coverage-map.v1 */

export interface CoverageDimension {
  pct: number;
  numerator: number;
  denominator: number;
}

export interface CoverageLeague {
  league_id: number;
  league_name: string;
  country: string;
  country_iso3: string | null;
  logo_url: string | null;
  wikipedia_url: string | null;
  eligible_matches: number;
  p0_pct: number;
  p1_pct: number;
  p2_pct: number;
  coverage_total_pct: number;
  universe_tier: string;
  universe_coverage: Record<string, number>;
  dimensions: Record<string, CoverageDimension>;
}

export interface CoverageCountry {
  country_iso3: string;
  country_name: string;
  league_count: number;
  eligible_matches: number;
  coverage_total_pct: number;
  p0_pct: number;
  p1_pct: number;
  p2_pct: number;
  universe_tier: string;
  universe_coverage: Record<string, number>;
  dimensions: Record<string, CoverageDimension>;
}

export interface CoverageMapData {
  contract_version: string;
  countries: CoverageCountry[];
  leagues?: CoverageLeague[];
  summary: {
    countries: number;
    leagues: number;
    eligible_matches: number;
    coverage_total_pct_mean: number;
  };
  color_scale: Array<{ min: number; max: number; color: string }>;
  weights: Record<string, number>;
}

export type CoverageWindow =
  | "current_season"
  | "prev_season"
  | "prev_season_2"
  | "since_2023"
  | "last_365d"
  | "season_to_date";
