/**
 * Settings Types
 *
 * Types for the Settings section (read-only in Phase 0)
 */

export type SettingsSection =
  | "general"
  | "timezone"
  | "notifications"
  | "api_keys"
  | "model_versions"
  | "feature_flags"
  | "users"
  | "ia_features";

export type Environment = "prod" | "staging" | "local";

export type ApiKeyStatus = "missing" | "configured" | "invalid";

export type UserRole = "admin" | "readonly";

/**
 * Model version info
 */
export interface ModelVersionInfo {
  modelA: string;
  shadow: string;
  updatedAt: string; // ISO
}

/**
 * Feature flag
 */
export interface FeatureFlag {
  id: string;
  name: string;
  description?: string;
  enabled: boolean;
  updatedAt?: string; // ISO
}

/**
 * User info
 */
export interface SettingsUser {
  id: number;
  email: string;
  role: UserRole;
  lastLogin?: string; // ISO
  createdAt?: string; // ISO
}

/**
 * Settings summary for overview
 */
export interface SettingsSummary {
  lastUpdated: string; // ISO
  environment: Environment;
  timezoneDisplay: string;
  narrativeProvider: string;
  apiFootballKeyStatus: ApiKeyStatus;
  modelVersions: ModelVersionInfo;
  featureFlags: FeatureFlag[];
  users: SettingsUser[];
}

/**
 * Filters for feature flags table
 */
export interface FeatureFlagsFilters {
  search?: string;
  enabled?: boolean;
}

/**
 * Filters for users table
 */
export interface UsersFilters {
  search?: string;
  role?: UserRole[];
}

/**
 * Section labels for display
 */
export const SETTINGS_SECTION_LABELS: Record<SettingsSection, string> = {
  general: "General",
  timezone: "Timezone",
  notifications: "Notifications",
  api_keys: "API Keys",
  model_versions: "Model Versions",
  feature_flags: "Feature Flags",
  users: "Users & Permissions",
  ia_features: "IA Features",
};

/**
 * All settings sections in order
 */
export const SETTINGS_SECTIONS: SettingsSection[] = [
  "general",
  "timezone",
  "notifications",
  "api_keys",
  "model_versions",
  "feature_flags",
  "users",
  "ia_features",
];

/**
 * Environment labels
 */
export const ENVIRONMENT_LABELS: Record<Environment, string> = {
  prod: "Production",
  staging: "Staging",
  local: "Local",
};

/**
 * API key status labels
 */
export const API_KEY_STATUS_LABELS: Record<ApiKeyStatus, string> = {
  missing: "Not Configured",
  configured: "Configured",
  invalid: "Invalid",
};

/**
 * User role labels
 */
export const USER_ROLE_LABELS: Record<UserRole, string> = {
  admin: "Admin",
  readonly: "Read-only",
};

/**
 * All user roles
 */
export const USER_ROLES: UserRole[] = ["admin", "readonly"];

/**
 * LLM Model info from backend catalog
 */
export interface LlmModelInfo {
  id: string;
  display_name: string;
  provider: string;
  input_price: number; // per 1M tokens
  output_price: number; // per 1M tokens
  max_tokens: number;
}

/**
 * Narratives enabled state (3-state: true/false/null)
 * null = inherit from env var FASTPATH_ENABLED
 */
export type NarrativesEnabledState = boolean | null;

/**
 * IA Features configuration
 */
export interface IaFeaturesConfig {
  narratives_enabled: NarrativesEnabledState;
  narrative_feedback_enabled: boolean; // Read-only placeholder
  primary_model: string;
  temperature: number;
  max_tokens: number;
  effective_enabled: boolean; // Resolved value after inheritance
  env_fastpath_enabled: boolean; // For "Inherit" display
  available_models: LlmModelInfo[];
}

/**
 * IA Features update payload (partial)
 */
export interface IaFeaturesUpdatePayload {
  narratives_enabled?: NarrativesEnabledState;
  primary_model?: string;
  temperature?: number;
  max_tokens?: number;
}

// =============================================================================
// IA Features: Visibility Types (Fase 2)
// =============================================================================

/**
 * Prompt template response
 */
export interface PromptTemplateResponse {
  version: string;
  prompt_template: string;
  char_count: number;
  notes: string;
}

/**
 * Match data for preview
 */
export interface PreviewMatchData {
  match_id: number;
  home_team: string;
  away_team: string;
  home_team_id: number;
  away_team_id: number;
  league_name: string;
  date: string;
  home_goals: number;
  away_goals: number;
  venue: { name?: string; city?: string };
  stats: Record<string, unknown>;
  prediction: Record<string, unknown>;
  events: unknown[];
  market_odds: Record<string, number>;
  derived_facts: Record<string, unknown>;
  narrative_style: Record<string, unknown>;
}

/**
 * Payload preview response
 */
export interface PayloadPreviewResponse {
  match_id: number;
  match_label: string;
  status: string;
  prompt_preview: string;
  match_data: PreviewMatchData;
}

/**
 * Call history item
 */
export interface CallHistoryItem {
  match_id: number;
  match_label: string;
  generated_at: string | null;
  model: string | null;
  prompt_version: string | null;
  tokens_in: number;
  tokens_out: number;
  latency_ms: number | null;
  exec_ms: number | null;
  cost_usd: number;
  status: string;
  audit_url: string;
}

/**
 * Call history response
 */
export interface CallHistoryResponse {
  items: CallHistoryItem[];
  total: number;
  limit: number;
}

/**
 * Match option for select dropdowns
 */
export interface MatchOption {
  id: number;
  label: string;
  status: string;
}

// =============================================================================
// IA Features: Playground Types (Fase 3)
// =============================================================================

/**
 * Playground request payload
 */
export interface PlaygroundRequest {
  match_id: number;
  temperature?: number;
  max_tokens?: number;
  model?: string;
}

/**
 * Playground response
 */
export interface PlaygroundResponse {
  narrative: {
    title: string;
    body: string;
    key_factors?: unknown[];
  };
  model_used: string;
  metrics: {
    tokens_in: number;
    tokens_out: number;
    latency_ms: number;
    cost_usd: number;
  };
  warnings: string[];
  rate_limit: {
    remaining: number;
    reset_at: string;
  };
}
