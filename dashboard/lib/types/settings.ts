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
  | "users";

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
