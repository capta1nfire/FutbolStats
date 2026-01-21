/**
 * Settings mock data
 * Provides deterministic data for settings (read-only in Phase 0)
 */

import {
  SettingsSummary,
  FeatureFlag,
  SettingsUser,
  FeatureFlagsFilters,
  UsersFilters,
} from "@/lib/types";
import { mockConfig, simulateDelay, checkMockError } from "./config";

/**
 * Static base timestamp for deterministic mock data
 */
const BASE_TIMESTAMP = new Date("2026-01-20T12:00:00Z").getTime();

/**
 * Mock feature flags
 */
const featureFlags: FeatureFlag[] = [
  {
    id: "shadow_model_evaluation",
    name: "Shadow Model Evaluation",
    description: "Enable two-stage model evaluation in shadow mode",
    enabled: true,
    updatedAt: new Date(BASE_TIMESTAMP - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "live_narratives",
    name: "Live Match Narratives",
    description: "Generate LLM narratives for live matches",
    enabled: true,
    updatedAt: new Date(BASE_TIMESTAMP - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "sensor_b_diagnostics",
    name: "Sensor B Diagnostics",
    description: "Enable calibration diagnostics for predictions",
    enabled: true,
    updatedAt: new Date(BASE_TIMESTAMP - 7 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "odds_sync",
    name: "Odds Synchronization",
    description: "Sync betting odds from API-Football",
    enabled: true,
    updatedAt: new Date(BASE_TIMESTAMP - 10 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "extended_leagues",
    name: "Extended Leagues Coverage",
    description: "Include additional leagues beyond core coverage",
    enabled: false,
    updatedAt: new Date(BASE_TIMESTAMP - 14 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "prediction_caching",
    name: "Prediction Caching",
    description: "Cache predictions to reduce computation",
    enabled: true,
    updatedAt: new Date(BASE_TIMESTAMP - 20 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "auto_incident_creation",
    name: "Auto Incident Creation",
    description: "Automatically create incidents for anomalies",
    enabled: false,
    updatedAt: new Date(BASE_TIMESTAMP - 25 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "gemini_provider",
    name: "Gemini LLM Provider",
    description: "Use Gemini instead of RunPod for narratives",
    enabled: true,
    updatedAt: new Date(BASE_TIMESTAMP - 1 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

/**
 * Mock users
 */
const users: SettingsUser[] = [
  {
    id: 1,
    email: "david@futbolstats.io",
    role: "admin",
    lastLogin: new Date(BASE_TIMESTAMP - 1 * 60 * 60 * 1000).toISOString(),
    createdAt: new Date(BASE_TIMESTAMP - 180 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 2,
    email: "ops@futbolstats.io",
    role: "admin",
    lastLogin: new Date(BASE_TIMESTAMP - 4 * 60 * 60 * 1000).toISOString(),
    createdAt: new Date(BASE_TIMESTAMP - 120 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 3,
    email: "analyst@futbolstats.io",
    role: "readonly",
    lastLogin: new Date(BASE_TIMESTAMP - 24 * 60 * 60 * 1000).toISOString(),
    createdAt: new Date(BASE_TIMESTAMP - 60 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 4,
    email: "auditor@futbolstats.io",
    role: "readonly",
    lastLogin: new Date(BASE_TIMESTAMP - 48 * 60 * 60 * 1000).toISOString(),
    createdAt: new Date(BASE_TIMESTAMP - 30 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

/**
 * Mock settings summary
 */
const settingsSummary: SettingsSummary = {
  lastUpdated: new Date(BASE_TIMESTAMP - 1 * 60 * 60 * 1000).toISOString(),
  environment: "prod",
  timezoneDisplay: "UTC (Coordinated Universal Time)",
  narrativeProvider: "gemini",
  apiFootballKeyStatus: "configured",
  modelVersions: {
    modelA: "xgb_v1.0.0",
    shadow: "xgb_v2.0.0-form",
    updatedAt: new Date(BASE_TIMESTAMP - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
  featureFlags,
  users,
};

/**
 * Empty settings for empty scenario
 */
const emptySettingsSummary: SettingsSummary = {
  ...settingsSummary,
  featureFlags: [],
  users: [],
};

/**
 * Get settings summary
 */
export async function getSettingsSummaryMock(): Promise<SettingsSummary> {
  await simulateDelay();
  checkMockError();

  switch (mockConfig.scenario) {
    case "empty":
      return emptySettingsSummary;
    default:
      return settingsSummary;
  }
}

/**
 * Get feature flags with optional filters
 */
export async function getFeatureFlagsMock(
  filters?: FeatureFlagsFilters
): Promise<FeatureFlag[]> {
  await simulateDelay();
  checkMockError();

  let data: FeatureFlag[];

  switch (mockConfig.scenario) {
    case "empty":
      data = [];
      break;
    default:
      data = [...featureFlags];
  }

  // Apply filters
  if (filters) {
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter(
        (f) =>
          f.name.toLowerCase().includes(search) ||
          f.id.toLowerCase().includes(search) ||
          (f.description && f.description.toLowerCase().includes(search))
      );
    }
    if (filters.enabled !== undefined) {
      data = data.filter((f) => f.enabled === filters.enabled);
    }
  }

  return data;
}

/**
 * Get users with optional filters
 */
export async function getUsersMock(
  filters?: UsersFilters
): Promise<SettingsUser[]> {
  await simulateDelay();
  checkMockError();

  let data: SettingsUser[];

  switch (mockConfig.scenario) {
    case "empty":
      data = [];
      break;
    default:
      data = [...users];
  }

  // Apply filters
  if (filters) {
    if (filters.search) {
      const search = filters.search.toLowerCase();
      data = data.filter((u) => u.email.toLowerCase().includes(search));
    }
    if (filters.role && filters.role.length > 0) {
      data = data.filter((u) => filters.role!.includes(u.role));
    }
  }

  return data;
}

/**
 * Mask an API key for display (security)
 * Shows only first 4 and last 4 characters
 */
export function maskApiKey(key: string): string {
  if (key.length <= 8) return "****";
  return `${key.slice(0, 4)}${"*".repeat(Math.min(key.length - 8, 20))}${key.slice(-4)}`;
}
