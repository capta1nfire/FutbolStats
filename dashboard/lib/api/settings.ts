/**
 * Settings API Adapter
 *
 * Safe extraction and adaptation of data from /dashboard/settings/*.json
 * Designed to be resilient to partial or malformed responses.
 */

// ============================================================================
// Helpers
// ============================================================================

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// ============================================================================
// Settings Summary Types & Parser
// ============================================================================

/**
 * Integration status from summary
 */
export interface SettingsIntegration {
  key: string;
  configured: boolean;
  source: string;
}

/**
 * Settings summary from backend
 */
export interface SettingsSummary {
  readonly: boolean;
  sections: string[];
  notes?: string;
  integrations: SettingsIntegration[];
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Parse settings summary from API response
 */
export function parseSettingsSummary(response: unknown): SettingsSummary | null {
  if (!isObject(response)) return null;

  const data = response.data;
  if (!isObject(data)) return null;

  const readonly = typeof data.readonly === "boolean" ? data.readonly : true;
  const sections = Array.isArray(data.sections)
    ? data.sections.filter((s): s is string => typeof s === "string")
    : [];
  const notes = typeof data.notes === "string" ? data.notes : undefined;

  // Parse integrations
  const integrations: SettingsIntegration[] = [];
  if (isObject(data.integrations)) {
    for (const [key, value] of Object.entries(data.integrations)) {
      if (isObject(value)) {
        integrations.push({
          key,
          configured: typeof value.configured === "boolean" ? value.configured : false,
          source: typeof value.source === "string" ? value.source : "unknown",
        });
      }
    }
  }

  const generatedAt = typeof response.generated_at === "string" ? response.generated_at : null;
  const cached = typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds = typeof response.cache_age_seconds === "number" ? response.cache_age_seconds : 0;

  return {
    readonly,
    sections,
    notes,
    integrations,
    generatedAt,
    cached,
    cacheAgeSeconds,
  };
}

// ============================================================================
// Feature Flags Types & Parser
// ============================================================================

/**
 * Feature flag scope
 */
export type FeatureFlagScope = "llm" | "sota" | "sensor" | "jobs" | "predictions" | "other";

/**
 * Single feature flag
 */
export interface FeatureFlag {
  key: string;
  enabled: boolean | null;
  scope: FeatureFlagScope;
  description: string;
  source: string;
}

/**
 * Feature flags response with pagination
 */
export interface FeatureFlagsResponse {
  flags: FeatureFlag[];
  total: number;
  page: number;
  limit: number;
  pages: number;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Validate feature flag scope
 */
function isValidScope(scope: unknown): scope is FeatureFlagScope {
  return (
    scope === "llm" ||
    scope === "sota" ||
    scope === "sensor" ||
    scope === "jobs" ||
    scope === "predictions" ||
    scope === "other"
  );
}

/**
 * Parse feature flags from API response
 */
export function parseFeatureFlags(response: unknown): FeatureFlagsResponse | null {
  if (!isObject(response)) return null;

  const data = response.data;
  if (!isObject(data)) return null;

  const rawFlags = data.flags;
  if (!Array.isArray(rawFlags)) return null;

  const flags: FeatureFlag[] = [];
  for (const flag of rawFlags) {
    if (!isObject(flag)) continue;

    const key = flag.key;
    if (typeof key !== "string") continue;

    const enabled = typeof flag.enabled === "boolean" ? flag.enabled : null;
    const scope = isValidScope(flag.scope) ? flag.scope : "other";
    const description = typeof flag.description === "string" ? flag.description : "";
    const source = typeof flag.source === "string" ? flag.source : "unknown";

    flags.push({ key, enabled, scope, description, source });
  }

  const total = typeof data.total === "number" ? data.total : flags.length;
  const page = typeof data.page === "number" ? data.page : 1;
  const limit = typeof data.limit === "number" ? data.limit : 50;
  const pages = typeof data.pages === "number" ? data.pages : 1;

  const generatedAt = typeof response.generated_at === "string" ? response.generated_at : null;
  const cached = typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds = typeof response.cache_age_seconds === "number" ? response.cache_age_seconds : 0;

  return {
    flags,
    total,
    page,
    limit,
    pages,
    generatedAt,
    cached,
    cacheAgeSeconds,
  };
}

// ============================================================================
// Model Versions Types & Parser
// ============================================================================

/**
 * Single model version
 */
export interface ModelVersion {
  name: string;
  version: string;
  source: string;
  updatedAt: string | null;
}

/**
 * Model versions response
 */
export interface ModelVersionsResponse {
  models: ModelVersion[];
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Parse model versions from API response
 */
export function parseModelVersions(response: unknown): ModelVersionsResponse | null {
  if (!isObject(response)) return null;

  const data = response.data;
  if (!isObject(data)) return null;

  const rawModels = data.models;
  if (!Array.isArray(rawModels)) return null;

  const models: ModelVersion[] = [];
  for (const model of rawModels) {
    if (!isObject(model)) continue;

    const name = model.name;
    if (typeof name !== "string") continue;

    const version = typeof model.version === "string" ? model.version : "unknown";
    const source = typeof model.source === "string" ? model.source : "unknown";
    const updatedAt = typeof model.updated_at === "string" ? model.updated_at : null;

    models.push({ name, version, source, updatedAt });
  }

  const generatedAt = typeof response.generated_at === "string" ? response.generated_at : null;
  const cached = typeof response.cached === "boolean" ? response.cached : false;
  const cacheAgeSeconds = typeof response.cache_age_seconds === "number" ? response.cache_age_seconds : 0;

  return {
    models,
    generatedAt,
    cached,
    cacheAgeSeconds,
  };
}

// ============================================================================
// Metadata Extraction
// ============================================================================

/**
 * Common metadata from any settings response
 */
export interface SettingsMetadata {
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number;
}

/**
 * Extract common metadata from response
 */
export function extractSettingsMetadata(response: unknown): SettingsMetadata {
  if (!isObject(response)) {
    return { generatedAt: null, cached: false, cacheAgeSeconds: 0 };
  }

  return {
    generatedAt: typeof response.generated_at === "string" ? response.generated_at : null,
    cached: typeof response.cached === "boolean" ? response.cached : false,
    cacheAgeSeconds: typeof response.cache_age_seconds === "number" ? response.cache_age_seconds : 0,
  };
}
