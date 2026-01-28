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

// ============================================================================
// IA Features Types & Parser
// ============================================================================

/**
 * LLM Model info from backend catalog
 */
export interface LlmModelInfo {
  id: string;
  displayName: string;
  provider: string;
  inputPrice: number; // per 1M tokens
  outputPrice: number; // per 1M tokens
  maxTokens: number;
}

/**
 * IA Features configuration
 */
export interface IaFeaturesConfig {
  narrativesEnabled: boolean | null; // null = inherit from env
  narrativeFeedbackEnabled: boolean; // Read-only placeholder
  primaryModel: string;
  temperature: number;
  maxTokens: number;
  effectiveEnabled: boolean; // Resolved value after inheritance
  envFastpathEnabled: boolean; // For "Inherit" display
  availableModels: LlmModelInfo[];
}

/**
 * IA Features response
 */
export interface IaFeaturesResponse extends SettingsMetadata {
  config: IaFeaturesConfig;
}

/**
 * Parse IA Features from API response
 */
export function parseIaFeatures(response: unknown): IaFeaturesResponse | null {
  if (!isObject(response)) return null;

  const data = response.data;
  if (!isObject(data)) return null;

  // Parse narratives_enabled (can be null, true, or false)
  const narrativesEnabled =
    data.narratives_enabled === null
      ? null
      : typeof data.narratives_enabled === "boolean"
        ? data.narratives_enabled
        : null;

  const narrativeFeedbackEnabled =
    typeof data.narrative_feedback_enabled === "boolean"
      ? data.narrative_feedback_enabled
      : false;

  const primaryModel =
    typeof data.primary_model === "string" ? data.primary_model : "gemini-2.5-flash-lite";

  const temperature =
    typeof data.temperature === "number" ? data.temperature : 0.7;

  const maxTokens =
    typeof data.max_tokens === "number" ? data.max_tokens : 4096;

  const effectiveEnabled =
    typeof data.effective_enabled === "boolean" ? data.effective_enabled : false;

  const envFastpathEnabled =
    typeof data.env_fastpath_enabled === "boolean" ? data.env_fastpath_enabled : false;

  // Parse available models
  const availableModels: LlmModelInfo[] = [];
  if (Array.isArray(data.available_models)) {
    for (const model of data.available_models) {
      if (!isObject(model)) continue;
      const id = typeof model.id === "string" ? model.id : "";
      if (!id) continue;

      availableModels.push({
        id,
        displayName: typeof model.display_name === "string" ? model.display_name : id,
        provider: typeof model.provider === "string" ? model.provider : "unknown",
        inputPrice: typeof model.input_price === "number" ? model.input_price : 0,
        outputPrice: typeof model.output_price === "number" ? model.output_price : 0,
        maxTokens: typeof model.max_tokens === "number" ? model.max_tokens : 4096,
      });
    }
  }

  const metadata = extractSettingsMetadata(response);

  return {
    ...metadata,
    config: {
      narrativesEnabled,
      narrativeFeedbackEnabled,
      primaryModel,
      temperature,
      maxTokens,
      effectiveEnabled,
      envFastpathEnabled,
      availableModels,
    },
  };
}

/**
 * IA Features update payload
 */
export interface IaFeaturesUpdatePayload {
  narratives_enabled?: boolean | null;
  primary_model?: string;
  temperature?: number;
  max_tokens?: number;
}
