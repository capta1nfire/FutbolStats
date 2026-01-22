/**
 * OPS API Adapter
 *
 * Safe extraction and adaptation of data from /dashboard/ops.json
 * Designed to be resilient to partial or malformed responses.
 */

import { ApiBudget, ApiBudgetStatus } from "@/lib/types";

/**
 * Raw response type - intentionally loose since backend schema may evolve
 */
export type OpsResponse = unknown;

/**
 * Safely check if value is a non-null object
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Safely get nested property
 */
function getNestedValue(obj: unknown, ...keys: string[]): unknown {
  let current = obj;
  for (const key of keys) {
    if (!isObject(current)) return undefined;
    current = current[key];
  }
  return current;
}

/**
 * Extract budget object from ops response
 *
 * Expected structure: { data: { budget: {...} } }
 */
export function extractBudget(ops: OpsResponse): unknown | null {
  if (!isObject(ops)) return null;

  const budget = getNestedValue(ops, "data", "budget");
  if (!isObject(budget)) return null;

  return budget;
}

/**
 * Validate ApiBudgetStatus
 */
function isValidStatus(status: unknown): status is ApiBudgetStatus {
  return (
    status === "ok" ||
    status === "warning" ||
    status === "critical" ||
    status === "degraded"
  );
}

/**
 * Adapt raw budget object to ApiBudget type
 *
 * Only assumes these fields are stable (confirmed in spec):
 * - status, plan, plan_end, active
 * - requests_today, requests_limit, requests_remaining
 * - cached, cache_age_seconds
 * - tokens_reset_tz, tokens_reset_local_time, tokens_reset_at_la, tokens_reset_at_utc, tokens_reset_note
 *
 * Returns null if critical fields are missing or have wrong types.
 */
export function adaptApiBudget(raw: unknown): ApiBudget | null {
  if (!isObject(raw)) return null;

  // Required fields with type validation
  const status = raw.status;
  if (!isValidStatus(status)) return null;

  const plan = raw.plan;
  if (typeof plan !== "string") return null;

  const active = raw.active;
  if (typeof active !== "boolean") return null;

  const requests_today = raw.requests_today;
  if (typeof requests_today !== "number") return null;

  const requests_limit = raw.requests_limit;
  if (typeof requests_limit !== "number") return null;

  const requests_remaining = raw.requests_remaining;
  if (typeof requests_remaining !== "number") return null;

  // Optional but expected fields with defaults
  const cached = typeof raw.cached === "boolean" ? raw.cached : false;
  const cache_age_seconds =
    typeof raw.cache_age_seconds === "number" ? raw.cache_age_seconds : 0;

  // Optional string fields
  const plan_end =
    typeof raw.plan_end === "string" ? raw.plan_end : undefined;
  const tokens_reset_at_la =
    typeof raw.tokens_reset_at_la === "string"
      ? raw.tokens_reset_at_la
      : undefined;
  const tokens_reset_note =
    typeof raw.tokens_reset_note === "string"
      ? raw.tokens_reset_note
      : undefined;

  return {
    status,
    plan,
    plan_end,
    active,
    requests_today,
    requests_limit,
    requests_remaining,
    cached,
    cache_age_seconds,
    tokens_reset_at_la,
    tokens_reset_note,
  };
}

/**
 * Combined extraction and adaptation
 *
 * Returns ApiBudget if successful, null otherwise
 */
export function parseOpsBudget(ops: OpsResponse): ApiBudget | null {
  const rawBudget = extractBudget(ops);
  if (!rawBudget) return null;
  return adaptApiBudget(rawBudget);
}
