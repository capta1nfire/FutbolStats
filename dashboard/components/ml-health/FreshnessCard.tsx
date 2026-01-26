"use client";

import { Freshness } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";
import { DegradedAlert } from "./DegradedAlert";
import { Clock } from "lucide-react";

interface FreshnessCardProps {
  data: Freshness | null;
}

/**
 * Format hours for display
 */
function formatHours(hours: number | undefined): string {
  if (hours === undefined) return "—";
  return `${hours.toFixed(1)}h`;
}

/**
 * Freshness / Staleness Card
 *
 * Shows two sections:
 * - age_hours_now: Early warning real (odds/xg with p50/p95/max)
 * - lead_time_hours: Context (odds/xg with p50/p95/max)
 */
export function FreshnessCard({ data }: FreshnessCardProps) {
  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Clock className="h-4 w-4" />
          <span className="text-sm">Freshness data unavailable</span>
        </div>
      </div>
    );
  }

  const { age_hours_now, lead_time_hours, status, _degraded, _error } = data;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-foreground">Freshness / Staleness</h3>
        </div>
        <StatusBadge status={status} />
      </div>

      {/* Degraded alert */}
      {_degraded && (
        <div className="mb-4">
          <DegradedAlert error={_error} />
        </div>
      )}

      {/* Two sub-blocks */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Age Hours Now (Early Warning) */}
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs font-medium text-muted-foreground mb-3">
            Current Age (Early Warning)
          </p>
          <div className="space-y-3">
            {/* Odds */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted-foreground">Odds</span>
              </div>
              <div className="flex items-center gap-2">
                <StatPill label="p50" value={formatHours(age_hours_now?.odds?.p50)} />
                <StatPill label="p95" value={formatHours(age_hours_now?.odds?.p95)} highlight />
                <StatPill label="max" value={formatHours(age_hours_now?.odds?.max)} />
              </div>
            </div>
            {/* xG */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted-foreground">xG</span>
              </div>
              <div className="flex items-center gap-2">
                <StatPill label="p50" value={formatHours(age_hours_now?.xg?.p50)} />
                <StatPill label="p95" value={formatHours(age_hours_now?.xg?.p95)} highlight />
                <StatPill label="max" value={formatHours(age_hours_now?.xg?.max)} />
              </div>
            </div>
          </div>
        </div>

        {/* Lead Time Hours (Context) */}
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs font-medium text-muted-foreground mb-3">
            Lead Time (Context)
          </p>
          <div className="space-y-3">
            {/* Odds */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted-foreground">Odds</span>
              </div>
              <div className="flex items-center gap-2">
                <StatPill label="p50" value={formatHours(lead_time_hours?.odds?.p50)} />
                <StatPill label="p95" value={formatHours(lead_time_hours?.odds?.p95)} />
                <StatPill label="max" value={formatHours(lead_time_hours?.odds?.max)} />
              </div>
            </div>
            {/* xG */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-muted-foreground">xG</span>
              </div>
              <div className="flex items-center gap-2">
                <StatPill label="p50" value={formatHours(lead_time_hours?.xg?.p50)} />
                <StatPill label="p95" value={formatHours(lead_time_hours?.xg?.p95)} />
                <StatPill label="max" value={formatHours(lead_time_hours?.xg?.max)} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Small stat pill component
 */
function StatPill({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
        highlight
          ? "bg-yellow-500/10 border border-yellow-500/20"
          : "bg-muted/50"
      }`}
    >
      <span className="text-muted-foreground">{label}:</span>
      <span className={`tabular-nums font-medium ${highlight ? "text-yellow-400" : "text-foreground"}`}>
        {value}
      </span>
    </div>
  );
}
