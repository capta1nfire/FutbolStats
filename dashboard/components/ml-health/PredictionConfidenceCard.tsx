"use client";

import { PredictionConfidence } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";
import { DegradedAlert } from "./DegradedAlert";
import { TrendingUp, Award } from "lucide-react";

interface PredictionConfidenceCardProps {
  data: PredictionConfidence | null;
}

/**
 * Prediction Confidence Card
 *
 * Shows:
 * - entropy: avg, p50, p95 (optional p25/p75)
 * - tier_distribution: gold/silver/copper
 * - sample_n + window_days
 */
export function PredictionConfidenceCard({ data }: PredictionConfidenceCardProps) {
  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <TrendingUp className="h-4 w-4" />
          <span className="text-sm">Prediction Confidence unavailable</span>
        </div>
      </div>
    );
  }

  const { entropy, tier_distribution, sample_n, window_days, _degraded, _error } = data;
  const gold = tier_distribution?.gold ?? 0;
  const silver = tier_distribution?.silver ?? 0;
  const copper = tier_distribution?.copper ?? 0;
  const totalTiers = gold + silver + copper;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-foreground">Prediction Confidence</h3>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{(sample_n ?? 0).toLocaleString()} samples</span>
          <span>|</span>
          <span>{window_days ?? 0}d window</span>
        </div>
      </div>

      {/* Degraded alert */}
      {_degraded && (
        <div className="mb-4">
          <DegradedAlert error={_error} />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Entropy Stats */}
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs font-medium text-muted-foreground mb-3">
            Entropy Distribution
          </p>
          <div className="space-y-2">
            <EntropyRow label="Average" value={entropy?.avg} />
            <EntropyRow label="p25" value={entropy?.p25} muted />
            <EntropyRow label="p50 (Median)" value={entropy?.p50} />
            <EntropyRow label="p75" value={entropy?.p75} muted />
            <EntropyRow label="p95" value={entropy?.p95} />
          </div>
        </div>

        {/* Tier Distribution */}
        <div className="p-3 bg-background rounded border border-border">
          <p className="text-xs font-medium text-muted-foreground mb-3">
            Tier Distribution
          </p>
          <div className="space-y-3">
            <TierBar
              label="Gold"
              count={gold}
              total={totalTiers}
              color="bg-yellow-400"
            />
            <TierBar
              label="Silver"
              count={silver}
              total={totalTiers}
              color="bg-gray-400"
            />
            <TierBar
              label="Copper"
              count={copper}
              total={totalTiers}
              color="bg-orange-600"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Entropy stat row
 */
function EntropyRow({
  label,
  value,
  muted = false,
}: {
  label: string;
  value: number | undefined;
  muted?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className={`text-xs ${muted ? "text-muted-foreground/60" : "text-muted-foreground"}`}>
        {label}
      </span>
      <span className={`text-sm tabular-nums font-medium ${muted ? "text-foreground/60" : "text-foreground"}`}>
        {value !== undefined ? value.toFixed(3) : "—"}
      </span>
    </div>
  );
}

/**
 * Tier distribution bar
 */
function TierBar({
  label,
  count,
  total,
  color,
}: {
  label: string;
  count: number;
  total: number;
  color: string;
}) {
  const pct = total > 0 ? (count / total) * 100 : 0;

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Award className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
        <span className="text-xs tabular-nums text-foreground">
          {count.toLocaleString()} ({pct.toFixed(0)}%)
        </span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
