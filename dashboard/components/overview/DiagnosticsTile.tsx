"use client";

import { OpsShadowMode, OpsSensorB } from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import { FlaskConical, Activity } from "lucide-react";

interface DiagnosticsTileProps {
  shadowMode: OpsShadowMode | null;
  sensorB: OpsSensorB | null;
  className?: string;
}

/**
 * Format percentage delta with sign
 */
function formatDelta(delta: number, suffix = "%"): string {
  const sign = delta >= 0 ? "+" : "";
  return `${sign}${(delta * 100).toFixed(1)}${suffix}`;
}

/**
 * Get color for recommendation status
 */
function getRecommendationColor(status: string): string {
  if (status === "GO") return "text-[var(--status-success-text)]";
  if (status === "NO_GO") return "text-[var(--status-error-text)]";
  return "text-[var(--status-warning-text)]";
}

/**
 * Get color for sensor state
 */
function getSensorStateColor(state: string): string {
  if (state === "NOMINAL") return "text-[var(--status-success-text)]";
  if (state === "CALIBRATING") return "text-[var(--status-warning-text)]";
  if (state === "OVERFITTING_SUSPECTED") return "text-orange-400";
  return "text-muted-foreground";
}

/**
 * Diagnostics Tile
 *
 * Displays Shadow Mode and Sensor B diagnostics.
 * Marked as "Diagnostic / not prod" - does not affect overall ops status.
 */
export function DiagnosticsTile({
  shadowMode,
  sensorB,
  className,
}: DiagnosticsTileProps) {
  const hasShadow = shadowMode !== null;
  const hasSensor = sensorB !== null;

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-4",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Diagnostics</h3>
        </div>
        <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground border border-border">
          Not Prod
        </span>
      </div>

      <div className="space-y-4">
        {/* Shadow Mode Section */}
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <Activity className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-xs font-medium text-foreground">Shadow Mode</span>
            {shadowMode?.state.enabled ? (
              <span className="px-1 py-0.5 text-[9px] rounded bg-[var(--status-success-bg)] text-[var(--status-success-text)] border border-[var(--status-success-border)]">
                ON
              </span>
            ) : (
              <span className="px-1 py-0.5 text-[9px] rounded bg-muted text-muted-foreground border border-border">
                OFF
              </span>
            )}
          </div>

          {hasShadow ? (
            <div className="pl-5 space-y-1">
              {/* Recommendation */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Recommendation:</span>
                <span className={cn("font-medium", getRecommendationColor(shadowMode.recommendation.status))}>
                  {shadowMode.recommendation.status}
                </span>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-x-4 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Brier:</span>
                  <span className={cn(
                    "tabular-nums",
                    shadowMode.metrics.delta_brier < 0 ? "text-[var(--status-success-text)]" : "text-[var(--status-error-text)]"
                  )}>
                    {formatDelta(shadowMode.metrics.delta_brier, "")}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Accuracy:</span>
                  <span className={cn(
                    "tabular-nums",
                    shadowMode.metrics.delta_accuracy > 0 ? "text-[var(--status-success-text)]" : "text-[var(--status-error-text)]"
                  )}>
                    {formatDelta(shadowMode.metrics.delta_accuracy)}
                  </span>
                </div>
              </div>

              {/* Counts */}
              <div className="text-xs text-muted-foreground">
                {shadowMode.counts.shadow_predictions_evaluated} evaluated / {shadowMode.counts.shadow_predictions_pending} pending
              </div>
            </div>
          ) : (
            <div className="pl-5 text-xs text-muted-foreground/50">No data</div>
          )}
        </div>

        {/* Sensor B Section */}
        <div className="pt-3 border-t border-border">
          <div className="flex items-center gap-1.5 mb-2">
            <Activity className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-xs font-medium text-foreground">Sensor B</span>
            {sensorB?.is_ready ? (
              <span className="px-1 py-0.5 text-[9px] rounded bg-[var(--status-success-bg)] text-[var(--status-success-text)] border border-[var(--status-success-border)]">
                Ready
              </span>
            ) : (
              <span className="px-1 py-0.5 text-[9px] rounded bg-muted text-muted-foreground border border-border">
                Not Ready
              </span>
            )}
          </div>

          {hasSensor ? (
            <div className="pl-5 space-y-1">
              {/* State */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">State:</span>
                <span className={cn("font-medium", getSensorStateColor(sensorB.state))}>
                  {sensorB.state.replace(/_/g, " ")}
                </span>
              </div>

              {/* Signal score */}
              {sensorB.signal_score !== undefined && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Signal:</span>
                  <span className={cn(
                    "tabular-nums",
                    sensorB.signal_score > 0 ? "text-[var(--status-success-text)]" : sensorB.signal_score < -3 ? "text-[var(--status-error-text)]" : "text-[var(--status-warning-text)]"
                  )}>
                    {sensorB.signal_score.toFixed(2)}
                  </span>
                </div>
              )}

              {/* Brier delta */}
              {sensorB.delta_brier !== undefined && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Brier delta:</span>
                  <span className={cn(
                    "tabular-nums",
                    sensorB.delta_brier < 0 ? "text-[var(--status-success-text)]" : "text-[var(--status-error-text)]"
                  )}>
                    {sensorB.delta_brier > 0 ? "+" : ""}{sensorB.delta_brier.toFixed(3)}
                  </span>
                </div>
              )}

              {/* Note */}
              {sensorB.note && (
                <div className="text-[10px] text-muted-foreground/70 italic">
                  {sensorB.note}
                </div>
              )}
            </div>
          ) : (
            <div className="pl-5 text-xs text-muted-foreground/50">No data</div>
          )}
        </div>
      </div>
    </div>
  );
}
