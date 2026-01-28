"use client";

import { OpsFastpathHealth } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Zap } from "lucide-react";

interface FastpathHealthTileProps {
  fastpath: OpsFastpathHealth | null;
  className?: string;
  isMockFallback?: boolean;
}

const statusDot: Record<ApiBudgetStatus, string> = {
  ok: "bg-[var(--status-success-text)]",
  warning: "bg-[var(--status-warning-text)]",
  critical: "bg-[var(--status-error-text)]",
  degraded: "bg-orange-500",
};

/**
 * Fastpath Health Tile
 *
 * Displays LLM narrative generation pipeline health.
 */
export function FastpathHealthTile({
  fastpath,
  className,
  isMockFallback = false,
}: FastpathHealthTileProps) {
  const isDegraded = !fastpath || isMockFallback;
  const displayStatus: ApiBudgetStatus = fastpath?.status ?? "degraded";

  const last60m = fastpath?.last_60m;
  const errorRate = last60m?.error_rate_pct ?? 0;

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
          <Zap className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Fastpath</h3>
        </div>
        <div className="flex items-center gap-2">
          {!fastpath?.enabled && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground border border-border">
              Disabled
            </span>
          )}
          {displayStatus === "ok" ? (
            <span className="h-2.5 w-2.5 rounded-full bg-[var(--status-success-text)]" title="OK" />
          ) : (
            <span className={cn("h-2.5 w-2.5 rounded-full", statusDot[displayStatus])} />
          )}
        </div>
      </div>

      {/* Status reason */}
      {fastpath?.status_reason && (
        <div className="text-xs text-muted-foreground mb-3">
          {fastpath.status_reason}
        </div>
      )}

      {/* 60m stats grid */}
      <div className="grid grid-cols-4 gap-2 mb-3">
        <div className="text-center">
          <div className="text-lg font-bold text-foreground tabular-nums">
            {last60m?.ok ?? 0}
          </div>
          <div className="text-[10px] text-muted-foreground">OK</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-foreground tabular-nums">
            {last60m?.error ?? 0}
          </div>
          <div className="text-[10px] text-muted-foreground">Errors</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-foreground tabular-nums">
            {last60m?.in_queue ?? 0}
          </div>
          <div className="text-[10px] text-muted-foreground">Queue</div>
        </div>
        <div className="text-center">
          <div className={cn(
            "text-lg font-bold tabular-nums",
            errorRate > 10 ? "text-[var(--status-error-text)]" : errorRate > 5 ? "text-[var(--status-warning-text)]" : "text-foreground"
          )}>
            {errorRate.toFixed(1)}%
          </div>
          <div className="text-[10px] text-muted-foreground">Err Rate</div>
        </div>
      </div>

      {/* Pending ready */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Pending ready:</span>
        <span className="text-foreground tabular-nums">{fastpath?.pending_ready ?? 0}</span>
      </div>

      {/* Last tick */}
      {fastpath?.minutes_since_tick !== undefined && (
        <div className="flex items-center justify-between text-xs text-muted-foreground mt-1">
          <span>Last tick:</span>
          <span className="text-foreground tabular-nums">
            {fastpath.minutes_since_tick < 1 ? "<1m ago" : `${Math.round(fastpath.minutes_since_tick)}m ago`}
          </span>
        </div>
      )}

      {/* Top error codes */}
      {fastpath?.top_error_codes_60m && Object.keys(fastpath.top_error_codes_60m).length > 0 && (
        <div className="mt-3 pt-3 border-t border-border">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
            Top Errors (60m)
          </div>
          <div className="space-y-0.5">
            {Object.entries(fastpath.top_error_codes_60m).slice(0, 3).map(([code, count]) => (
              <div key={code} className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground truncate">{code}</span>
                <span className="text-[var(--status-error-text)] tabular-nums">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Degraded indicator */}
      {isDegraded && (
        <div className="mt-3 pt-3 border-t border-border">
          <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground border border-border">
            Degraded (mock)
          </span>
        </div>
      )}
    </div>
  );
}
