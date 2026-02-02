"use client";

import { OpsFastpathHealth } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Zap } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface FastpathCompactTileProps {
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
 * Fastpath Compact Tile
 *
 * Compact version for 2x2 grid. Shows key 60m stats.
 */
export function FastpathCompactTile({
  fastpath,
  className,
  isMockFallback = false,
}: FastpathCompactTileProps) {
  const isDegraded = !fastpath || isMockFallback;
  const displayStatus: ApiBudgetStatus = fastpath?.status ?? "degraded";

  const last60m = fastpath?.last_60m;
  const errorRate = last60m?.error_rate_pct ?? 0;

  return (
    <div
      className={cn(
        "bg-tile border border-border rounded-lg p-3 h-full flex flex-col",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Zap className="h-3.5 w-3.5 text-primary" />
          <h3 className="text-xs font-semibold text-foreground">Fastpath</h3>
        </div>
        <div className="flex items-center gap-1.5">
          {!fastpath?.enabled && (
            <span className="text-[9px] text-muted-foreground bg-muted px-1 rounded">off</span>
          )}
          <span className={cn("h-2 w-2 rounded-full", statusDot[displayStatus])} />
        </div>
      </div>

      {/* 60m stats - compact grid */}
      <div className="grid grid-cols-4 gap-1 flex-1">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="text-center cursor-default">
                <div className="text-lg font-bold text-foreground tabular-nums">
                  {last60m?.ok ?? 0}
                </div>
                <div className="text-[9px] text-muted-foreground">OK</div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>Successful narratives (60m)</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="text-center cursor-default">
                <div className="text-lg font-bold text-foreground tabular-nums">
                  {last60m?.error ?? 0}
                </div>
                <div className="text-[9px] text-muted-foreground">Err</div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>Errors (60m)</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="text-center cursor-default">
                <div className="text-lg font-bold text-foreground tabular-nums">
                  {last60m?.in_queue ?? 0}
                </div>
                <div className="text-[9px] text-muted-foreground">Q</div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>In queue</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="text-center cursor-default">
                <div className={cn(
                  "text-lg font-bold tabular-nums",
                  errorRate > 10 ? "text-[var(--status-error-text)]" : errorRate > 5 ? "text-[var(--status-warning-text)]" : "text-foreground"
                )}>
                  {errorRate.toFixed(0)}%
                </div>
                <div className="text-[9px] text-muted-foreground">Rate</div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>Error rate (60m)</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Bottom */}
      <div className="text-[10px] text-muted-foreground mt-2 pt-2 border-t border-border flex justify-between">
        <span>Pending: {fastpath?.pending_ready ?? 0}</span>
        {isDegraded && (
          <span className="text-orange-400">mock</span>
        )}
      </div>
    </div>
  );
}
