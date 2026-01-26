"use client";

import { OpsTitan, TitanStatus, TitanJobStatus } from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import { Brain, CheckCircle2, Clock, AlertTriangle } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface TitanCompactTileProps {
  titan: OpsTitan | null;
  className?: string;
  isMockFallback?: boolean;
}

const statusColors: Record<TitanStatus, string> = {
  ok: "text-green-400",
  building: "text-yellow-400",
  warn: "text-orange-400",
  error: "text-red-400",
  unavailable: "text-muted-foreground",
};

const statusBgColors: Record<TitanStatus, string> = {
  ok: "bg-green-500/20",
  building: "bg-yellow-500/20",
  warn: "bg-orange-500/20",
  error: "bg-red-500/20",
  unavailable: "bg-muted/50",
};

const statusLabels: Record<TitanStatus, string> = {
  ok: "Ready",
  building: "Building",
  warn: "Warning",
  error: "Error",
  unavailable: "N/A",
};

const jobStatusColors: Record<TitanJobStatus, string> = {
  success: "text-green-400",
  failed: "text-red-400",
  never_run: "text-muted-foreground",
};

const progressBarColors: Record<TitanStatus, string> = {
  ok: "bg-green-500",
  building: "bg-yellow-500",
  warn: "bg-orange-500",
  error: "bg-red-500",
  unavailable: "bg-muted",
};

/**
 * TITAN OMNISCIENCE Compact Tile
 *
 * Shows pilot gate progress, status, and job health.
 */
export function TitanCompactTile({
  titan,
  className,
  isMockFallback = false,
}: TitanCompactTileProps) {
  const isDegraded = !titan || isMockFallback;
  const status = titan?.status ?? "unavailable";
  const gate = titan?.gate;
  const job = titan?.job;

  const pctToFormal = gate?.pct_to_formal ?? 0;
  const nCurrent = gate?.n_current ?? 0;
  const nTarget = gate?.n_target_formal ?? 500;

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-3 h-full flex flex-col",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Brain className="h-3.5 w-3.5 text-primary" />
          <h3 className="text-xs font-semibold text-foreground">TITAN</h3>
        </div>
        <span
          className={cn(
            "px-1.5 py-0.5 text-[9px] font-medium rounded-full",
            statusBgColors[status],
            statusColors[status]
          )}
        >
          {statusLabels[status]}
        </span>
      </div>

      {/* Progress bar */}
      <div className="mb-2">
        <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
          <div
            className={cn("h-full transition-all duration-500", progressBarColors[status])}
            style={{ width: `${Math.min(pctToFormal, 100)}%` }}
          />
        </div>
      </div>

      {/* Gate progress */}
      <div className="flex-1 space-y-1.5">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center justify-between text-xs cursor-default">
                <span className="text-muted-foreground">Formal Gate</span>
                <span className={cn("font-medium tabular-nums", statusColors[status])}>
                  {nCurrent}/{nTarget}
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>Partidos evaluables para evaluación formal (N=500)</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {titan?.feature_matrix && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-between text-xs cursor-default">
                  <span className="text-muted-foreground">Matrix</span>
                  <span className="font-medium tabular-nums text-foreground">
                    {titan.feature_matrix.total_rows} rows
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Total filas en feature matrix</p>
                <p className="text-muted-foreground">
                  xG: {titan.feature_matrix.tier1b_pct.toFixed(1)}%
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {/* Job status */}
        {job && (
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Job</span>
            <span className={cn("font-medium", jobStatusColors[job.last_status])}>
              {job.last_status === "success" && <CheckCircle2 className="h-3 w-3 inline mr-0.5" />}
              {job.last_status === "failed" && <AlertTriangle className="h-3 w-3 inline mr-0.5" />}
              {job.last_status === "never_run" && <Clock className="h-3 w-3 inline mr-0.5" />}
              {job.last_status.replace("_", " ")}
            </span>
          </div>
        )}
      </div>

      {/* Note */}
      {titan?.note && (
        <div className="text-[10px] text-muted-foreground mt-2 pt-2 border-t border-border truncate">
          {titan.note}
        </div>
      )}

      {isDegraded && (
        <div className="text-[10px] text-orange-400 mt-2 pt-2 border-t border-border">
          mock
        </div>
      )}
    </div>
  );
}
