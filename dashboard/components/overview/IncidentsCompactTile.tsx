"use client";

import { ActiveIncident } from "@/lib/types";
import { cn } from "@/lib/utils";
import { AlertTriangle, AlertCircle, CheckCircle2 } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface IncidentsCompactTileProps {
  incidents: ActiveIncident[];
  className?: string;
  isMockFallback?: boolean;
}

const severityColors = {
  critical: "bg-[var(--status-error-text)]",
  warning: "bg-[var(--status-warning-text)]",
  info: "bg-[var(--status-info-text)]",
};

/**
 * Incidents Compact Tile
 *
 * Compact version for grid layout. Shows incident count by severity.
 */
export function IncidentsCompactTile({
  incidents,
  className,
  isMockFallback = false,
}: IncidentsCompactTileProps) {
  const criticalCount = incidents.filter((i) => i.severity === "critical").length;
  const warningCount = incidents.filter((i) => i.severity === "warning").length;
  const infoCount = incidents.filter((i) => i.severity === "info").length;
  const totalCount = incidents.length;

  const hasIncidents = totalCount > 0;
  const hasCritical = criticalCount > 0;

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
          <AlertTriangle className="h-3.5 w-3.5 text-primary" />
          <h3 className="text-xs font-semibold text-foreground">Incidents</h3>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground">{totalCount}</span>
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              hasCritical
                ? "bg-[var(--status-error-text)]"
                : hasIncidents
                ? "bg-[var(--status-warning-text)]"
                : "bg-[var(--status-success-text)]"
            )}
          />
        </div>
      </div>

      {/* Content */}
      <div className="space-y-1.5 flex-1">
        {hasIncidents ? (
          <>
            {criticalCount > 0 && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-1.5 cursor-default">
                      <span className={cn("h-2 w-2 rounded-full", severityColors.critical)} />
                      <span className="text-xs text-foreground">Critical</span>
                      <span className="text-[10px] text-muted-foreground tabular-nums">
                        {criticalCount}
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>{criticalCount} critical incident{criticalCount !== 1 ? "s" : ""}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            {warningCount > 0 && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-1.5 cursor-default">
                      <span className={cn("h-2 w-2 rounded-full", severityColors.warning)} />
                      <span className="text-xs text-foreground">Warning</span>
                      <span className="text-[10px] text-muted-foreground tabular-nums">
                        {warningCount}
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>{warningCount} warning{warningCount !== 1 ? "s" : ""}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            {infoCount > 0 && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-1.5 cursor-default">
                      <span className={cn("h-2 w-2 rounded-full", severityColors.info)} />
                      <span className="text-xs text-foreground">Info</span>
                      <span className="text-[10px] text-muted-foreground tabular-nums">
                        {infoCount}
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    <p>{infoCount} info</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </>
        ) : (
          <div className="flex items-center gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-[var(--status-success-text)]" />
            <span className="text-xs text-muted-foreground">All clear</span>
          </div>
        )}
      </div>

      {/* Bottom */}
      {isMockFallback && (
        <div className="text-[10px] text-orange-400 mt-2 pt-2 border-t border-border">
          mock
        </div>
      )}
    </div>
  );
}
