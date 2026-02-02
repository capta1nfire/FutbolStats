"use client";

import { OpsMovement } from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import { ArrowUpDown } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface MovementSummaryTileProps {
  movement: OpsMovement | null;
  className?: string;
  isMovementDegraded?: boolean;
}

/**
 * Movement Summary Tile
 *
 * Shows lineup and market movement metrics (24h).
 */
export function MovementSummaryTile({
  movement,
  className,
  isMovementDegraded = false,
}: MovementSummaryTileProps) {
  const isDegraded = !movement || isMovementDegraded;

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
          <ArrowUpDown className="h-3.5 w-3.5 text-primary" />
          <h3 className="text-xs font-semibold text-foreground">Movement (24h)</h3>
        </div>
        {isDegraded && (
          <span className="text-[9px] text-orange-400 bg-orange-500/10 px-1 rounded">mock</span>
        )}
      </div>

      {/* Movement metrics */}
      {!isDegraded && movement ? (
        <div className="grid grid-cols-2 gap-3 flex-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="cursor-default">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">
                    Lineup
                  </div>
                  <div className="text-xl font-bold tabular-nums text-foreground">
                    {movement.lineup_movement_24h}
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Lineup changes detected in last 24h</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="cursor-default">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">
                    Market
                  </div>
                  <div className="text-xl font-bold tabular-nums text-foreground">
                    {movement.market_movement_24h}
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Market movements detected in last 24h</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      ) : (
        <div className="text-xs text-muted-foreground">Movement data unavailable</div>
      )}
    </div>
  );
}
