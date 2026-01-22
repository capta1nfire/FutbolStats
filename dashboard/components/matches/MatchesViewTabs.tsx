"use client";

import { Calendar, History } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export type MatchesView = "upcoming" | "finished";

interface MatchesViewTabsProps {
  activeView: MatchesView;
  onViewChange: (view: MatchesView) => void;
}

/**
 * Tab selector for switching between Upcoming and Finished matches
 * Uses icons with tooltips for compact display
 */
export function MatchesViewTabs({ activeView, onViewChange }: MatchesViewTabsProps) {
  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex gap-1 p-1 bg-surface rounded-lg border border-border">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => onViewChange("upcoming")}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                activeView === "upcoming"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              )}
              aria-label="Upcoming matches"
              aria-pressed={activeView === "upcoming"}
            >
              <Calendar className="h-4 w-4" strokeWidth={1.5} />
              <span>Upcoming</span>
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>View upcoming matches</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => onViewChange("finished")}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                activeView === "finished"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              )}
              aria-label="Finished matches"
              aria-pressed={activeView === "finished"}
            >
              <History className="h-4 w-4" strokeWidth={1.5} />
              <span>Finished</span>
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>View finished matches with results</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}
