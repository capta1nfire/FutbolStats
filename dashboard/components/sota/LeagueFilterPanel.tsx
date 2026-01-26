"use client";

import { useMemo, useCallback } from "react";
import { ChevronLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { LeagueVisibilityState } from "@/lib/hooks/use-league-visibility";

/**
 * League option for the filter panel
 */
export interface LeagueOption {
  league_id: number;
  name: string;
}

interface LeagueFilterPanelProps {
  /** Whether the panel is visible */
  open: boolean;
  /** Available leagues to filter */
  leagues: LeagueOption[];
  /** Current visibility state */
  leagueVisibility: LeagueVisibilityState;
  /** Called when a league's visibility changes */
  onLeagueVisibilityChange: (leagueId: number, visible: boolean) => void;
  /** Called when "All" is toggled */
  onAllLeaguesChange: (visible: boolean, leagueIds: number[]) => void;
  /** Called when "Restore" is clicked */
  onRestore: () => void;
  /** Called when "Done" is clicked (closes panel) */
  onDone: () => void;
  /** Called when collapse button is clicked (collapses entire Left Rail) */
  onCollapse?: () => void;
  /** Additional class names */
  className?: string;
}

/**
 * League Filter Panel for Feature Coverage Matrix
 *
 * Similar to CustomizeColumnsPanel but for filtering leagues/competitions.
 * Appears in the 3rd column when Features view is active.
 */
export function LeagueFilterPanel({
  open,
  leagues,
  leagueVisibility,
  onLeagueVisibilityChange,
  onAllLeaguesChange,
  onRestore,
  onDone,
  onCollapse,
  className,
}: LeagueFilterPanelProps) {
  // Calculate "All" checkbox state
  const { allChecked, allIndeterminate } = useMemo(() => {
    const visibleCount = leagues.filter(
      (league) => leagueVisibility[String(league.league_id)] !== false
    ).length;
    const total = leagues.length;

    return {
      allChecked: visibleCount === total,
      allIndeterminate: visibleCount > 0 && visibleCount < total,
    };
  }, [leagues, leagueVisibility]);

  // Handle "All" checkbox
  const handleAllChange = useCallback(
    (checked: boolean) => {
      onAllLeaguesChange(
        checked,
        leagues.map((l) => l.league_id)
      );
    },
    [leagues, onAllLeaguesChange]
  );

  // Handle individual league checkbox
  const handleLeagueChange = useCallback(
    (leagueId: number, checked: boolean) => {
      onLeagueVisibilityChange(leagueId, checked);
    },
    [onLeagueVisibilityChange]
  );

  // Don't render if not open or no leagues
  if (!open || leagues.length === 0) {
    return null;
  }

  return (
    <aside
      className={cn(
        "w-[200px] border-r border-border bg-sidebar flex flex-col shrink-0 overflow-hidden",
        className
      )}
    >
      {/* Header */}
      <div className="h-12 flex items-center justify-between px-3">
        <h3 className="text-sm font-medium text-foreground">
          Filter Leagues
        </h3>
        {onCollapse && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onCollapse}
            className="h-8 w-8"
            aria-label="Collapse filters"
          >
            <ChevronLeft className="h-4 w-4" strokeWidth={1.5} />
          </Button>
        )}
      </div>

      {/* "All" checkbox */}
      <div className="px-3 py-2">
        <label className="flex items-center gap-2 cursor-pointer">
          <Checkbox
            checked={allIndeterminate ? "indeterminate" : allChecked}
            onCheckedChange={(checked) => handleAllChange(checked === true)}
            aria-label="Toggle all leagues"
          />
          <span className="text-sm text-foreground">All</span>
          <span className="text-xs text-muted-foreground">({leagues.length})</span>
        </label>
      </div>

      {/* League list with scroll */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="px-3 py-2 space-y-0.5">
          {leagues.map((league) => {
            const isVisible = leagueVisibility[String(league.league_id)] !== false;

            return (
              <label
                key={league.league_id}
                className="flex items-center gap-2 py-1.5 cursor-pointer hover:bg-accent/30 rounded px-1 -mx-1"
              >
                <Checkbox
                  checked={isVisible}
                  onCheckedChange={(checked) =>
                    handleLeagueChange(league.league_id, checked === true)
                  }
                  aria-label={`Show ${league.name}`}
                />
                <span className="text-sm text-foreground truncate" title={league.name}>
                  {league.name}
                </span>
              </label>
            );
          })}
        </div>
      </ScrollArea>

      {/* Footer: Restore + Done */}
      {/* Shadow uses ::before pseudo-element with gradient to render above ScrollArea content */}
      <div className="px-4 py-4 flex items-center justify-between bg-sidebar shrink-0 relative z-10 before:absolute before:left-0 before:right-0 before:bottom-full before:h-4 before:bg-gradient-to-t before:from-black/30 before:to-transparent before:pointer-events-none">
        <div className="h-8 flex items-center">
          <button
            onClick={onRestore}
            className="text-sm font-medium text-primary hover:text-primary-hover transition-colors"
          >
            Restore
          </button>
        </div>
        <div className="h-8 flex items-center">
          <button
            onClick={onDone}
            className="text-sm font-semibold text-primary hover:text-primary-hover transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </aside>
  );
}
