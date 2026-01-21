"use client";

import { useMemo } from "react";
import { ApiBudget, ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Info, AlertTriangle, ExternalLink } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ApiBudgetCardProps {
  budget: ApiBudget;
  className?: string;
}

const statusColors: Record<ApiBudgetStatus, string> = {
  ok: "bg-green-500/20 text-green-400 border-green-500/30",
  warning: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  critical: "bg-red-500/20 text-red-400 border-red-500/30",
  degraded: "bg-orange-500/20 text-orange-400 border-orange-500/30",
};

const statusLabels: Record<ApiBudgetStatus, string> = {
  ok: "OK",
  warning: "Warning",
  critical: "Critical",
  degraded: "Degraded",
};

const progressColors: Record<ApiBudgetStatus, string> = {
  ok: "bg-green-500",
  warning: "bg-yellow-500",
  critical: "bg-red-500",
  degraded: "bg-orange-500",
};

/**
 * Format time remaining until reset
 */
function formatTimeUntilReset(resetAtIso: string | undefined): string {
  if (!resetAtIso) {
    return "~16:00 LA";
  }

  const resetTime = new Date(resetAtIso).getTime();
  const now = Date.now();
  let diffMs = resetTime - now;

  // If reset time has passed, assume it's tomorrow
  if (diffMs < 0) {
    diffMs += 24 * 60 * 60 * 1000;
  }

  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

/**
 * Format cache age
 */
function formatCacheAge(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ago`;
}

/**
 * API Budget Card
 *
 * Displays API rate limit information with progress bar and countdown.
 * Designed for the Overview side column.
 */
export function ApiBudgetCard({ budget, className }: ApiBudgetCardProps) {
  const usedPct = useMemo(() => {
    if (!budget.requests_limit || budget.requests_limit === 0) return 0;
    return Math.round((budget.requests_today / budget.requests_limit) * 100);
  }, [budget.requests_today, budget.requests_limit]);

  const isInactive = !budget.active;
  const isDegraded = budget.status === "degraded" || isInactive;
  const isStale = budget.cache_age_seconds > 600; // 10 minutes
  const showCached = budget.cached || budget.cache_age_seconds > 60;
  const hasLimit = budget.requests_limit && budget.requests_limit > 0;

  // Determine effective status for display
  const displayStatus: ApiBudgetStatus = isInactive ? "degraded" : budget.status;

  // Format plan expiration
  const planEndFormatted = useMemo(() => {
    if (!budget.plan_end) return null;
    try {
      const date = new Date(budget.plan_end);
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return null;
    }
  }, [budget.plan_end]);

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-4",
        className
      )}
    >
      {/* Header: Title + Status Pill */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground">API Budget</h3>
        <span
          className={cn(
            "px-2 py-0.5 text-xs font-medium rounded-full border",
            statusColors[displayStatus]
          )}
        >
          {isInactive ? "Inactive" : statusLabels[displayStatus]}
        </span>
      </div>

      {/* Subtitle: Plan info */}
      <div className="text-xs text-muted-foreground mb-4">
        Plan {budget.plan}
        {planEndFormatted && (
          <span className="ml-1">
            &bull; Expires {planEndFormatted}
          </span>
        )}
      </div>

      {/* Degraded/Inactive state */}
      {isDegraded && (
        <div className="flex items-center gap-2 mb-4 p-2 rounded bg-orange-500/10 border border-orange-500/20">
          <AlertTriangle className="h-4 w-4 text-orange-400 shrink-0" />
          <span className="text-xs text-orange-400">
            {isInactive ? "API is currently inactive" : "API service is degraded"}
          </span>
          <button
            className="ml-auto text-xs text-orange-400 hover:text-orange-300 flex items-center gap-1"
            onClick={() => {/* Placeholder action */}}
          >
            Check status
            <ExternalLink className="h-3 w-3" />
          </button>
        </div>
      )}

      {/* Main Metric */}
      <div className="mb-2">
        <div className="text-2xl font-bold text-foreground tabular-nums">
          {budget.requests_today.toLocaleString()}
          {hasLimit && (
            <span className="text-muted-foreground font-normal">
              {" / "}
              {budget.requests_limit.toLocaleString()}
            </span>
          )}
        </div>
        <div className="text-xs text-muted-foreground">
          Remaining: {budget.requests_remaining.toLocaleString()}
        </div>
      </div>

      {/* Progress Bar */}
      {hasLimit ? (
        <div className="mb-4">
          <div className="h-2 bg-background rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                progressColors[displayStatus]
              )}
              style={{ width: `${Math.min(usedPct, 100)}%` }}
            />
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {usedPct}% used
          </div>
        </div>
      ) : (
        <div className="mb-4 text-xs text-muted-foreground italic">
          Limit unavailable
        </div>
      )}

      {/* Reset Info */}
      <div className="flex items-center justify-between text-xs">
        <div className="text-muted-foreground">
          Resets in:{" "}
          <span className="text-foreground font-medium">
            {formatTimeUntilReset(budget.tokens_reset_at_la)}
          </span>
        </div>
        {budget.tokens_reset_note && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-muted-foreground hover:text-foreground">
                  <Info className="h-3.5 w-3.5" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-[200px]">
                <p className="text-xs">{budget.tokens_reset_note}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {/* Cache freshness */}
      {showCached && (
        <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border">
          <span className="text-xs text-muted-foreground">
            Cached: {formatCacheAge(budget.cache_age_seconds)}
          </span>
          {isStale && (
            <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
              stale
            </span>
          )}
        </div>
      )}
    </div>
  );
}
