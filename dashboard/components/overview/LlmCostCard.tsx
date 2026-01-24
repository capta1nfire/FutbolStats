"use client";

import { OpsLlmCost } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Sparkles } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface LlmCostCardProps {
  llmCost: OpsLlmCost | null;
  className?: string;
  isMockFallback?: boolean;
}

const statusColors: Record<ApiBudgetStatus, string> = {
  ok: "bg-green-500/20 text-green-400 border-green-500/30",
  warning: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  critical: "bg-red-500/20 text-red-400 border-red-500/30",
  degraded: "bg-orange-500/20 text-orange-400 border-orange-500/30",
};

/**
 * Format USD cost with appropriate precision
 */
function formatCost(usd: number): string {
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  if (usd < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}

/**
 * Format token count with K suffix
 */
function formatTokens(count: number): string {
  if (count < 1000) return count.toString();
  return `${(count / 1000).toFixed(1)}K`;
}

/**
 * LLM Cost Card (compact)
 *
 * Displays LLM usage costs for the left rail.
 */
export function LlmCostCard({
  llmCost,
  className,
  isMockFallback = false,
}: LlmCostCardProps) {
  const isDegraded = !llmCost || isMockFallback;
  const displayStatus: ApiBudgetStatus = llmCost?.status ?? "degraded";

  return (
    <div
      className={cn(
        "bg-surface border border-border rounded-lg p-4 overflow-hidden",
        className
      )}
    >
      {/* Header: Title + Status */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
          <Sparkles className="h-4 w-4 text-primary" />
          LLM Cost
        </h3>
        {displayStatus === "ok" ? (
          <span className="h-2.5 w-2.5 rounded-full bg-green-500" title="OK" />
        ) : (
          <span
            className={cn(
              "px-2 py-0.5 text-xs font-medium rounded-full border",
              statusColors[displayStatus]
            )}
          >
            {displayStatus.charAt(0).toUpperCase() + displayStatus.slice(1)}
          </span>
        )}
      </div>

      {/* Provider */}
      {llmCost?.provider && (
        <div className="text-xs text-muted-foreground mb-3">
          {llmCost.provider.charAt(0).toUpperCase() + llmCost.provider.slice(1)}
        </div>
      )}

      {/* Primary cost metric: 28d (aligns with Google billing) */}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="mb-3 cursor-help">
              <div className="text-2xl font-bold text-foreground tabular-nums">
                {formatCost(llmCost?.cost_28d_usd ?? 0)}
              </div>
              <div className="text-[10px] text-muted-foreground">
                28d billing period
              </div>
            </div>
          </TooltipTrigger>
          <TooltipContent side="top">
            <p>28d aligns with Google AI Studio billing window</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Secondary cost metrics: 24h and 7d */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div>
          <div className="text-sm font-medium text-foreground tabular-nums">
            {formatCost(llmCost?.cost_24h_usd ?? 0)}
          </div>
          <div className="text-[10px] text-muted-foreground">24h</div>
        </div>
        <div>
          <div className="text-sm font-medium text-foreground tabular-nums">
            {formatCost(llmCost?.cost_7d_usd ?? 0)}
          </div>
          <div className="text-[10px] text-muted-foreground">7d</div>
        </div>
      </div>

      {/* Requests & tokens (28d to match billing) */}
      <div className="text-xs text-muted-foreground space-y-0.5">
        <div className="flex justify-between">
          <span>Requests 28d:</span>
          <span className="text-foreground tabular-nums">{llmCost?.requests_28d ?? 0}</span>
        </div>
        <div className="flex justify-between">
          <span>Tokens 28d:</span>
          <span className="text-foreground tabular-nums">
            {formatTokens((llmCost?.tokens_in_28d ?? 0) + (llmCost?.tokens_out_28d ?? 0))}
          </span>
        </div>
      </div>

      {/* Note */}
      {llmCost?.note && (
        <div className="text-[10px] text-muted-foreground/70 italic mt-2 pt-2 border-t border-border">
          {llmCost.note}
        </div>
      )}

      {/* Degraded indicator */}
      {isDegraded && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border cursor-help">
                <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground border border-border">
                  Degraded (mock)
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p>LLM cost data unavailable.</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}
