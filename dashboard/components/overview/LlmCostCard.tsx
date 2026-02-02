"use client";

import { OpsLlmCost } from "@/lib/api/ops";
import { ApiBudgetStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Info, Sparkles } from "lucide-react";
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
  ok: "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]",
  warning: "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]",
  critical: "bg-[var(--status-error-bg)] text-[var(--status-error-text)] border-[var(--status-error-border)]",
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
 * Format price for display (e.g., 0.1 -> "$0.10")
 */
function formatPrice(price: number): string {
  if (price < 1) return `$${price.toFixed(2)}`;
  return `$${price.toFixed(2)}`;
}

/**
 * Format model name for display (e.g., "gemini-2.0-flash" -> "2.0 flash")
 */
function formatModelName(model: string): string {
  return model.replace("gemini-", "").replace(/-/g, " ");
}

/**
 * Build pricing tooltip content from dynamic data
 *
 * Priority:
 * 1. model_usage_28d - show top model + breakdown (when available)
 * 2. model_pricing - show pricing table
 * 3. pricing_input/output_per_1m - show current model pricing
 * 4. note - show note text
 * 5. fallback message
 */
function buildPricingTooltip(llmCost: OpsLlmCost | null): React.ReactNode {
  // If we have model usage breakdown (28d), show it
  if (llmCost?.model_usage_28d && llmCost.model_usage_28d.length > 0) {
    const usage = llmCost.model_usage_28d;
    // Sort by cost descending to get top model first
    const sorted = [...usage].sort((a, b) => b.cost_usd - a.cost_usd);
    const topModels = sorted.slice(0, 3); // Show up to 3 models

    return (
      <div className="text-xs leading-relaxed space-y-1.5 max-w-[220px]">
        {llmCost.pricing_source && (
          <div className="text-muted-foreground text-[10px]">
            Source: {llmCost.pricing_source}
          </div>
        )}
        <div className="font-medium">Usage (28d)</div>
        {topModels.map((m, i) => (
          <div key={i} className="flex justify-between gap-3">
            <span className="truncate">{formatModelName(m.model)}</span>
            <span className="tabular-nums shrink-0">
              {formatCost(m.cost_usd)} · {m.requests}req
            </span>
          </div>
        ))}
        {llmCost.model_pricing && (
          <div className="pt-1.5 border-t border-border/50 text-muted-foreground">
            <div className="text-[10px]">Pricing (per 1M tokens)</div>
            {Object.entries(llmCost.model_pricing).slice(0, 2).map(([model, p]) => (
              <div key={model} className="flex justify-between gap-2">
                <span className="truncate">{formatModelName(model)}</span>
                <span className="tabular-nums shrink-0">
                  {formatPrice(p.input)}/{formatPrice(p.output)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Fallback: If we have model_pricing from backend, use it
  if (llmCost?.model_pricing) {
    const entries = Object.entries(llmCost.model_pricing);
    // Group by price tier (flash vs pro)
    const flashModels = entries.filter(([m]) => m.includes("flash"));
    const proModels = entries.filter(([m]) => m.includes("pro"));

    const lines: string[] = [];

    if (flashModels.length > 0) {
      const [, pricing] = flashModels[0];
      lines.push(`Flash: ${formatPrice(pricing.input)}/${formatPrice(pricing.output)}`);
    }

    if (proModels.length > 0) {
      const [, pricing] = proModels[0];
      lines.push(`Pro: ${formatPrice(pricing.input)}/${formatPrice(pricing.output)}`);
    }

    if (lines.length > 0) {
      return (
        <div className="text-xs leading-relaxed">
          {llmCost.pricing_source && (
            <div className="text-muted-foreground text-[10px] mb-1">
              Source: {llmCost.pricing_source}
            </div>
          )}
          <span className="font-medium">Pricing (per 1M tokens)</span><br />
          {lines.map((line, i) => (
            <span key={i}>{line}{i < lines.length - 1 && <br />}</span>
          ))}
        </div>
      );
    }
  }

  // Fallback: use current model pricing if available
  if (llmCost?.pricing_input_per_1m !== undefined && llmCost?.pricing_output_per_1m !== undefined) {
    const modelName = llmCost.model ? formatModelName(llmCost.model) : "Current";
    return (
      <div className="text-xs leading-relaxed">
        {llmCost.pricing_source && (
          <div className="text-muted-foreground text-[10px] mb-1">
            Source: {llmCost.pricing_source}
          </div>
        )}
        <span className="font-medium">Pricing (per 1M tokens)</span><br />
        {modelName}: {formatPrice(llmCost.pricing_input_per_1m)}/{formatPrice(llmCost.pricing_output_per_1m)}
      </div>
    );
  }

  // Final fallback: show note if available
  if (llmCost?.note) {
    return <p className="text-xs leading-relaxed max-w-[200px]">{llmCost.note}</p>;
  }

  // No pricing info available
  return <p className="text-xs text-muted-foreground">Pricing info unavailable</p>;
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
        "bg-tile border border-border rounded-lg p-4 overflow-hidden",
        className
      )}
    >
      {/* Header: Title + Info + Status */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
          <Sparkles className="h-4 w-4 text-primary" />
          LLM Cost
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3.5 w-3.5 text-primary cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="right" align="start">
                {buildPricingTooltip(llmCost)}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </h3>
        {displayStatus === "ok" ? (
          <span className="h-2.5 w-2.5 rounded-full bg-[var(--status-success-text)]" title="OK" />
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
