"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { Sparkles } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";

interface OverviewDrawerLlmProps {
  tab: OverviewTab;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars -- tab prop reserved for future use
export function OverviewDrawerLlm({ tab }: OverviewDrawerLlmProps) {
  // Only summary tab for now
  return <LlmSummaryTab />;
}

function formatCost(usd: number): string {
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  if (usd < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}

function formatTokens(count: number): string {
  if (count < 1000) return count.toString();
  if (count < 1000000) return `${(count / 1000).toFixed(1)}K`;
  return `${(count / 1000000).toFixed(2)}M`;
}

function LlmSummaryTab() {
  const { llmCost, isLlmCostDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!llmCost || isLlmCostDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        LLM cost data unavailable
      </div>
    );
  }

  const statusColors = {
    ok: "text-[var(--status-success-text)]",
    warning: "text-[var(--status-warning-text)]",
    critical: "text-[var(--status-error-text)]",
    degraded: "text-orange-400",
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Sparkles className="h-4 w-4 text-primary" />
        <span>LLM Cost ({llmCost.provider})</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Status</span>
        <span className={`text-sm font-medium ${statusColors[llmCost.status]}`}>
          {llmCost.status.charAt(0).toUpperCase() + llmCost.status.slice(1)}
        </span>
      </div>

      {/* Cost breakdown */}
      <div className="bg-muted/30 rounded-lg p-4">
        <div className="text-3xl font-bold text-foreground tabular-nums mb-1">
          {formatCost(llmCost.cost_28d_usd)}
        </div>
        <p className="text-xs text-muted-foreground">28-day billing period</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-lg font-semibold text-foreground tabular-nums">
            {formatCost(llmCost.cost_24h_usd)}
          </div>
          <p className="text-xs text-muted-foreground">24h</p>
        </div>
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-lg font-semibold text-foreground tabular-nums">
            {formatCost(llmCost.cost_7d_usd)}
          </div>
          <p className="text-xs text-muted-foreground">7d</p>
        </div>
      </div>

      {/* Usage stats */}
      <div className="space-y-2 pt-2 border-t border-border">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Requests (28d)</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {llmCost.requests_28d.toLocaleString()}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Tokens (28d)</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {formatTokens(llmCost.tokens_in_28d + llmCost.tokens_out_28d)}
          </span>
        </div>
      </div>

      {/* Model info */}
      {llmCost.model && (
        <div className="flex items-center justify-between pt-2 border-t border-border">
          <span className="text-sm text-muted-foreground">Model</span>
          <span className="text-sm text-foreground">
            {llmCost.model.replace("gemini-", "").replace(/-/g, " ")}
          </span>
        </div>
      )}
    </div>
  );
}
