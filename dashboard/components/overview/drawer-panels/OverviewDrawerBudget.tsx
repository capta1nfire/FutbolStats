"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { CreditCard, Clock } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";

interface OverviewDrawerBudgetProps {
  tab: OverviewTab;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars -- tab prop reserved for future use
export function OverviewDrawerBudget({ tab }: OverviewDrawerBudgetProps) {
  // Only summary tab for now
  return <BudgetSummaryTab />;
}

function BudgetSummaryTab() {
  const { budget, isBudgetDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!budget || isBudgetDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Budget data unavailable
      </div>
    );
  }

  const statusColors = {
    ok: "text-[var(--status-success-text)]",
    warning: "text-[var(--status-warning-text)]",
    critical: "text-[var(--status-error-text)]",
    degraded: "text-orange-400",
  };

  const usagePercent = budget.requests_limit > 0
    ? (budget.requests_today / budget.requests_limit) * 100
    : 0;

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <CreditCard className="h-4 w-4 text-primary" />
        <span>API-Football Budget</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Status</span>
        <span className={`text-sm font-medium ${statusColors[budget.status]}`}>
          {budget.status.charAt(0).toUpperCase() + budget.status.slice(1)}
        </span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Plan</span>
        <span className="text-sm font-medium text-foreground">
          {budget.plan}
        </span>
      </div>

      {/* Usage */}
      <div className="bg-muted/30 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-foreground">Requests Today</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {budget.requests_today.toLocaleString()} / {budget.requests_limit.toLocaleString()}
          </span>
        </div>
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${
              usagePercent > 90 ? "bg-[var(--status-error-text)]" :
              usagePercent > 70 ? "bg-[var(--status-warning-text)]" :
              "bg-[var(--status-success-text)]"
            }`}
            style={{ width: `${Math.min(usagePercent, 100)}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-xs text-muted-foreground mt-2">
          <span>{usagePercent.toFixed(1)}% used</span>
          <span>{budget.requests_remaining.toLocaleString()} remaining</span>
        </div>
      </div>

      {/* Reset time */}
      {budget.tokens_reset_at_la && (
        <div className="flex items-center justify-between pt-2 border-t border-border">
          <span className="text-sm text-muted-foreground flex items-center gap-1.5">
            <Clock className="h-3.5 w-3.5" />
            Resets
          </span>
          <span className="text-sm text-foreground">
            {budget.tokens_reset_at_la}
          </span>
        </div>
      )}
    </div>
  );
}
