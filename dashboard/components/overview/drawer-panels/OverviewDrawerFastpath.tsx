"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { AlertCircle, Zap, Clock } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";

interface OverviewDrawerFastpathProps {
  tab: OverviewTab;
}

export function OverviewDrawerFastpath({ tab }: OverviewDrawerFastpathProps) {
  if (tab === "runs") {
    return <FastpathRunsTab />;
  }
  return <FastpathSummaryTab />;
}

function FastpathSummaryTab() {
  const { fastpath, isFastpathDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!fastpath || isFastpathDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Fastpath data unavailable
      </div>
    );
  }

  const statusColors = {
    ok: "text-green-400",
    warning: "text-yellow-400",
    critical: "text-red-400",
    degraded: "text-orange-400",
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Status</span>
        <span className={`text-sm font-medium ${statusColors[fastpath.status]}`}>
          {fastpath.status.charAt(0).toUpperCase() + fastpath.status.slice(1)}
        </span>
      </div>

      <div className="bg-muted/30 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-2">
          <Zap className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium text-foreground">Backlog</span>
        </div>
        <div className="text-3xl font-bold text-foreground tabular-nums">
          {fastpath.pending_ready ?? 0}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Matches ready for narrative
        </p>
      </div>

      {fastpath.minutes_since_tick !== undefined && (
        <div className="flex items-center justify-between pt-2 border-t border-border">
          <span className="text-sm text-muted-foreground flex items-center gap-1.5">
            <Clock className="h-3.5 w-3.5" />
            Last Tick
          </span>
          <span className="text-sm text-foreground">
            {fastpath.minutes_since_tick < 60
              ? `${Math.round(fastpath.minutes_since_tick)}m ago`
              : `${Math.floor(fastpath.minutes_since_tick / 60)}h ago`}
          </span>
        </div>
      )}
    </div>
  );
}

function FastpathRunsTab() {
  return (
    <div className="p-4">
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          Fastpath runs history coming soon
        </p>
      </div>
    </div>
  );
}
