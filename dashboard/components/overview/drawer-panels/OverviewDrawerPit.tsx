"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { AlertCircle, TrendingUp, CheckCircle2 } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";

interface OverviewDrawerPitProps {
  tab: OverviewTab;
}

export function OverviewDrawerPit({ tab }: OverviewDrawerPitProps) {
  if (tab === "timeline") {
    return <PitTimelineTab />;
  }
  return <PitSummaryTab />;
}

function PitSummaryTab() {
  const { progress, pitActivity, isProgressDegraded, isPitActivityDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if ((!progress && !pitActivity) || (isProgressDegraded && isPitActivityDegraded)) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        PIT progress data unavailable
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Progress */}
      {progress && (
        <div className="space-y-3">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Evaluation Progress
          </h3>
          <div className="bg-muted/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium text-foreground">Baseline Coverage</span>
            </div>
            <div className="text-3xl font-bold text-foreground tabular-nums">
              {progress.baseline_coverage_pct?.toFixed(1) ?? 0}%
            </div>
            <div className="flex items-center justify-between text-xs text-muted-foreground mt-2">
              <span>With baseline: {progress.pit_with_baseline ?? 0}</span>
              <span>Target: {progress.target_baseline_coverage_pct ?? 0}%</span>
            </div>
            {/* Progress bar */}
            <div className="h-2 bg-muted rounded-full overflow-hidden mt-2">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${progress.baseline_coverage_pct ?? 0}%` }}
              />
            </div>
          </div>
          {/* Additional stats */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-muted/30 rounded-lg p-3">
              <div className="text-lg font-semibold text-foreground tabular-nums">
                {progress.pit_snapshots_30d}
              </div>
              <p className="text-xs text-muted-foreground">Snapshots (30d)</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-3">
              <div className="text-lg font-semibold text-foreground tabular-nums">
                {progress.pit_bets_30d}
              </div>
              <p className="text-xs text-muted-foreground">Bets (30d)</p>
            </div>
          </div>
          {progress.ready_for_retest && (
            <div className="flex items-center gap-2 text-[var(--status-success-text)] text-sm">
              <CheckCircle2 className="h-4 w-4" />
              <span>Ready for retest</span>
            </div>
          )}
        </div>
      )}

      {/* Activity */}
      {pitActivity && (
        <div className="space-y-3">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Live Predictions Activity
          </h3>
          <div className="flex items-center justify-between py-2">
            <span className="text-sm text-muted-foreground">Live (24h)</span>
            <span className="text-sm font-medium text-foreground tabular-nums">
              {pitActivity.live_24h ?? 0}
            </span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-sm text-muted-foreground">Live (60m)</span>
            <span className="text-sm font-medium text-foreground tabular-nums">
              {pitActivity.live_60m ?? 0}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function PitTimelineTab() {
  return (
    <div className="p-4">
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          PIT timeline coming soon
        </p>
      </div>
    </div>
  );
}
