"use client";

import { useState } from "react";
import { OverviewTab } from "@/lib/overview-drawer";
import { AlertCircle, TrendingUp, Clock, ChevronLeft, ChevronRight } from "lucide-react";
import { useOpsOverview, usePredictionsMissing } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";

interface OverviewDrawerPredictionsProps {
  tab: OverviewTab;
}

/**
 * Predictions panel content for overview drawer
 *
 * Tabs:
 * - summary: Coverage from ops/rollup
 * - missing: Paginated missing list (from /api/predictions/missing)
 */
export function OverviewDrawerPredictions({ tab }: OverviewDrawerPredictionsProps) {
  if (tab === "missing") {
    return <PredictionsMissingTab />;
  }

  return <PredictionsSummaryTab />;
}

/**
 * Summary tab - uses existing ops data
 */
function PredictionsSummaryTab() {
  const { predictions, isPredictionsDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!predictions || isPredictionsDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Predictions data unavailable
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
      {/* Status */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Status</span>
        <span className={`text-sm font-medium ${statusColors[predictions.status]}`}>
          {predictions.status.charAt(0).toUpperCase() + predictions.status.slice(1)}
        </span>
      </div>

      {/* NS Coverage (Next 48h) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">NS Coverage</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {predictions.ns_coverage_pct?.toFixed(1) ?? 0}%
          </span>
        </div>
        {/* Progress bar */}
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${predictions.ns_coverage_pct ?? 0}%` }}
          />
        </div>
      </div>

      {/* Next 48h stats */}
      <div className="pt-2 border-t border-border space-y-2">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <TrendingUp className="h-3.5 w-3.5" />
          <span>Next 48h (NS)</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Total Matches</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {predictions.ns_matches_next_48h ?? 0}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">With Prediction</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {(predictions.ns_matches_next_48h ?? 0) - (predictions.ns_matches_next_48h_missing_prediction ?? 0)}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Missing</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {predictions.ns_matches_next_48h_missing_prediction ?? 0}
          </span>
        </div>
      </div>

      {/* FT Coverage (Last 48h) */}
      <div className="pt-2 border-t border-border space-y-2">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <TrendingUp className="h-3.5 w-3.5" />
          <span>Last 48h (FT)</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">FT Coverage</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {predictions.ft_coverage_pct?.toFixed(1) ?? 0}%
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">FT Matches</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {predictions.ft_matches_last_48h ?? 0}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Missing Prediction</span>
          <span className="text-sm font-medium text-foreground tabular-nums">
            {predictions.ft_matches_last_48h_missing_prediction ?? 0}
          </span>
        </div>
      </div>
    </div>
  );
}

const hoursOptions = [
  { value: 24, label: "24h" },
  { value: 48, label: "48h" },
  { value: 72, label: "72h" },
];

/**
 * Missing tab - paginated list from /api/predictions/missing
 */
function PredictionsMissingTab() {
  const [hours, setHours] = useState(48);
  const [page, setPage] = useState(1);
  const limit = 10;

  const { matches, total, hasMore, isLoading, isDegraded } = usePredictionsMissing({
    hours,
    page,
    limit,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (isDegraded) {
    return (
      <div className="p-4">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">
            Unable to fetch missing predictions
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Backend endpoint unavailable
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Hours filter */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Time window</span>
        <div className="flex gap-1">
          {hoursOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => {
                setHours(opt.value);
                setPage(1);
              }}
              className={cn(
                "px-2 py-1 text-xs rounded-full transition-colors",
                hours === opt.value
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Missing matches list */}
      {matches.length === 0 ? (
        <div className="bg-green-500/10 rounded-lg p-4 text-center">
          <p className="text-sm text-green-400">No missing predictions</p>
          <p className="text-xs text-muted-foreground mt-1">
            All matches in the next {hours}h have predictions
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {matches.map((match) => (
            <div
              key={match.fixture_id}
              className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {match.home_team} vs {match.away_team}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {match.league_name}
                  </p>
                </div>
                <span
                  className={cn(
                    "shrink-0 px-1.5 py-0.5 text-[10px] rounded",
                    match.hours_until_kickoff <= 2
                      ? "bg-red-500/20 text-red-400"
                      : match.hours_until_kickoff <= 6
                        ? "bg-orange-500/20 text-orange-400"
                        : "bg-yellow-500/20 text-yellow-400"
                  )}
                >
                  {match.hours_until_kickoff}h
                </span>
              </div>
              <div className="flex items-center gap-2 mt-1.5 text-[10px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {new Date(match.kickoff_utc).toLocaleString()}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-muted/50">
                  #{match.fixture_id}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {total > limit && (
        <div className="flex items-center justify-between pt-2 border-t border-border">
          <span className="text-xs text-muted-foreground">
            {(page - 1) * limit + 1}-{Math.min(page * limit, total)} of {total}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-50"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={!hasMore}
              className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-50"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
