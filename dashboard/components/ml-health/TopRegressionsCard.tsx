"use client";

import { TopRegressions } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";
import { DegradedAlert } from "./DegradedAlert";
import { TrendingDown, Info } from "lucide-react";

interface TopRegressionsCardProps {
  data: TopRegressions | null;
}

/**
 * Top Regressions Card
 *
 * Shows placeholder while status = "not_ready"
 * When ready, will show regression list
 */
export function TopRegressionsCard({ data }: TopRegressionsCardProps) {
  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <TrendingDown className="h-4 w-4" />
          <span className="text-sm">Top Regressions unavailable</span>
        </div>
      </div>
    );
  }

  const isNotReady = data.status === "not_ready";
  const { _degraded, _error } = data;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <TrendingDown className="h-4 w-4 text-orange-400" />
          <h3 className="text-sm font-semibold text-foreground">Top Regressions</h3>
        </div>
        <StatusBadge status={data.status} />
      </div>

      {/* Degraded alert */}
      {_degraded && (
        <div className="mb-4">
          <DegradedAlert error={_error} />
        </div>
      )}

      {/* Content */}
      {isNotReady ? (
        <div className="p-4 bg-muted/30 rounded border border-border flex items-start gap-3">
          <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
          <div>
            <p className="text-sm text-muted-foreground">
              {data.note || "Regression analysis not yet available"}
            </p>
          </div>
        </div>
      ) : (
        // Future: render actual regressions when backend provides them
        <div className="text-sm text-muted-foreground">
          Regression data available - render list here
        </div>
      )}
    </div>
  );
}
