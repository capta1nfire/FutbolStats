"use client";

import { PredictionCoverage } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { TrendingUp, AlertTriangle, ExternalLink } from "lucide-react";

interface PredictionsCoverageCardProps {
  coverage?: PredictionCoverage | null;
  onViewMissing?: () => void;
  isLoading?: boolean;
}

export function PredictionsCoverageCard({
  coverage,
  onViewMissing,
  isLoading,
}: PredictionsCoverageCardProps) {
  if (isLoading || !coverage) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-border animate-pulse">
        <div className="h-6 bg-background rounded w-1/3 mb-2" />
        <div className="h-8 bg-background rounded w-1/4 mb-2" />
        <div className="h-4 bg-background rounded w-1/2" />
      </div>
    );
  }

  const isGoodCoverage = coverage.coveragePct >= 90;
  const hasMissing = coverage.missingCount > 0;

  return (
    <div className="bg-surface rounded-lg p-4 border border-border">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium text-muted-foreground">
              Prediction Coverage
            </span>
            <span className="text-xs text-muted-foreground">
              ({coverage.periodLabel})
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className={`text-3xl font-bold ${
                isGoodCoverage ? "text-success" : "text-warning"
              }`}
            >
              {coverage.coveragePct}%
            </span>
            <span className="text-sm text-muted-foreground">
              {coverage.withPrediction} of {coverage.totalMatches} matches
            </span>
          </div>
        </div>

        {hasMissing && (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-error">
              <AlertTriangle className="h-4 w-4" />
              <span className="text-sm font-medium">
                {coverage.missingCount} missing
              </span>
            </div>
            {onViewMissing && (
              <Button
                variant="outline"
                size="sm"
                onClick={onViewMissing}
                className="text-xs"
              >
                View missing
                <ExternalLink className="h-3 w-3 ml-1" />
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
