"use client";

import { useOpsOverview } from "@/lib/hooks";
import { mockApiBudget } from "@/lib/mocks";
import {
  ApiBudgetCard,
  SentryHealthCard,
  LlmCostCard,
  OverallOpsTile,
  PredictionsHealthTile,
  JobsHealthTile,
  FastpathHealthTile,
  DiagnosticsTile,
} from "@/components/overview";
import { AlertCircle } from "lucide-react";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

/**
 * Overview Page
 *
 * Ops dashboard designed for 5-10 second scan.
 *
 * Layout:
 * - Left Rail (277px): API-Football Budget, Sentry, LLM Cost
 * - Main: Overall Ops + Tiles grid (Predictions, Jobs, Fastpath, Diagnostics)
 */
export default function OverviewPage() {
  const {
    budget,
    sentry,
    llmCost,
    jobs,
    fastpath,
    predictions,
    shadowMode,
    sensorB,
    freshness,
    isBudgetDegraded,
    isSentryDegraded,
    isLlmCostDegraded,
    isJobsDegraded,
    isFastpathDegraded,
    isPredictionsDegraded,
    isDegraded,
    isLoading,
    error,
    requestId,
    refetch,
  } = useOpsOverview();

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <Loader size="md" />
      </div>
    );
  }

  // Full error state (only if ALL data failed)
  if (error && isDegraded) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <AlertCircle className="h-12 w-12 text-error" />
          <div className="text-center">
            <p className="text-foreground font-medium mb-1">
              Failed to load overview
            </p>
            <p className="text-sm text-muted-foreground mb-4">
              {error.message}
            </p>
          </div>
          <Button variant="outline" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Use real data with mock fallback for budget only
  const displayBudget = budget ?? mockApiBudget;

  // Build statuses for overall rollup
  const overallStatuses = {
    jobs: jobs?.status ?? null,
    predictions: predictions?.status ?? null,
    fastpath: fastpath?.status ?? null,
    budget: budget?.status ?? null,
    sentry: sentry?.status ?? null,
    llmCost: llmCost?.status ?? null,
  };

  return (
    <div className="h-full flex overflow-hidden">
      {/* Left Rail: Budget + Sentry + LLM Cost */}
      <aside className="w-[277px] shrink-0 border-r border-border bg-sidebar flex flex-col">
        {/* Header */}
        <div className="h-12 flex items-center px-3 border-b border-border">
          <span className="text-sm font-medium text-foreground">Services</span>
        </div>
        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="p-3 space-y-3">
            <ApiBudgetCard
              budget={displayBudget}
              isMockFallback={isBudgetDegraded}
              requestId={requestId}
            />
            <SentryHealthCard
              sentry={sentry}
              isMockFallback={isSentryDegraded}
            />
            <LlmCostCard
              llmCost={llmCost}
              isMockFallback={isLlmCostDegraded}
            />
          </div>
        </ScrollArea>
      </aside>

      {/* Main content */}
      <ScrollArea className="flex-1">
        <div className="p-6 space-y-6">
          {/* Overall Ops - full width at top */}
          <OverallOpsTile
            statuses={overallStatuses}
            freshness={freshness}
            onRefresh={() => refetch()}
          />

          {/* Main tiles grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Predictions Health */}
            <PredictionsHealthTile
              predictions={predictions}
              isMockFallback={isPredictionsDegraded}
            />

            {/* Jobs Health */}
            <JobsHealthTile
              jobs={jobs}
              isMockFallback={isJobsDegraded}
            />

            {/* Fastpath Health */}
            <FastpathHealthTile
              fastpath={fastpath}
              isMockFallback={isFastpathDegraded}
            />

            {/* Diagnostics (Shadow Mode + Sensor B) */}
            <DiagnosticsTile
              shadowMode={shadowMode}
              sensorB={sensorB}
            />
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
