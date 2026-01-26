"use client";

import {
  useOpsOverview,
  useUpcomingMatches,
  useActiveIncidentsApi,
} from "@/lib/hooks";
import {
  mockApiBudget,
  mockUpcomingMatches,
  mockActiveIncidents,
} from "@/lib/mocks";
import {
  ApiBudgetCard,
  SentryHealthCard,
  LlmCostCard,
  UpcomingMatchesList,
  ActiveIncidentsList,
  SotaEnrichmentSection,
  // New compact components for above-the-fold layout
  OverallOpsBar,
  PredictionsCompactTile,
  JobsCompactTile,
  FastpathCompactTile,
  PitProgressCompactTile,
  MovementSummaryTile,
  DiagnosticsTile,
  TitanCompactTile,
  // Drawer
  OverviewDrawer,
} from "@/components/overview";
import { useOverviewDrawer } from "@/lib/hooks/use-overview-drawer";
import { OverviewPanel } from "@/lib/overview-drawer";
import { AlertCircle, AlertTriangle, Calendar } from "lucide-react";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

/**
 * Overview Page
 *
 * Ops dashboard designed for 5-10 second scan.
 * Above-the-fold layout optimized for 1440x900.
 *
 * Layout:
 * - Left Rail (277px): API-Football Budget, Sentry, LLM Cost, Upcoming Matches
 * - Main:
 *   - Row 1: Overall Ops Bar (≤48px)
 *   - Row 2: Grid 2x2 (Predictions, Jobs, Fastpath, PIT Progress)
 *   - Row 3: Grid 2 cols (SOTA Enrichment, Movement Summary)
 */
/**
 * Clickable tile wrapper that opens the drawer
 */
function ClickableTile({
  panel,
  children,
  className,
}: {
  panel: OverviewPanel;
  children: React.ReactNode;
  className?: string;
}) {
  const { openDrawer } = useOverviewDrawer();

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => openDrawer({ panel })}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') openDrawer({ panel }); }}
      className={`w-full text-left cursor-pointer hover:ring-1 hover:ring-primary/50 rounded-lg transition-all ${className ?? ""}`}
    >
      {children}
    </div>
  );
}

export default function OverviewPage() {
  const {
    budget,
    sentry,
    llmCost,
    jobs,
    fastpath,
    predictions,
    freshness,
    progress,
    pitActivity,
    movement,
    sotaEnrichment,
    isBudgetDegraded,
    isSentryDegraded,
    isLlmCostDegraded,
    isJobsDegraded,
    isFastpathDegraded,
    isPredictionsDegraded,
    isProgressDegraded,
    isPitActivityDegraded,
    isMovementDegraded,
    isSotaEnrichmentDegraded,
    shadowMode,
    sensorB,
    titan,
    isTitanDegraded,
    isDegraded,
    isLoading,
    error,
    requestId,
    refetch,
  } = useOpsOverview();

  // Fetch upcoming matches from real backend
  const {
    matches: upcomingMatches,
    isDegraded: isUpcomingDegraded,
    isLoading: isUpcomingLoading,
  } = useUpcomingMatches();

  // Fetch active incidents from real backend
  const {
    incidents: activeIncidents,
    isDegraded: isIncidentsDegraded,
    isLoading: isIncidentsLoading,
  } = useActiveIncidentsApi();

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

  // Use real upcoming matches with mock fallback
  const displayUpcomingMatches = upcomingMatches ?? mockUpcomingMatches;

  // Use real active incidents with mock fallback
  const displayActiveIncidents = activeIncidents ?? mockActiveIncidents;

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
      {/* Overview Drawer (URL-controlled) */}
      <OverviewDrawer />

      {/* Left Rail: Budget + Sentry + LLM Cost + Upcoming Matches */}
      <aside className="w-[277px] min-w-[277px] shrink-0 border-r border-border bg-sidebar flex flex-col overflow-hidden">
        {/* Header */}
        <div className="h-12 flex items-center px-3 border-b border-border">
          <span className="text-sm font-medium text-foreground">Services</span>
        </div>
        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="p-3 space-y-3">
            <ClickableTile panel="budget">
              <ApiBudgetCard
                budget={displayBudget}
                isMockFallback={isBudgetDegraded}
                requestId={requestId}
              />
            </ClickableTile>
            <ClickableTile panel="sentry">
              <SentryHealthCard
                sentry={sentry}
                isMockFallback={isSentryDegraded}
              />
            </ClickableTile>
            <ClickableTile panel="llm">
              <LlmCostCard
                llmCost={llmCost}
                isMockFallback={isLlmCostDegraded}
              />
            </ClickableTile>

            {/* Upcoming Matches */}
            <div className="pt-3 border-t border-border">
              <div className="flex items-center gap-2 mb-2">
                <Calendar className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium text-foreground">
                  Upcoming Matches
                </span>
                {isUpcomingDegraded && (
                  <span className="text-[10px] text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded">
                    mock
                  </span>
                )}
              </div>
              {isUpcomingLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader size="sm" />
                </div>
              ) : (
                <UpcomingMatchesList
                  matches={displayUpcomingMatches.slice(0, 5)}
                />
              )}
            </div>

            {/* Active Incidents */}
            <div className="pt-3 border-t border-border">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium text-foreground">
                  Active Incidents
                </span>
                {isIncidentsDegraded && (
                  <span className="text-[10px] text-yellow-400 bg-yellow-500/10 px-1.5 py-0.5 rounded">
                    mock
                  </span>
                )}
              </div>
              {isIncidentsLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader size="sm" />
                </div>
              ) : (
                <ActiveIncidentsList
                  incidents={displayActiveIncidents.slice(0, 5)}
                />
              )}
            </div>
          </div>
        </ScrollArea>
      </aside>

      {/* Main content - optimized for above-the-fold at 1440x900 */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="p-4 space-y-4 flex-1 overflow-auto">
          {/* Row 1: Overall Ops Bar (≤48px) */}
          <OverallOpsBar
            statuses={overallStatuses}
            jobs={jobs}
            freshness={freshness}
            onRefresh={() => refetch()}
          />

          {/* Row 2: Grid 2x2 - Predictions, Jobs, Fastpath, PIT Progress */}
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
            <ClickableTile panel="predictions">
              <PredictionsCompactTile
                predictions={predictions}
                isMockFallback={isPredictionsDegraded}
              />
            </ClickableTile>
            <ClickableTile panel="jobs">
              <JobsCompactTile
                jobs={jobs}
                isMockFallback={isJobsDegraded}
              />
            </ClickableTile>
            <ClickableTile panel="fastpath">
              <FastpathCompactTile
                fastpath={fastpath}
                isMockFallback={isFastpathDegraded}
              />
            </ClickableTile>
            <ClickableTile panel="pit">
              <PitProgressCompactTile
                progress={progress}
                pitActivity={pitActivity}
                isProgressDegraded={isProgressDegraded}
                isPitActivityDegraded={isPitActivityDegraded}
              />
            </ClickableTile>
          </div>

          {/* Row 2b: Diagnostics (Shadow Mode + Sensor B) */}
          {/* Row 2b: Diagnostics (Shadow Mode + Sensor B) + TITAN */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            <DiagnosticsTile
              shadowMode={shadowMode}
              sensorB={sensorB}
            />
            <TitanCompactTile
              titan={titan}
              isMockFallback={isTitanDegraded}
            />
          </div>

          {/* Row 3: SOTA Enrichment + Movement Summary (same row as last SOTA card) */}
          <ClickableTile panel="sota">
            <SotaEnrichmentSection
              data={sotaEnrichment}
              isMockFallback={isSotaEnrichmentDegraded}
              movementTile={
                <ClickableTile panel="movement">
                  <MovementSummaryTile
                    movement={movement}
                    isMovementDegraded={isMovementDegraded}
                  />
                </ClickableTile>
              }
            />
          </ClickableTile>
        </div>
      </div>
    </div>
  );
}
