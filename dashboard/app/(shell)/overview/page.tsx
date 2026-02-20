"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  useOpsOverview,
  useTodayMatches,
  useActiveIncidentsApi,
  useTeamSearch,
} from "@/lib/hooks";
import { useDebounce } from "@/lib/hooks/use-debounce";
import {
  mockApiBudget,
  mockActiveIncidents,
} from "@/lib/mocks";
import {
  ApiBudgetCard,
  SentryHealthCard,
  LlmCostCard,
  TodayMatchesList,
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
  IncidentsCompactTile,
  SofascoreCronCompactTile,
  // Model Benchmark
  ModelBenchmarkTile,
  // Drawer
  OverviewDrawer,
} from "@/components/overview";
import { useOverviewDrawer } from "@/lib/hooks/use-overview-drawer";
import { OverviewPanel } from "@/lib/overview-drawer";
import { AlertCircle, Calendar, ChevronLeft, ChevronRight, LayoutDashboard, Users } from "lucide-react";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { SearchInput } from "@/components/ui/search-input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/**
 * Overview Page
 *
 * Ops dashboard designed for 5-10 second scan.
 * Above-the-fold layout optimized for 1440x900.
 *
 * Layout:
 * - Left Rail (290px): Today Matches
 * - Main:
 *   - Row 1: Model Benchmark (full width)
 *   - Row 2: Grid 8 tiles (OverallOps, Incidents, Predictions, Jobs, Fastpath, PIT, TITAN)
 *   - Row 3: [API Budget + Sentry + LLM Cost] | Diagnostics
 *   - Row 4: SOTA Enrichment + Movement Summary
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
    sofascoreCron,
    isSofascoreCronDegraded,
    isDegraded,
    isLoading,
    error,
    requestId,
    refetch,
  } = useOpsOverview();

  // Fetch today's matches from real backend
  const {
    matches: todayMatches,
    isDegraded: isTodayDegraded,
    isLoading: isTodayLoading,
  } = useTodayMatches();

  // Fetch active incidents from real backend
  const {
    incidents: activeIncidents,
    isDegraded: isIncidentsDegraded,
    isLoading: isIncidentsLoading,
  } = useActiveIncidentsApi();

  // Left rail collapse state
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);

  // Team search
  const router = useRouter();
  const [teamSearchQuery, setTeamSearchQuery] = useState("");
  const [showTeamResults, setShowTeamResults] = useState(false);
  const teamSearchRef = useRef<HTMLDivElement>(null);
  const debouncedTeamSearch = useDebounce(teamSearchQuery, 200);
  const { data: teamSearchData, isLoading: isTeamSearching } = useTeamSearch(
    debouncedTeamSearch,
    showTeamResults
  );

  const handleTeamSearchChange = useCallback((value: string) => {
    setTeamSearchQuery(value);
    setShowTeamResults(value.length >= 2);
  }, []);

  const handleTeamClick = useCallback((teamId: number) => {
    setTeamSearchQuery("");
    setShowTeamResults(false);
    router.push(`/football?team=${teamId}`);
  }, [router]);

  // Close team search results when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (teamSearchRef.current && !teamSearchRef.current.contains(event.target as Node)) {
        setShowTeamResults(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

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

  // Use real active incidents with mock fallback
  const displayActiveIncidents = activeIncidents ?? mockActiveIncidents;

  // Today's matches (no mock fallback - show empty state if unavailable)
  const displayTodayMatches = todayMatches ?? [];

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
    <div className="h-full flex overflow-hidden relative">
      {/* Overview Drawer (URL-controlled) */}
      <OverviewDrawer />

      {/* Left Rail: Today Matches */}
      {leftRailCollapsed ? (
        /* Collapsed state: thin rail with expand button */
        <aside className="w-12 shrink-0 bg-sidebar flex flex-col items-center py-3 transition-smooth">
          <TooltipProvider delayDuration={0}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setLeftRailCollapsed(false)}
                  className="mb-2"
                  aria-label="Expand panel"
                >
                  <ChevronRight className="h-4 w-4" strokeWidth={1.5} />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>Expand panel</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-muted-foreground"
                  aria-label="Overview"
                >
                  <LayoutDashboard className="h-4 w-4" strokeWidth={1.5} />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>Overview</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </aside>
      ) : (
        /* Expanded state: full panel */
        <aside data-dev-ref="OverviewPage:LeftRail" className="w-[290px] min-w-[290px] shrink-0 bg-sidebar flex flex-col overflow-hidden transition-smooth">
          {/* Header with collapse button */}
          <div className="h-12 flex items-center justify-between pl-3 pr-0 shrink-0">
            <span className="text-sm font-semibold text-foreground">Overview</span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setLeftRailCollapsed(true)}
              className="h-8 w-8 -mr-1"
              aria-label="Collapse panel"
            >
              <ChevronLeft className="h-4 w-4" strokeWidth={1.5} />
            </Button>
          </div>

          {/* Team Search */}
          <div data-dev-ref="OverviewPage:TeamSearch" className="px-3 pt-3 pb-2 relative shrink-0" ref={teamSearchRef}>
            <SearchInput
              placeholder="Search"
              value={teamSearchQuery}
              onChange={handleTeamSearchChange}
              onFocus={() => teamSearchQuery.length >= 2 && setShowTeamResults(true)}
            />
            {showTeamResults && (
              <div className="absolute left-0 right-0 top-full mt-1 bg-popover border border-border rounded-md shadow-lg z-50 max-h-64 overflow-y-auto">
                {isTeamSearching ? (
                  <div className="flex items-center justify-center py-4">
                    <Loader size="sm" />
                  </div>
                ) : teamSearchData?.teams && teamSearchData.teams.length > 0 ? (
                  <div className="py-1">
                    {teamSearchData.teams.map((team) => (
                      <button
                        key={team.team_id}
                        onClick={() => handleTeamClick(team.team_id)}
                        className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted text-left"
                      >
                        {team.logo_url ? (
                          <img
                            src={team.logo_url}
                            alt=""
                            className="w-5 h-5 object-contain"
                          />
                        ) : (
                          <Users className="w-5 h-5 text-muted-foreground" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="truncate text-foreground">{team.display_name}</p>
                          <p className="text-xs text-muted-foreground truncate">
                            {team.country} · {team.team_type}
                          </p>
                        </div>
                      </button>
                    ))}
                    {teamSearchData.pagination.has_more && (
                      <div className="px-3 py-1.5 text-xs text-muted-foreground text-center border-t border-border">
                        +{Math.max(0, teamSearchData.pagination.total - teamSearchData.teams.length)} more
                      </div>
                    )}
                  </div>
                ) : debouncedTeamSearch.length >= 2 ? (
                  <div className="py-4 text-sm text-muted-foreground text-center">
                    No teams found
                  </div>
                ) : null}
              </div>
            )}
          </div>

          {/* Content - min-h-0 allows flex child to shrink below content size */}
          <ScrollArea className="flex-1 min-h-0">
            <div className="pl-3 pr-1.5 py-3 space-y-3">
              {/* Today Matches */}
              <div data-dev-ref="OverviewPage:TodayMatches">
                <div className="flex items-center gap-2 mb-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium text-foreground">
                    Today Matches
                  </span>
                  <span className="text-sm font-medium text-primary">
                    ({displayTodayMatches.length})
                  </span>
                  {isTodayDegraded && (
                    <span className="text-[10px] text-[var(--status-warning-text)] bg-[var(--status-warning-bg)] px-1.5 py-0.5 rounded">
                      error
                    </span>
                  )}
                </div>
                {isTodayLoading ? (
                  <div className="flex items-center justify-center py-4">
                    <Loader size="sm" />
                  </div>
                ) : (
                  <TodayMatchesList matches={displayTodayMatches} />
                )}
              </div>
            </div>
          </ScrollArea>
        </aside>
      )}

      {/* Main content - optimized for above-the-fold at 1440x900 */}
      <div data-dev-ref="OverviewPage:MainContent" className="flex-1 flex flex-col overflow-hidden">
        <div className="pl-1.5 pr-4 py-4 space-y-4 flex-1 overflow-auto">
          {/* Row 1: Model Benchmark - Full Width */}
          <ModelBenchmarkTile />

          {/* Row 2: Grid - 8 compact tiles */}
          <div data-dev-ref="OverviewPage:TilesGrid" className="grid grid-cols-4 xl:grid-cols-8 gap-3">
            <OverallOpsBar
              statuses={overallStatuses}
              jobs={jobs}
              freshness={freshness}
              onRefresh={() => refetch()}
              compact
            />
            <IncidentsCompactTile
              incidents={displayActiveIncidents}
              isMockFallback={isIncidentsDegraded}
            />
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
            <TitanCompactTile
              titan={titan}
              isMockFallback={isTitanDegraded}
            />
            <SofascoreCronCompactTile
              cron={sofascoreCron}
              isMockFallback={isSofascoreCronDegraded}
            />
          </div>

          {/* Row 3: API Budget + Sentry + LLM Cost | Diagnostics */}
          <div data-dev-ref="OverviewPage:BudgetDiagnosticsRow" className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {/* Left half: 3 compact cards */}
            <div className="grid grid-cols-3 gap-3">
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
            </div>
            <DiagnosticsTile
              shadowMode={shadowMode}
              sensorB={sensorB}
            />
          </div>

          {/* Row 4: SOTA Enrichment + Movement Summary */}
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
