"use client";

import { useMemo, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  useFeatureCoverage,
  FeatureCoverageLeague,
} from "@/lib/hooks/use-feature-coverage";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/**
 * Format percentage: show decimal only if not a whole number
 * 100.0 -> "100%", 94.5 -> "94.5%"
 */
function formatPct(pct: number): string {
  return pct % 1 === 0 ? `${pct}%` : `${pct.toFixed(1)}%`;
}

/**
 * Color thresholds for coverage cells (text only, no background)
 * - Green: 100%
 * - Yellow: 70% - 99%
 * - Red: < 70%
 */
function getCoverageColor(pct: number): string {
  if (pct >= 100) return "text-[var(--status-success-text)]";
  if (pct >= 70) return "text-[var(--status-warning-text)]";
  return "text-[var(--status-error-text)]";
}

/**
 * Badge component for tier labels
 */
function TierBadge({ badge }: { badge: string }) {
  const colors =
    badge === "PROD"
      ? "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)] border-[var(--tag-blue-border)]"
      : "bg-[var(--tag-purple-bg)] text-[var(--tag-purple-text)] border-[var(--tag-purple-border)]";

  return (
    <span
      className={cn(
        "px-1.5 py-0.5 text-[9px] font-medium rounded border",
        colors
      )}
    >
      {badge}
    </span>
  );
}

/**
 * Coverage cell with tooltip
 *
 * isProd: true for tier1 (PROD) features, false for TITAN features
 * Affects the label in tooltip: "Total (FT)" vs "Total (TITAN)"
 */
function CoverageCell({
  pct,
  n,
  matchesTotal,
  windowKey,
  leagueName,
  featureKey,
  isProd,
  leagueId,
  onCellClick,
}: {
  pct: number;
  n: number;
  matchesTotal: number;
  windowKey: string;
  leagueName: string;
  featureKey: string;
  isProd: boolean;
  leagueId: number;
  onCellClick?: (featureKey: string, leagueId: number) => void;
}) {
  const totalLabel = isProd ? "Total (FT):" : "Total (TITAN):";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={cn(
            "text-center text-sm tabular-nums",
            onCellClick ? "cursor-pointer hover:bg-emerald-500/15 rounded px-1.5 py-0.5" : "cursor-default",
            getCoverageColor(pct)
          )}
          role={onCellClick ? "button" : undefined}
          tabIndex={onCellClick ? 0 : undefined}
          onClick={() => onCellClick?.(featureKey, leagueId)}
          onKeyDown={
            onCellClick
              ? (e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onCellClick(featureKey, leagueId);
                  }
                }
              : undefined
          }
        >
          {formatPct(pct)}
        </div>
      </TooltipTrigger>
      <TooltipContent side="top" className="text-xs">
        <div className="space-y-1">
          <div className="font-medium">{featureKey}</div>
          <div className="text-muted-foreground">
            {leagueName} · {windowKey}
          </div>
          <div className="grid grid-cols-2 gap-x-3 text-muted-foreground">
            <span>Coverage:</span>
            <span className="text-foreground">{formatPct(pct)}</span>
            <span>Non-null:</span>
            <span className="text-foreground">{n.toLocaleString()}</span>
            <span>{totalLabel}</span>
            <span className="text-foreground">{matchesTotal.toLocaleString()}</span>
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * N cell (match count)
 */
function NCell({ n }: { n: number }) {
  return (
    <div className="text-center text-sm tabular-nums text-muted-foreground">
      {n.toLocaleString()}
    </div>
  );
}

/**
 * Coverage range filter options
 */
export type CoverageRangeFilter = "all" | "100" | "70-99" | "50-69" | "below50";

interface FeatureCoverageMatrixProps {
  className?: string;
  /** Function to check if a league is visible (if not provided, all are visible) */
  isLeagueVisible?: (leagueId: number) => boolean;
  /** Callback to receive available leagues when data loads */
  onLeaguesLoaded?: (leagues: FeatureCoverageLeague[]) => void;
  /** Current page for pagination (1-indexed) */
  currentPage?: number;
  /** Page size for pagination */
  pageSize?: number;
  /** Callback when total features count changes (for external pagination) */
  onTotalFeaturesChange?: (total: number) => void;
  /** Enabled tiers (if not provided, all are enabled) */
  enabledTiers?: Set<string>;
  /** Coverage range filter */
  coverageRangeFilter?: CoverageRangeFilter;
  /** Callback when a coverage cell is clicked (opens detail drawer) */
  onCellClick?: (featureKey: string, leagueId: number) => void;
}

/**
 * Feature Coverage Matrix
 *
 * Displays coverage matrix with:
 * - Sticky first column (Feature)
 * - Grouped headers by league with subcolumns per window
 * - Tier toggles to filter rows
 * - Sort by league average coverage
 * - Color-coded cells (green >80%, yellow 50-80%, red <50%)
 */
// Default enabled tiers
const DEFAULT_ENABLED_TIERS = new Set(["tier1", "tier1b", "tier1c", "tier1d"]);

/**
 * Check if a league's coverage matches the selected range filter
 */
function matchesCoverageRange(avgPct: number, filter: CoverageRangeFilter): boolean {
  switch (filter) {
    case "all":
      return true;
    case "100":
      return avgPct >= 100;
    case "70-99":
      return avgPct >= 70 && avgPct < 100;
    case "50-69":
      return avgPct >= 50 && avgPct < 70;
    case "below50":
      return avgPct < 50;
    default:
      return true;
  }
}

export function FeatureCoverageMatrix({
  className,
  isLeagueVisible,
  onLeaguesLoaded,
  currentPage = 1,
  pageSize = 25,
  onTotalFeaturesChange,
  enabledTiers = DEFAULT_ENABLED_TIERS,
  coverageRangeFilter = "all",
  onCellClick,
}: FeatureCoverageMatrixProps) {
  const { data, isLoading, error, refetch } = useFeatureCoverage();

  // Notify parent when leagues are available
  useEffect(() => {
    if (data?.data?.leagues && onLeaguesLoaded) {
      onLeaguesLoaded(data.data.leagues);
    }
  }, [data?.data?.leagues, onLeaguesLoaded]);

  // Process data
  const { sortedLeagues, filteredFeatures, windows, tiers, coverage, leagueSummaries } =
    useMemo(() => {
      if (!data?.data) {
        return {
          sortedLeagues: [],
          filteredFeatures: [],
          windows: [],
          tiers: [],
          coverage: {},
          leagueSummaries: {},
        };
      }

      const { windows, tiers, features, leagues, league_summaries, coverage } =
        data.data;

      // Filter features by enabled tiers
      const filteredFeatures = features.filter((f) =>
        enabledTiers.has(f.tier_id)
      );

      // Filter leagues by visibility (if filter function provided)
      let visibleLeagues = isLeagueVisible
        ? leagues.filter((l) => isLeagueVisible(l.league_id))
        : leagues;

      // Filter leagues by coverage range
      if (coverageRangeFilter !== "all") {
        visibleLeagues = visibleLeagues.filter((l) => {
          const avgPct = league_summaries[String(l.league_id)]?.total?.avg_pct ?? 0;
          return matchesCoverageRange(avgPct, coverageRangeFilter);
        });
      }

      // Sort leagues by total avg_pct (descending - highest first)
      const sortedLeagues = [...visibleLeagues].sort((a, b) => {
        const aAvg = league_summaries[String(a.league_id)]?.total?.avg_pct ?? 0;
        const bAvg = league_summaries[String(b.league_id)]?.total?.avg_pct ?? 0;
        return bAvg - aAvg;
      });

      return {
        sortedLeagues,
        filteredFeatures,
        windows,
        tiers,
        coverage,
        leagueSummaries: league_summaries,
      };
    }, [data, enabledTiers, isLeagueVisible, coverageRangeFilter]);

  // Notify parent of total features count for pagination
  useEffect(() => {
    if (onTotalFeaturesChange) {
      onTotalFeaturesChange(filteredFeatures.length);
    }
  }, [filteredFeatures.length, onTotalFeaturesChange]);

  // Paginate features
  const paginatedFeatures = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredFeatures.slice(start, start + pageSize);
  }, [filteredFeatures, currentPage, pageSize]);

  // Loading state - matches DataTable style
  if (isLoading) {
    return (
      <div className={cn("flex-1 flex items-center justify-center", className)}>
        <Loader size="md" />
      </div>
    );
  }

  // Error state - matches DataTable style
  if (error) {
    return (
      <div className={cn("flex-1 flex items-center justify-center", className)}>
        <div className="flex flex-col items-center gap-3 text-center">
          <p className="text-sm text-error">Failed to load feature coverage data</p>
          <Button variant="secondary" size="sm" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Empty state - matches DataTable style
  if (!data?.data || sortedLeagues.length === 0) {
    return (
      <div className={cn("flex-1 flex items-center justify-center", className)}>
        <p className="text-sm text-muted-foreground">No coverage data available</p>
      </div>
    );
  }

  // Column count per league: windows + Total + N
  const columnsPerLeague = windows.length + 2; // windows + Total + N

  return (
    <TooltipProvider delayDuration={150}>
    <div className={cn("flex-1 flex flex-col overflow-hidden", className)}>
      {/* Single scroll container for both header and body - syncs horizontal scroll */}
      <div className="flex-1 overflow-auto">
        <table className="border-collapse text-sm">
          {/* Sticky header - position:sticky with top:0 keeps it in sync with horizontal scroll */}
          <thead className="sticky top-0 z-20 bg-background">
            {/* Header row 1: League names */}
            <tr className="border-b border-border">
              {/* Row number column header - no border-r, fixed width */}
              <th
                className="sticky left-0 z-30 px-2 pt-3 pb-2 text-center font-semibold text-muted-foreground text-xs align-bottom bg-background"
                rowSpan={2}
                style={{ width: "40px", minWidth: "40px", maxWidth: "40px" }}
              >
                #
              </th>
              {/* Sticky Feature column header - both left and top sticky */}
              <th
                className="sticky z-30 px-3 pt-3 pb-2 text-left font-semibold text-muted-foreground text-sm align-bottom bg-background border-r border-border relative after:absolute after:top-0 after:right-0 after:bottom-0 after:w-4 after:translate-x-full after:bg-gradient-to-r after:from-black/20 after:to-transparent after:pointer-events-none"
                rowSpan={2}
                style={{ left: "40px", minWidth: "220px" }}
              >
                Feature
              </th>
              {/* League group headers */}
              {sortedLeagues.map((league) => {
                const summary = leagueSummaries[String(league.league_id)]?.total;
                return (
                  <th
                    key={league.league_id}
                    colSpan={columnsPerLeague}
                    className="px-3 pt-3 pb-1 text-center font-semibold text-sm border-r border-border"
                  >
                    <div className="flex flex-col items-center gap-0.5">
                      <span className="text-foreground whitespace-nowrap">
                        {league.name}
                      </span>
                      {summary && (
                        <span
                          className={cn(
                            "text-xs tabular-nums font-normal",
                            getCoverageColor(summary.avg_pct)
                          )}
                        >
                          {formatPct(summary.avg_pct)}
                        </span>
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>

            {/* Header row 2: Window subcolumns */}
            <tr className="border-b border-border">
              {sortedLeagues.flatMap((league) => [
                ...windows.map((window) => (
                  <th
                    key={`${league.league_id}-${window.key}`}
                    className="px-3 py-2 text-center align-middle text-xs font-medium text-muted-foreground border-r border-border bg-background"
                  >
                    {window.key}
                  </th>
                )),
                <th
                  key={`${league.league_id}-total`}
                  className="px-3 py-2 text-center align-middle text-xs font-medium text-muted-foreground border-r border-border bg-background"
                >
                  Total
                </th>,
                <th
                  key={`${league.league_id}-n`}
                  className="px-3 py-2 text-center align-middle text-xs font-medium text-muted-foreground border-r border-border bg-background"
                >
                  N
                </th>,
              ])}
            </tr>
          </thead>

          {/* Body */}
          <tbody>
            {paginatedFeatures.map((feature, idx) => (
              <tr
                key={feature.key}
                className="border-b border-border transition-colors hover:bg-accent/50"
              >
                {/* Row number cell - no border-r, fixed width */}
                <td
                  className="sticky left-0 z-10 px-2 py-2.5 text-center text-xs tabular-nums text-muted-foreground/50 bg-background"
                  style={{ width: "40px", minWidth: "40px" }}
                >
                  {(currentPage - 1) * pageSize + idx + 1}
                </td>
                {/* Sticky Feature cell */}
                <td
                  className="sticky z-10 px-3 py-2.5 bg-background border-r border-border relative after:absolute after:top-0 after:right-0 after:bottom-0 after:w-4 after:translate-x-full after:bg-gradient-to-r after:from-black/20 after:to-transparent after:pointer-events-none"
                  style={{ left: "40px", minWidth: "220px" }}
                >
                  <div className="flex items-center gap-2">
                    <TierBadge badge={feature.badge} />
                    <span className="text-sm text-foreground truncate max-w-[160px]">
                      {feature.key}
                    </span>
                  </div>
                </td>

                {/* Coverage cells per league */}
                {sortedLeagues.flatMap((league) => {
                  const leagueId = String(league.league_id);
                  const featureCoverage = coverage[feature.key]?.[leagueId];
                  const leagueSummary = leagueSummaries[leagueId];
                  // tier1 = PROD (uses FT matches), tier1b/1c/1d = TITAN (uses feature_matrix rows)
                  const isProd = feature.tier_id === "tier1";

                  return [
                    ...windows.map((window) => {
                      const cell = featureCoverage?.[window.key];
                      const summaryForWindow = leagueSummary?.[window.key];
                      // Use correct denominator based on tier
                      const matchesTotal = isProd
                        ? (summaryForWindow?.matches_total_ft ?? 0)
                        : (summaryForWindow?.matches_total_titan ?? 0);

                      if (!cell) {
                        return (
                          <td
                            key={`${feature.key}-${leagueId}-${window.key}`}
                            className="px-3 py-2.5 text-center text-sm text-muted-foreground border-r border-border"
                          >
                            -
                          </td>
                        );
                      }

                      return (
                        <td
                          key={`${feature.key}-${leagueId}-${window.key}`}
                          className="px-3 py-2.5 border-r border-border"
                        >
                          <CoverageCell
                            pct={cell.pct}
                            n={cell.n}
                            matchesTotal={matchesTotal}
                            windowKey={window.key}
                            leagueName={league.name}
                            featureKey={feature.key}
                            isProd={isProd}
                            leagueId={league.league_id}
                            onCellClick={onCellClick}
                          />
                        </td>
                      );
                    }),
                    /* Total column */
                    <td
                      key={`${feature.key}-${leagueId}-total`}
                      className="px-3 py-2.5 border-r border-border"
                    >
                      {featureCoverage?.total ? (
                        <CoverageCell
                          pct={featureCoverage.total.pct}
                          n={featureCoverage.total.n}
                          matchesTotal={
                            isProd
                              ? (leagueSummary?.total?.matches_total_ft ?? 0)
                              : (leagueSummary?.total?.matches_total_titan ?? 0)
                          }
                          windowKey="Total"
                          leagueName={league.name}
                          featureKey={feature.key}
                          isProd={isProd}
                          leagueId={league.league_id}
                          onCellClick={onCellClick}
                        />
                      ) : (
                        <div className="text-center text-sm text-muted-foreground">
                          -
                        </div>
                      )}
                    </td>,
                    /* N column */
                    <td
                      key={`${feature.key}-${leagueId}-n`}
                      className="px-3 py-2.5 border-r border-border"
                    >
                      <NCell n={featureCoverage?.total?.n ?? 0} />
                    </td>,
                  ];
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
    </TooltipProvider>
  );
}
