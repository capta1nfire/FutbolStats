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
 * Color thresholds for coverage cells
 */
function getCoverageColor(pct: number): string {
  if (pct > 80) return "bg-green-500/20 text-green-400";
  if (pct >= 50) return "bg-yellow-500/20 text-yellow-400";
  return "bg-red-500/20 text-red-400";
}

/**
 * Badge component for tier labels
 */
function TierBadge({ badge }: { badge: string }) {
  const colors =
    badge === "PROD"
      ? "bg-green-500/20 text-green-400 border-green-500/30"
      : "bg-purple-500/20 text-purple-400 border-purple-500/30";

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
}: {
  pct: number;
  n: number;
  matchesTotal: number;
  windowKey: string;
  leagueName: string;
  featureKey: string;
  isProd: boolean;
}) {
  const totalLabel = isProd ? "Total (FT):" : "Total (TITAN):";

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              "text-center text-sm tabular-nums cursor-default",
              getCoverageColor(pct)
            )}
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
    </TooltipProvider>
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

export function FeatureCoverageMatrix({
  className,
  isLeagueVisible,
  onLeaguesLoaded,
  currentPage = 1,
  pageSize = 25,
  onTotalFeaturesChange,
  enabledTiers = DEFAULT_ENABLED_TIERS,
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
      const visibleLeagues = isLeagueVisible
        ? leagues.filter((l) => isLeagueVisible(l.league_id))
        : leagues;

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
    }, [data, enabledTiers, isLeagueVisible]);

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
    <div className={cn("flex-1 flex flex-col overflow-hidden", className)}>
      {/* Single scroll container for both header and body - syncs horizontal scroll */}
      <div className="flex-1 overflow-auto">
        <table className="w-full border-collapse text-sm">
          {/* Sticky header - position:sticky with top:0 keeps it in sync with horizontal scroll */}
          <thead className="sticky top-0 z-20 bg-background">
            {/* Header row 1: League names */}
            <tr className="border-b border-border">
              {/* Sticky Feature column header - both left and top sticky */}
              <th
                className="sticky left-0 z-30 px-3 pt-3 pb-2 text-left font-semibold text-muted-foreground text-sm align-bottom bg-background border-r border-border relative after:absolute after:top-0 after:right-0 after:bottom-0 after:w-4 after:translate-x-full after:bg-gradient-to-r after:from-black/20 after:to-transparent after:pointer-events-none"
                rowSpan={2}
                style={{ minWidth: "220px" }}
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
                    className="px-3 pb-2 text-center text-xs font-medium text-muted-foreground border-r border-border bg-background"
                  >
                    {window.key}
                  </th>
                )),
                <th
                  key={`${league.league_id}-total`}
                  className="px-3 pb-2 text-center text-xs font-medium text-muted-foreground border-r border-border bg-background"
                >
                  Total
                </th>,
                <th
                  key={`${league.league_id}-n`}
                  className="px-3 pb-2 text-center text-xs font-medium text-muted-foreground border-r border-border bg-background"
                >
                  N
                </th>,
              ])}
            </tr>
          </thead>

          {/* Body */}
          <tbody>
            {paginatedFeatures.map((feature) => (
              <tr
                key={feature.key}
                className="border-b border-border transition-colors hover:bg-accent/50"
              >
                {/* Sticky Feature cell */}
                <td
                  className="sticky left-0 z-10 px-3 py-2.5 bg-background border-r border-border relative after:absolute after:top-0 after:right-0 after:bottom-0 after:w-4 after:translate-x-full after:bg-gradient-to-r after:from-black/20 after:to-transparent after:pointer-events-none"
                  style={{ minWidth: "220px" }}
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

      {/* Footer info */}
      {data?.generated_at && (
        <div className="flex-shrink-0 flex items-center justify-end gap-2 px-3 py-2 text-[10px] text-muted-foreground border-t border-border">
          <span>
            Generated: {new Date(data.generated_at).toLocaleString()}
          </span>
          {data.cached && data.cache_age_seconds !== null && (
            <span>· Cached {Math.round(data.cache_age_seconds / 60)}m ago</span>
          )}
        </div>
      )}
    </div>
  );
}
