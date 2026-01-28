"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  useIsDesktop,
  useFootballTournaments,
  useFootballCountries,
} from "@/lib/hooks";
import type {
  FeatureCoverageCellSelection,
  TournamentEntry,
} from "@/lib/types";
import type { FeatureCoverageResponse } from "@/lib/hooks/use-feature-coverage";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { IconTabs } from "@/components/ui/icon-tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  BarChart3,
  Trophy,
  ExternalLink,
  Calendar,
  Users,
  TrendingUp,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

const DETAIL_TABS = [
  { id: "overview", icon: <BarChart3 />, label: "Overview" },
  { id: "tournament", icon: <Trophy />, label: "Tournament" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getCoverageColor(pct: number): string {
  if (pct >= 95) return "text-[var(--status-success-text)]";
  if (pct >= 80) return "text-[var(--status-success-text)]/70";
  if (pct >= 50) return "text-[var(--status-warning-text)]";
  if (pct > 0) return "text-[var(--status-error-text)]";
  return "text-muted-foreground";
}

function formatPct(pct: number): string {
  if (pct >= 100) return "100%";
  if (pct === 0) return "0%";
  return `${pct.toFixed(1)}%`;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface FeatureCoverageDetailDrawerProps {
  selection: FeatureCoverageCellSelection | null;
  open: boolean;
  onClose: () => void;
  coverageData: FeatureCoverageResponse["data"] | null;
}

// ---------------------------------------------------------------------------
// Tab Content
// ---------------------------------------------------------------------------

function DetailTabContent({
  selection,
  coverageData,
  activeTab,
}: {
  selection: FeatureCoverageCellSelection;
  coverageData: FeatureCoverageResponse["data"];
  activeTab: string;
}) {
  const router = useRouter();

  // Find feature metadata
  const feature = useMemo(
    () => coverageData.features.find((f) => f.key === selection.featureKey),
    [coverageData.features, selection.featureKey]
  );

  // Find league
  const league = useMemo(
    () => coverageData.leagues.find((l) => l.league_id === selection.leagueId),
    [coverageData.leagues, selection.leagueId]
  );

  const leagueIdStr = String(selection.leagueId);
  const featureCoverage = coverageData.coverage[selection.featureKey]?.[leagueIdStr];
  const leagueSummary = coverageData.league_summaries[leagueIdStr];

  // Tier determines denominator: tier1 = PROD (FT), others = TITAN
  const isProd = feature?.tier_id === "tier1";
  const totalLabel = isProd ? "Total (FT)" : "Total (TITAN)";

  // Tournament data (lazy fetch via TanStack Query - cached)
  const { data: tournamentsData, isLoading: isTournamentsLoading } =
    useFootballTournaments();

  const tournament: TournamentEntry | null = useMemo(() => {
    if (!tournamentsData?.tournaments) return null;
    return (
      tournamentsData.tournaments.find(
        (t) => t.league_id === selection.leagueId
      ) ?? null
    );
  }, [tournamentsData, selection.leagueId]);

  // Countries data for league_id → country mapping (domestic leagues)
  const { data: countriesData } = useFootballCountries();

  const leagueCountry: string | null = useMemo(() => {
    if (!countriesData?.countries) return null;
    for (const c of countriesData.countries) {
      if (c.leagues.some((l) => l.league_id === selection.leagueId)) {
        return c.country;
      }
    }
    return null;
  }, [countriesData, selection.leagueId]);

  return (
    <div className="w-full">
      {/* Overview Tab */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          {/* Feature + League header */}
          <div className="bg-surface rounded-lg p-4 space-y-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                {feature && (
                  <span
                    className={cn(
                      "text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded",
                      feature.badge === "PROD"
                        ? "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)]"
                        : "bg-[var(--tag-indigo-bg)] text-[var(--tag-indigo-text)]"
                    )}
                  >
                    {feature.badge}
                  </span>
                )}
                <span className="text-sm font-mono text-foreground">
                  {selection.featureKey}
                </span>
              </div>
              <p className="text-sm text-muted-foreground">
                {league?.name ?? `League ${selection.leagueId}`}
              </p>
            </div>

            {feature && (
              <div className="pt-2 border-t border-border text-xs text-muted-foreground">
                <span>Source: </span>
                <span className="text-foreground font-mono">
                  {feature.source}
                </span>
              </div>
            )}
          </div>

          {/* Coverage breakdown by window */}
          <div className="bg-surface rounded-lg p-4">
            <h3 className="text-sm font-medium text-foreground mb-3">
              Coverage by Season
            </h3>
            <div className="overflow-hidden rounded border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-background">
                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                      Window
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                      Coverage
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                      Non-null
                    </th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                      {totalLabel}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {/* Dynamic windows from backend */}
                  {coverageData.windows.map((window) => {
                    const cell = featureCoverage?.[window.key];
                    const summaryForWindow = leagueSummary?.[window.key];
                    const matchesTotal = isProd
                      ? (summaryForWindow?.matches_total_ft ?? 0)
                      : (summaryForWindow?.matches_total_titan ?? 0);

                    return (
                      <tr
                        key={window.key}
                        className="border-b border-border last:border-b-0"
                      >
                        <td className="px-3 py-2 text-foreground">
                          {window.key}
                        </td>
                        <td
                          className={cn(
                            "px-3 py-2 text-right tabular-nums font-medium",
                            cell ? getCoverageColor(cell.pct) : "text-muted-foreground"
                          )}
                        >
                          {cell ? formatPct(cell.pct) : "-"}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          {cell?.n?.toLocaleString() ?? "-"}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          {matchesTotal > 0
                            ? matchesTotal.toLocaleString()
                            : "-"}
                        </td>
                      </tr>
                    );
                  })}

                  {/* Total row */}
                  {(() => {
                    const totalCell = featureCoverage?.total;
                    const totalSummary = leagueSummary?.total;
                    const matchesTotal = isProd
                      ? (totalSummary?.matches_total_ft ?? 0)
                      : (totalSummary?.matches_total_titan ?? 0);

                    return (
                      <tr className="bg-background font-medium">
                        <td className="px-3 py-2 text-foreground">Total</td>
                        <td
                          className={cn(
                            "px-3 py-2 text-right tabular-nums",
                            totalCell
                              ? getCoverageColor(totalCell.pct)
                              : "text-muted-foreground"
                          )}
                        >
                          {totalCell ? formatPct(totalCell.pct) : "-"}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          {totalCell?.n?.toLocaleString() ?? "-"}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          {matchesTotal > 0
                            ? matchesTotal.toLocaleString()
                            : "-"}
                        </td>
                      </tr>
                    );
                  })()}
                </tbody>
              </table>
            </div>
          </div>

          {/* League summary */}
          {leagueSummary?.total && (
            <div className="bg-surface rounded-lg p-4">
              <h3 className="text-sm font-medium text-foreground mb-2">
                League Summary
              </h3>
              <div className="flex items-center gap-2 text-sm">
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Avg Coverage:</span>
                <span
                  className={cn(
                    "font-medium tabular-nums",
                    getCoverageColor(leagueSummary.total.avg_pct)
                  )}
                >
                  {formatPct(leagueSummary.total.avg_pct)}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tournament Tab */}
      {activeTab === "tournament" && (
        <div className="space-y-4">
          {isTournamentsLoading ? (
            <div className="space-y-3">
              <div className="h-8 bg-surface rounded animate-pulse" />
              <div className="h-24 bg-surface rounded animate-pulse" />
            </div>
          ) : tournament ? (
            <>
              {/* Tournament info card */}
              <div className="bg-surface rounded-lg p-4 space-y-3">
                <div className="space-y-1">
                  <h3 className="text-sm font-semibold text-foreground">
                    {tournament.name}
                  </h3>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span
                      className={cn(
                        "text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded",
                        tournament.kind === "international"
                          ? "bg-[var(--tag-purple-bg)] text-[var(--tag-purple-text)]"
                          : tournament.kind === "cup"
                          ? "bg-[var(--tag-orange-bg)] text-[var(--tag-orange-text)]"
                          : "bg-[var(--tag-gray-bg)] text-[var(--tag-gray-text)]"
                      )}
                    >
                      {tournament.kind}
                    </span>
                    <span
                      className={cn(
                        "text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded",
                        tournament.priority === "high"
                          ? "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)]"
                          : tournament.priority === "medium"
                          ? "bg-[var(--tag-orange-bg)] text-[var(--tag-orange-text)]"
                          : "bg-[var(--tag-gray-bg)] text-[var(--tag-gray-text)]"
                      )}
                    >
                      {tournament.priority}
                    </span>
                    {tournament.country && (
                      <span className="text-xs text-muted-foreground">
                        {tournament.country}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Tournament stats */}
              <div className="bg-surface rounded-lg p-4 space-y-3">
                <h3 className="text-sm font-medium text-foreground">
                  Tournament Stats
                </h3>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    <BarChart3 className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      Total Matches:
                    </span>
                    <span className="text-foreground tabular-nums">
                      {tournament.stats.total_matches.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Calendar className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Last 30d:</span>
                    <span className="text-foreground tabular-nums">
                      {tournament.stats.matches_30d.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Users className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">
                      Participants:
                    </span>
                    <span className="text-foreground tabular-nums">
                      {tournament.stats.participants_count.toLocaleString()}
                    </span>
                  </div>
                  {tournament.stats.with_stats_pct != null && (
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        Stats Coverage:
                      </span>
                      <span
                        className={cn(
                          "tabular-nums font-medium",
                          getCoverageColor(tournament.stats.with_stats_pct)
                        )}
                      >
                        {formatPct(tournament.stats.with_stats_pct)}
                      </span>
                    </div>
                  )}
                  {tournament.stats.with_odds_pct != null && (
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        Odds Coverage:
                      </span>
                      <span
                        className={cn(
                          "tabular-nums font-medium",
                          getCoverageColor(tournament.stats.with_odds_pct)
                        )}
                      >
                        {formatPct(tournament.stats.with_odds_pct)}
                      </span>
                    </div>
                  )}
                  {tournament.stats.seasons_range && (
                    <div className="flex items-center gap-2 text-sm">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Seasons:</span>
                      <span className="text-foreground tabular-nums">
                        {tournament.stats.seasons_range[0]} &ndash;{" "}
                        {tournament.stats.seasons_range[1]}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Navigation to Football */}
              <Button
                variant="secondary"
                size="sm"
                className="w-full gap-2"
                onClick={() =>
                  router.push(
                    `/football?category=tournaments_competitions&league=${selection.leagueId}`
                  )
                }
              >
                <ExternalLink className="h-4 w-4" />
                Ver en Football
              </Button>
            </>
          ) : (
            /* No tournament match → domestic league, navigate via leagues_by_country */
            <div className="bg-surface rounded-lg p-4 text-center space-y-3">
              <Trophy className="h-8 w-8 text-muted-foreground mx-auto" />
              <p className="text-sm text-muted-foreground">
                Not a tournament/cup. This is a domestic league.
              </p>
              <Button
                variant="secondary"
                size="sm"
                className="w-full gap-2"
                onClick={() => {
                  const params = new URLSearchParams({
                    category: "leagues_by_country",
                    league: String(selection.leagueId),
                  });
                  if (leagueCountry) {
                    params.set("country", leagueCountry);
                  }
                  router.push(`/football?${params.toString()}`);
                }}
              >
                <ExternalLink className="h-4 w-4" />
                Ver en Football
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mobile wrapper (tabs + content together)
// ---------------------------------------------------------------------------

function DetailContentMobile({
  selection,
  coverageData,
}: {
  selection: FeatureCoverageCellSelection;
  coverageData: FeatureCoverageResponse["data"];
}) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="w-full space-y-3">
      <IconTabs
        tabs={DETAIL_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />
      <DetailTabContent
        selection={selection}
        coverageData={coverageData}
        activeTab={activeTab}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Feature Coverage Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer with tabs in fixedContent
 * Mobile/Tablet (<1280px): Sheet overlay
 *
 * Follows same pattern as DataQualityDetailDrawer.
 */
export function FeatureCoverageDetailDrawer({
  selection,
  open,
  onClose,
  coverageData,
}: FeatureCoverageDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const [activeTab, setActiveTab] = useState("overview");

  const league = useMemo(() => {
    if (!selection || !coverageData) return null;
    return coverageData.leagues.find(
      (l) => l.league_id === selection.leagueId
    );
  }, [selection, coverageData]);

  const drawerTitle = league?.name ?? "Coverage Detail";

  // Desktop: overlay drawer with tabs in fixedContent
  if (isDesktop) {
    return (
      <DetailDrawer
        open={open}
        onClose={onClose}
        title={drawerTitle}
        fixedContent={
          selection &&
          coverageData && (
            <IconTabs
              tabs={DETAIL_TABS}
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            />
          )
        }
      >
        {selection && coverageData ? (
          <DetailTabContent
            selection={selection}
            coverageData={coverageData}
            activeTab={activeTab}
          />
        ) : (
          <p className="text-muted-foreground text-sm">
            Click a coverage cell to view details
          </p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {drawerTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {selection && coverageData ? (
              <DetailContentMobile
                selection={selection}
                coverageData={coverageData}
              />
            ) : (
              <p className="text-muted-foreground text-sm">
                Click a coverage cell to view details
              </p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
