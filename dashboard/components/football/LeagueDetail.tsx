"use client";

import Image from "next/image";
import { useCallback, useMemo, useState } from "react";
import { useFootballLeague, useFootballTeam, useStandings } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import type { StandingEntry } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { IconTabs } from "@/components/ui/icon-tabs";
import { QualificationBadge } from "@/components/ui/qualification-badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Loader } from "@/components/ui/loader";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  RefreshCw,
  AlertTriangle,
  ArrowLeft,
  Trophy,
  Globe,
  Calendar,
  BarChart3,
  Users,
  TrendingUp,
  MapPin,
  Building2,
  Settings,
} from "lucide-react";
import { toast } from "sonner";

const LEAGUE_DETAIL_TABS = [
  { id: "standings", icon: null, label: "Standings" },
  { id: "next", icon: null, label: "Next Matches" },
  { id: "stats", icon: null, label: "Stats by Season" },
] as const;

type LeagueDetailTabId = (typeof LEAGUE_DETAIL_TABS)[number]["id"];

const TEAM_DETAIL_TABS = [
  { id: "overview", label: "Overview" },
  { id: "matches", label: "Matches" },
  { id: "stats", label: "Stats" },
  { id: "transfers", label: "Transfers" },
] as const;

type TeamDetailTabId = (typeof TEAM_DETAIL_TABS)[number]["id"];

/**
 * Country Flag Component
 */
function CountryFlag({ country, size = 14 }: { country: string; size?: number }) {
  const isoCode = getCountryIsoCode(country);

  if (!isoCode) {
    return <Globe className="text-muted-foreground" style={{ width: size, height: size }} />;
  }

  return (
    <Image
      src={`/flags/${isoCode}.svg`}
      alt={`${country} flag`}
      width={size}
      height={size}
      className="rounded-full object-cover"
    />
  );
}

interface LeagueDetailProps {
  leagueId: number;
  onBack: () => void;
  onTeamSelect: (teamId: number) => void;
  onSettingsClick?: () => void;
}

/**
 * Stats Table Component
 */
function StatsTable({
  stats,
}: {
  stats: {
    season: number;
    total_matches: number;
    finished: number;
    with_stats_pct: number | null;
    with_odds_pct: number | null;
  }[];
}) {
  if (!stats || stats.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No season stats available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground">Season</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Matches</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Finished</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Stats %</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Odds %</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((s) => (
            <tr key={s.season} className="border-b border-border last:border-0">
              <td className="py-2 px-3 font-medium text-foreground">{s.season}</td>
              <td className="py-2 px-3 text-right text-muted-foreground">{s.total_matches}</td>
              <td className="py-2 px-3 text-right text-muted-foreground">{s.finished}</td>
              <td className="py-2 px-3 text-right text-muted-foreground">
                {s.with_stats_pct !== null ? `${s.with_stats_pct.toFixed(0)}%` : "-"}
              </td>
              <td className="py-2 px-3 text-right text-muted-foreground">
                {s.with_odds_pct !== null ? `${s.with_odds_pct.toFixed(0)}%` : "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Recent Matches List
 */
function RecentMatchesList({
  matches,
  onTeamSelect,
}: {
  matches: {
    match_id: number;
    date: string | null;
    status: string;
    home_team: string;
    away_team: string;
    home_team_id?: number;
    away_team_id?: number;
    score: string | null;
  }[];
  onTeamSelect?: (teamId: number) => void;
}) {
  if (!matches || matches.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No recent matches
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {matches.map((match) => {
        const dateStr = match.date
          ? new Date(match.date).toLocaleDateString([], { month: "short", day: "numeric" })
          : "TBD";
        const timeStr = match.date
          ? new Date(match.date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
          : "--:--";

        return (
          <div key={match.match_id} className="py-2 px-3 flex items-center gap-3">
            <div className="w-20 shrink-0">
              <p className="text-xs text-muted-foreground">{dateStr}</p>
              <p className="text-xs text-muted-foreground">{timeStr}</p>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-foreground truncate">
                {match.home_team_id && onTeamSelect ? (
                  <button
                    onClick={() => onTeamSelect(match.home_team_id!)}
                    className="text-primary hover:text-primary-hover transition-colors no-underline hover:no-underline"
                  >
                    {match.home_team}
                  </button>
                ) : (
                  <span>{match.home_team}</span>
                )}
                <span className="text-muted-foreground"> vs </span>
                {match.away_team_id && onTeamSelect ? (
                  <button
                    onClick={() => onTeamSelect(match.away_team_id!)}
                    className="text-primary hover:text-primary-hover transition-colors no-underline hover:no-underline"
                  >
                    {match.away_team}
                  </button>
                ) : (
                  <span>{match.away_team}</span>
                )}
              </p>
            </div>
            <div className="shrink-0 text-right">
              {match.score ? (
                <span className="text-sm font-medium text-foreground">{match.score}</span>
              ) : (
                <span className="text-xs px-1.5 py-0.5 bg-muted rounded text-muted-foreground capitalize">
                  {match.status}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Standings Table Component
 */
function StandingsTable({
  standings,
  onTeamSelect,
}: {
  standings: StandingEntry[];
  onTeamSelect?: (teamId: number) => void;
}) {
  if (standings.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No standings data
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">#</th>
            <th className="text-left py-2 px-2 text-xs font-medium text-muted-foreground">Team</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">P</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">W</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">D</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">L</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">GD</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">Pts</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-20">Form</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((entry) => {
            const teamClickable = entry.teamId > 0 && onTeamSelect;

            return (
              <tr key={entry.position} className="border-b border-border last:border-0 hover:bg-muted/30">
                <td className="text-center py-1.5 px-2">
                  <QualificationBadge value={entry.position} active={entry.position <= 8} />
                </td>
                <td className="py-1.5 px-2">
                  <div className="flex items-center gap-2">
                    {entry.teamLogo ? (
                      <Image src={entry.teamLogo} alt="" width={16} height={16} className="rounded-full object-cover" />
                    ) : (
                      <Globe className="h-4 w-4 text-muted-foreground" />
                    )}
                    {teamClickable ? (
                      <button
                        onClick={() => onTeamSelect(entry.teamId)}
                        className="text-primary hover:text-primary-hover transition-colors no-underline hover:no-underline text-left truncate"
                      >
                        {entry.teamName}
                      </button>
                    ) : (
                      <span className="text-foreground truncate">{entry.teamName}</span>
                    )}
                    {entry.description && entry.position > 8 && (
                      <span className="text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground shrink-0">
                        {entry.description}
                      </span>
                    )}
                  </div>
                </td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.played}</td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.won}</td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.drawn}</td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.lost}</td>
                <td className={cn(
                  "text-center py-1.5 px-2 font-medium",
                  entry.goalDiff > 0 ? "text-[var(--status-success-text)]" : entry.goalDiff < 0 ? "text-[var(--status-error-text)]" : "text-muted-foreground"
                )}>
                  {entry.goalDiff > 0 ? `+${entry.goalDiff}` : entry.goalDiff}
                </td>
                <td className="text-center py-1.5 px-2 font-bold text-foreground">{entry.points}</td>
                <td className="text-center py-1.5 px-2">
                  {entry.form ? (
                    <div className="flex items-center justify-center gap-0.5">
                      {entry.form.split("").slice(-5).map((ch, i) => (
                        <span
                          key={i}
                          className={cn(
                            "w-2 h-2 rounded-full",
                            ch === "W" ? "bg-[var(--status-success-text)]" : ch === "D" ? "bg-muted-foreground" : ch === "L" ? "bg-[var(--status-error-text)]" : "bg-muted"
                          )}
                        />
                      ))}
                    </div>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * LeagueDetail Component
 *
 * Displays detailed league information:
 * - League info (name, country, kind, priority)
 * - Group info if part of a group
 * - Stats by season table
 * - TITAN coverage
 * - Recent matches
 * - Standings
 */
export function LeagueDetail({ leagueId, onBack, onTeamSelect, onSettingsClick }: LeagueDetailProps) {
  const [activeTab, setActiveTab] = useState<LeagueDetailTabId>("standings");
  const [selectedTeamId, setSelectedTeamId] = useState<number | null>(null);
  const [teamTab, setTeamTab] = useState<TeamDetailTabId>("overview");
  const { data, isLoading, error, refetch } = useFootballLeague(leagueId);
  const { data: standingsData, isLoading: isStandingsLoading } = useStandings(leagueId);
  const selectedTeam = useFootballTeam(selectedTeamId);

  // MUST be called on every render (Rules of Hooks). Keep defensive for loading/error.
  const nextMatches = useMemo(() => {
    const matches = data?.recent_matches ?? [];
    // Treat "no score" as upcoming. Status may vary; keep it defensive.
    return matches.filter((m) => !m.score);
  }, [data?.recent_matches]);

  const handleTeamSelect = useCallback(
    (teamId: number) => {
      setSelectedTeamId(teamId);
      onTeamSelect(teamId);
    },
    [onTeamSelect]
  );

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader size="md" />
      </div>
    );
  }

  // Error state
  if (error || !data) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center max-w-md">
          <AlertTriangle className="h-12 w-12 text-[var(--status-warning-text)]" />
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-2">
              League Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || `Unable to fetch league ${leagueId}`}
            </p>
          </div>
          <div className="flex gap-2">
            <Button onClick={onBack} variant="outline">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button onClick={() => refetch()} variant="secondary">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const { league, group, stats_by_season, titan, recent_matches } = data;

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        <div className="w-full flex flex-col lg:flex-row gap-4 lg:items-start min-w-0">
          {/* League container: header + tabs + tab content */}
          <div
            className={cn(
              "rounded-lg border border-border overflow-hidden w-full min-w-0",
              selectedTeamId !== null ? "lg:w-[30%]" : "lg:max-w-[30%] lg:mr-auto"
            )}
          >
            {/* League header */}
            <div className="px-4 py-3 border-b border-border flex items-start gap-4">
              <Button variant="ghost" size="icon" onClick={onBack} className="shrink-0 mt-0.5">
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <Trophy className="h-5 w-5 text-primary" />
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <h1 className="text-lg font-semibold text-foreground cursor-help truncate">
                        {league.name}
                      </h1>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" sideOffset={6}>
                      <div className="space-y-1">
                        <div className="text-foreground">
                          <span className="text-muted-foreground">Match Weight:</span>{" "}
                          <span className="font-medium text-foreground">
                            {league.match_weight ?? 1}
                          </span>
                        </div>
                        <div className="text-muted-foreground">
                          1 = peso estándar. Valores mayores aumentan la influencia de esta liga
                          en cálculos internos; menores la reducen.
                        </div>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <div className="flex items-center gap-3 mt-1 text-sm text-muted-foreground flex-wrap">
                  <span className="flex items-center gap-1">
                    <CountryFlag country={league.country} size={14} />
                    {league.country}
                  </span>
                  {league.kind && <span className="capitalize">{league.kind}</span>}
                  {league.priority && (
                    <span className="px-1.5 py-0.5 bg-muted rounded text-xs">
                      {league.priority}
                    </span>
                  )}
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => refetch()} className="shrink-0">
                <RefreshCw className="h-4 w-4" />
              </Button>
              {onSettingsClick && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onSettingsClick}
                  className="shrink-0 h-8 w-8"
                  aria-label="League settings"
                >
                  <Settings className="h-4 w-4" />
                </Button>
              )}
            </div>

            {/* Tabs */}
            <div className="px-4 py-3 border-b border-border">
              <IconTabs
                tabs={
                  LEAGUE_DETAIL_TABS as unknown as { id: string; icon: React.ReactNode; label: string }[]
                }
                value={activeTab}
                onValueChange={(v) => setActiveTab(v as LeagueDetailTabId)}
                showLabels
                className="w-full"
              />
            </div>

            {/* Tab content */}
            {activeTab === "standings" && (
              <div>
                <div className="px-4 py-3 border-b border-border flex items-center gap-2">
                  <Trophy className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-semibold text-foreground">Standings</h2>
                  {standingsData && (
                    <span className="text-xs text-muted-foreground ml-auto">
                      {standingsData.season} &middot; {standingsData.source}
                    </span>
                  )}
                  {standingsData?.isPlaceholder && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]">
                      Placeholder
                    </span>
                  )}
                  {standingsData?.isCalculated && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--status-info-bg)] text-[var(--status-info-text)]">
                      Calculated
                    </span>
                  )}
                </div>
                {isStandingsLoading ? (
                  <div className="flex items-center justify-center py-6">
                    <Loader size="sm" />
                  </div>
                ) : standingsData && standingsData.standings.length > 0 ? (
                  <StandingsTable
                    standings={standingsData.standings}
                    onTeamSelect={handleTeamSelect}
                  />
                ) : (
                  <div className="px-4 py-4 text-sm text-muted-foreground">
                    No standings available for this league
                  </div>
                )}
              </div>
            )}

            {activeTab === "next" && (
              <div>
                <div className="px-4 py-3 border-b border-border flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-semibold text-foreground">Next Matches</h2>
                  <span className="text-xs text-muted-foreground ml-auto">
                    {nextMatches.length}
                  </span>
                </div>
                <RecentMatchesList matches={nextMatches} onTeamSelect={handleTeamSelect} />
              </div>
            )}

            {activeTab === "stats" && (
              <div>
                <div className="px-4 py-3 border-b border-border flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-muted-foreground" />
                  <h2 className="text-sm font-semibold text-foreground">Stats by Season</h2>
                </div>
                <StatsTable stats={stats_by_season} />
              </div>
            )}
          </div>

          {/* Club container (only when selected) */}
          {selectedTeamId !== null && (
            <div className="flex flex-col gap-4 w-full lg:w-[48%] min-w-0">
              {/* Team Header */}
              <div className="rounded-lg border border-border px-4 py-3 flex items-center gap-3">
                {selectedTeam.data?.team?.logo_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={selectedTeam.data.team.logo_url}
                    alt=""
                    className="h-24 w-24 object-contain"
                  />
                ) : (
                  <div className="h-24 w-24 rounded-full bg-muted" />
                )}
                <div className="min-w-0 flex-1">
                  <h2 className="text-2xl font-semibold text-foreground truncate">
                    {selectedTeam.data?.team?.display_name ?? selectedTeam.data?.team?.name ?? "Team details"}
                  </h2>
                  {(selectedTeam.data?.wikidata_enrichment?.city || selectedTeam.data?.team?.country) && (
                    <div className="flex items-center gap-1.5 text-sm text-muted-foreground mt-1">
                      <MapPin className="h-3.5 w-3.5" />
                      <span>
                        {[selectedTeam.data?.wikidata_enrichment?.city, selectedTeam.data?.team?.country]
                          .filter(Boolean)
                          .join(", ")}
                      </span>
                    </div>
                  )}
                  {selectedTeam.data?.wikidata_enrichment?.stadium_name && (
                    <div className="flex items-center gap-1.5 text-sm text-muted-foreground mt-0.5">
                      <Building2 className="h-3.5 w-3.5" />
                      <span>
                        {selectedTeam.data.wikidata_enrichment.stadium_name}
                        {selectedTeam.data.wikidata_enrichment.stadium_capacity && (
                          <span className="ml-1">
                            ({selectedTeam.data.wikidata_enrichment.stadium_capacity.toLocaleString()})
                          </span>
                        )}
                      </span>
                    </div>
                  )}
                </div>
                {selectedTeam.isLoading && <Loader size="sm" />}
              </div>

              {/* Team Tabs */}
              <div className="flex gap-1 border-b border-border">
                {TEAM_DETAIL_TABS.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setTeamTab(tab.id)}
                    className={cn(
                      "px-4 py-2 text-sm font-medium transition-colors",
                      teamTab === tab.id
                        ? "text-foreground border-b-2 border-primary"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              {teamTab === "overview" && (
                <>
                  {!selectedTeam.isLoading && selectedTeam.data?.wikidata_enrichment ? (
                    <div className="rounded-lg border border-border overflow-hidden">
                      <table className="w-full text-sm">
                        <tbody className="divide-y divide-border">
                          {[
                            ["Team ID", selectedTeamId?.toString()],
                            ["Full name", selectedTeam.data.wikidata_enrichment.full_name],
                            ["Short name", selectedTeam.data.wikidata_enrichment.short_name],
                            ["City", selectedTeam.data.wikidata_enrichment.city],
                            ["Stadium", selectedTeam.data.wikidata_enrichment.stadium_name],
                            [
                              "Capacity",
                              selectedTeam.data.wikidata_enrichment.stadium_capacity != null
                                ? selectedTeam.data.wikidata_enrichment.stadium_capacity.toLocaleString()
                                : null,
                            ],
                            [
                              "Altitude (m)",
                              selectedTeam.data.wikidata_enrichment.stadium_altitude_m != null
                                ? selectedTeam.data.wikidata_enrichment.stadium_altitude_m.toLocaleString()
                                : null,
                            ],
                            [
                              "Coords",
                              selectedTeam.data.wikidata_enrichment.lat != null &&
                              selectedTeam.data.wikidata_enrichment.lon != null
                                ? `${selectedTeam.data.wikidata_enrichment.lat.toFixed(6)}, ${selectedTeam.data.wikidata_enrichment.lon.toFixed(6)}`
                                : null,
                            ],
                            ["Updated", selectedTeam.data.wikidata_enrichment.wikidata_updated_at],
                          ].map(([label, value]) => (
                            <tr key={label}>
                              <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                                {label}
                              </td>
                              <td className="py-2 px-3 text-sm text-foreground">
                                {value ? (
                                  label === "Team ID" ? (
                                    <button
                                      onClick={() => {
                                        navigator.clipboard.writeText(value);
                                        toast.success("Team ID copied");
                                      }}
                                      className="text-primary hover:opacity-80 transition-opacity"
                                    >
                                      {value}
                                    </button>
                                  ) : (
                                    <span className="break-words">{value}</span>
                                  )
                                ) : (
                                  <span className="text-muted-foreground">-</span>
                                )}
                              </td>
                            </tr>
                          ))}
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Website
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.website ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={selectedTeam.data.wikidata_enrichment.website}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    {selectedTeam.data.wikidata_enrichment.website}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              X.com (Twitter)
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.twitter ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={`https://x.com/${selectedTeam.data.wikidata_enrichment.twitter}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    @{selectedTeam.data.wikidata_enrichment.twitter}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Instagram
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.instagram ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={`https://www.instagram.com/${selectedTeam.data.wikidata_enrichment.instagram}/`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    @{selectedTeam.data.wikidata_enrichment.instagram}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Wikipedia
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.team.wiki?.wiki_url_cached ||
                              selectedTeam.data.team.wiki?.wiki_url ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={
                                      selectedTeam.data.team.wiki?.wiki_url_cached ||
                                      selectedTeam.data.team.wiki?.wiki_url ||
                                      "#"
                                    }
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    {selectedTeam.data.team.wiki?.wiki_url_cached ||
                                      selectedTeam.data.team.wiki?.wiki_url}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Wikidata ID
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.wikidata_id ? (
                                <span className="break-words">
                                  {selectedTeam.data.wikidata_enrichment.wikidata_id}
                                </span>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Stadium Wikidata
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.stadium_wikidata_id ? (
                                <span className="break-words">
                                  {selectedTeam.data.wikidata_enrichment.stadium_wikidata_id}
                                </span>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  ) : !selectedTeam.isLoading ? (
                    <div className="rounded-lg border border-border px-4 py-4 text-sm text-muted-foreground">
                      No enrichment data for this team
                    </div>
                  ) : null}
                </>
              )}

              {teamTab === "matches" && (
                <div className="rounded-lg border border-border px-4 py-8 text-center">
                  <Calendar className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Matches coming soon</p>
                </div>
              )}

              {teamTab === "stats" && (
                <div className="rounded-lg border border-border px-4 py-8 text-center">
                  <BarChart3 className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Stats coming soon</p>
                </div>
              )}

              {teamTab === "transfers" && (
                <div className="rounded-lg border border-border px-4 py-8 text-center">
                  <TrendingUp className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Transfers coming soon</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Group Info */}
        {group && (
          <div className="bg-[var(--tag-purple-bg)] border border-[var(--tag-purple-border)] rounded-lg p-4">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-[var(--tag-purple-text)]" />
              <span className="text-sm font-medium text-foreground">Part of Group</span>
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              {group.name} ({group.key})
            </p>
            {group.paired_handling && (
              <p className="text-xs text-muted-foreground mt-1">
                Paired handling: {group.paired_handling}
              </p>
            )}
          </div>
        )}

        {/* TITAN Coverage */}
        {titan && (
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="h-4 w-4 text-[var(--tag-purple-text)]" />
              <h2 className="text-sm font-semibold text-foreground">TITAN Coverage</h2>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <p className="text-xl font-semibold text-foreground">{titan.tier1_pct.toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Tier 1</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{titan.tier1}</p>
                <p className="text-xs text-muted-foreground">Tier 1 count</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{titan.tier1b}</p>
                <p className="text-xs text-muted-foreground">Tier 1b</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{titan.tier1c}</p>
                <p className="text-xs text-muted-foreground">Tier 1c</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{titan.tier1d}</p>
                <p className="text-xs text-muted-foreground">Tier 1d</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Total: {titan.total.toLocaleString()} matches
            </p>
          </div>
        )}

      </div>
    </ScrollArea>
  );
}
