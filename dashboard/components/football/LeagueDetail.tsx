"use client";

import Image from "next/image";
import { useFootballLeague, useStandings } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import type { StandingEntry } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
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
} from "lucide-react";

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
                    className="hover:text-primary hover:underline transition-colors"
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
                    className="hover:text-primary hover:underline transition-colors"
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
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.position}</td>
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
                        className="text-foreground hover:text-primary hover:underline transition-colors text-left truncate"
                      >
                        {entry.teamName}
                      </button>
                    ) : (
                      <span className="text-foreground truncate">{entry.teamName}</span>
                    )}
                    {entry.description && (
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
                  entry.goalDiff > 0 ? "text-green-500" : entry.goalDiff < 0 ? "text-red-500" : "text-muted-foreground"
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
                            ch === "W" ? "bg-green-500" : ch === "D" ? "bg-yellow-500" : ch === "L" ? "bg-red-500" : "bg-muted"
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
export function LeagueDetail({ leagueId, onBack, onTeamSelect }: LeagueDetailProps) {
  const { data, isLoading, error, refetch } = useFootballLeague(leagueId);
  const { data: standingsData, isLoading: isStandingsLoading } = useStandings(leagueId);

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
          <AlertTriangle className="h-12 w-12 text-yellow-400" />
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
        {/* Header with Back button */}
        <div className="flex items-start gap-4">
          <Button variant="ghost" size="icon" onClick={onBack} className="shrink-0 mt-1">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <Trophy className="h-5 w-5 text-amber-500" />
              <h1 className="text-lg font-semibold text-foreground">{league.name}</h1>
            </div>
            <div className="flex items-center gap-3 mt-1 text-sm text-muted-foreground">
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
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        {/* Group Info */}
        {group && (
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-purple-500" />
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

        {/* League Meta */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-card border border-border rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Match Type</p>
            <p className="text-sm font-medium text-foreground capitalize">
              {league.match_type || "-"}
            </p>
          </div>
          {league.match_weight !== null && (
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="text-xs text-muted-foreground">Match Weight</p>
              <p className="text-sm font-medium text-foreground">{league.match_weight}</p>
            </div>
          )}
        </div>

        {/* TITAN Coverage */}
        {titan && (
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="h-4 w-4 text-purple-500" />
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

        {/* Stats by Season */}
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Stats by Season</h2>
          </div>
          <StatsTable stats={stats_by_season} />
        </div>

        {/* Recent Matches */}
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Recent Matches</h2>
          </div>
          <RecentMatchesList matches={recent_matches} onTeamSelect={onTeamSelect} />
        </div>

        {/* Standings */}
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <Trophy className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Standings</h2>
            {standingsData && (
              <span className="text-xs text-muted-foreground ml-auto">
                {standingsData.season} &middot; {standingsData.source}
              </span>
            )}
            {standingsData?.isPlaceholder && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/15 text-yellow-600">Placeholder</span>
            )}
            {standingsData?.isCalculated && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-600">Calculated</span>
            )}
          </div>
          {isStandingsLoading ? (
            <div className="flex items-center justify-center py-6">
              <Loader size="sm" />
            </div>
          ) : standingsData && standingsData.standings.length > 0 ? (
            <StandingsTable standings={standingsData.standings} onTeamSelect={onTeamSelect} />
          ) : (
            <div className="px-4 py-4 text-sm text-muted-foreground">
              No standings available for this league
            </div>
          )}
        </div>
      </div>
    </ScrollArea>
  );
}
