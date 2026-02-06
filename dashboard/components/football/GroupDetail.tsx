"use client";

import Image from "next/image";
import { useState } from "react";
import { useFootballGroup, useStandings } from "@/lib/hooks";
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
  Users,
  Calendar,
  BarChart3,
  TrendingUp,
  ChevronRight,
  Globe,
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

interface GroupDetailProps {
  groupId: number;
  onBack: () => void;
  onLeagueSelect: (leagueId: number) => void;
  onTeamSelect: (teamId: number) => void;
}

/**
 * Member League Card
 */
function MemberLeagueCard({
  league,
  onClick,
}: {
  league: {
    league_id: number;
    name: string;
    kind: string;
    priority: string;
    match_type: string;
  };
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left bg-muted/50 border border-border rounded-lg p-3 hover:border-border-hover hover:bg-muted transition-colors group"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <Trophy className="h-4 w-4 text-primary shrink-0" />
          <span className="text-sm font-medium text-foreground truncate">
            {league.name}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {league.kind && (
            <span className="text-xs text-muted-foreground capitalize">
              {league.kind}
            </span>
          )}
          {league.priority && (
            <span className="text-xs px-1.5 py-0.5 bg-background rounded">
              {league.priority}
            </span>
          )}
          <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
        </div>
      </div>
    </button>
  );
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
    league_id?: number;
    status: string;
    home_team: string;
    away_team: string;
    home_display_name?: string;
    away_display_name?: string;
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
                    {match.home_display_name ?? match.home_team}
                  </button>
                ) : (
                  <span>{match.home_display_name ?? match.home_team}</span>
                )}
                <span className="text-muted-foreground"> vs </span>
                {match.away_team_id && onTeamSelect ? (
                  <button
                    onClick={() => onTeamSelect(match.away_team_id!)}
                    className="text-primary hover:text-primary-hover transition-colors no-underline hover:no-underline"
                  >
                    {match.away_display_name ?? match.away_team}
                  </button>
                ) : (
                  <span>{match.away_display_name ?? match.away_team}</span>
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
function GroupStandingsTable({
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
                        className="text-primary hover:text-primary-hover transition-colors no-underline hover:no-underline text-left truncate"
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
 * Standings section with tabs per member league
 */
function GroupStandingsSection({
  memberLeagues,
  onTeamSelect,
}: {
  memberLeagues: { league_id: number; name: string }[];
  onTeamSelect?: (teamId: number) => void;
}) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const selectedLeagueId = memberLeagues[selectedIdx]?.league_id ?? null;

  const { data: standingsData, isLoading: isStandingsLoading } = useStandings(
    selectedLeagueId,
    { enabled: selectedLeagueId !== null },
  );

  if (memberLeagues.length === 0) {
    return (
      <div className="bg-card border border-border rounded-lg p-4">
        <div className="flex items-center gap-2 mb-2">
          <Trophy className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold text-foreground">Standings</h2>
        </div>
        <p className="text-sm text-muted-foreground">No member leagues available</p>
      </div>
    );
  }

  return (
    <div className="bg-transparent border border-border rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <Trophy className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-sm font-semibold text-foreground">Standings</h2>
        {standingsData && (
          <span className="text-xs text-muted-foreground ml-auto">
            {standingsData.season} &middot; {standingsData.source}
          </span>
        )}
        {standingsData?.isPlaceholder && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]">Placeholder</span>
        )}
        {standingsData?.isCalculated && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--status-info-bg)] text-[var(--status-info-text)]">Calculated</span>
        )}
      </div>

      {/* Tabs (only if multiple member leagues) */}
      {memberLeagues.length > 1 && (
        <div className="px-4 py-2 border-b border-border flex gap-1 overflow-x-auto">
          {memberLeagues.map((ml, idx) => (
            <button
              key={ml.league_id}
              onClick={() => setSelectedIdx(idx)}
              className={cn(
                "px-3 py-1 rounded text-xs font-medium transition-colors whitespace-nowrap",
                idx === selectedIdx
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              )}
            >
              {ml.name}
            </button>
          ))}
        </div>
      )}

      {/* Standings content */}
      {isStandingsLoading ? (
        <div className="flex items-center justify-center py-6">
          <Loader size="sm" />
        </div>
      ) : standingsData && standingsData.standings.length > 0 ? (
        <GroupStandingsTable standings={standingsData.standings} onTeamSelect={onTeamSelect} />
      ) : (
        <div className="px-4 py-4 text-sm text-muted-foreground">
          No standings available for this league
        </div>
      )}
    </div>
  );
}

/**
 * GroupDetail Component
 *
 * Displays detailed group information:
 * - Group info (name, key, country, paired_handling)
 * - Member leagues list (clickable)
 * - is_active_all status
 * - Aggregated stats by season
 * - TITAN coverage
 * - Recent matches across all member leagues
 * - Standings per member league
 */
export function GroupDetail({
  groupId,
  onBack,
  onLeagueSelect,
  onTeamSelect,
}: GroupDetailProps) {
  const { data, isLoading, error, refetch } = useFootballGroup(groupId);

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
              Group Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || `Unable to fetch group ${groupId}`}
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

  const { group, member_leagues, is_active_all, stats_by_season, titan, recent_matches } =
    data;

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
              <Users className="h-5 w-5 text-[var(--tag-purple-text)]" />
              <h1 className="text-lg font-semibold text-foreground">{group.name}</h1>
            </div>
            <div className="flex items-center gap-3 mt-1 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <CountryFlag country={group.country} size={14} />
                {group.country}
              </span>
              <span className="px-1.5 py-0.5 bg-muted rounded text-xs">{group.group_key}</span>
              {group.paired_handling && (
                <span className="text-xs">Paired: {group.paired_handling}</span>
              )}
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        {/* Status Badge */}
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
              is_active_all
                ? "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border border-[var(--status-success-border)]"
                : "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border border-[var(--status-warning-border)]"
            }`}
          >
            {is_active_all ? "All Leagues Active" : "Partially Active"}
          </span>
        </div>

        {/* Member Leagues */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Trophy className="h-4 w-4 text-primary" />
            Member Leagues ({member_leagues.length})
          </h2>
          <div className="space-y-2">
            {member_leagues.map((league) => (
              <MemberLeagueCard
                key={league.league_id}
                league={league}
                onClick={() => onLeagueSelect(league.league_id)}
              />
            ))}
          </div>
        </div>

        {/* TITAN Coverage */}
        {titan && (
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="h-4 w-4 text-[var(--tag-purple-text)]" />
              <h2 className="text-sm font-semibold text-foreground">TITAN Coverage (Aggregated)</h2>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-xl font-semibold text-foreground">
                  {titan.tier1_pct.toFixed(1)}%
                </p>
                <p className="text-xs text-muted-foreground">Tier 1</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{titan.tier1}</p>
                <p className="text-xs text-muted-foreground">Tier 1 count</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">{titan.total.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">Total matches</p>
              </div>
            </div>
          </div>
        )}

        {/* Stats by Season (Aggregated) */}
        <div className="bg-transparent border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Stats by Season (Aggregated)</h2>
          </div>
          <StatsTable stats={stats_by_season} />
        </div>

        {/* Recent Matches */}
        <div className="bg-transparent border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Recent Matches (All Leagues)</h2>
          </div>
          <RecentMatchesList matches={recent_matches} onTeamSelect={onTeamSelect} />
        </div>

        {/* Standings per member league */}
        <GroupStandingsSection
          memberLeagues={member_leagues}
          onTeamSelect={onTeamSelect}
        />
      </div>
    </ScrollArea>
  );
}
