"use client";

import { useFootballLeague } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
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
  onTeamClick,
}: {
  matches: {
    match_id: number;
    date: string | null;
    status: string;
    home_team: string;
    away_team: string;
    score: string | null;
  }[];
  onTeamClick?: (teamName: string) => void;
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
                {match.home_team} vs {match.away_team}
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
 * LeagueDetail Component
 *
 * Displays detailed league information:
 * - League info (name, country, kind, priority)
 * - Group info if part of a group
 * - Stats by season table
 * - TITAN coverage
 * - Recent matches
 * - Standings placeholder
 */
export function LeagueDetail({ leagueId, onBack, onTeamSelect }: LeagueDetailProps) {
  const { data, isLoading, error, refetch } = useFootballLeague(leagueId);

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

  const { league, group, stats_by_season, titan, recent_matches, standings } = data;

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
                <Globe className="h-3 w-3" />
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
          <RecentMatchesList matches={recent_matches} />
        </div>

        {/* Standings Placeholder */}
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Trophy className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Standings</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            {standings.status === "not_available"
              ? standings.note || "Standings not available"
              : standings.note || "Coming soon"}
          </p>
        </div>
      </div>
    </ScrollArea>
  );
}
