"use client";

import Image from "next/image";
import { useWorldCupGroup } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  RefreshCw,
  AlertTriangle,
  Trophy,
  ArrowLeft,
  Calendar,
  Clock,
  Globe,
  XCircle,
} from "lucide-react";
import type { WorldCupStandingEntry, WorldCupGroupMatch } from "@/lib/types";

/**
 * Team Flag Component
 * Uses logo_url from backend if available, falls back to country flag
 */
function TeamFlag({ name, logoUrl, size = 20 }: { name: string; logoUrl: string | null; size?: number }) {
  // Use logo_url if provided by backend
  if (logoUrl) {
    return (
      <img
        src={logoUrl}
        alt={name}
        width={size}
        height={size}
        className="rounded-full object-cover"
      />
    );
  }

  // Fall back to country flag
  const isoCode = getCountryIsoCode(name);
  if (isoCode) {
    return (
      <Image
        src={`/flags/${isoCode}.svg`}
        alt={`${name} flag`}
        width={size}
        height={size}
        className="rounded-full object-cover"
      />
    );
  }

  return <Globe className="text-muted-foreground" style={{ width: size, height: size }} />;
}

/**
 * Standings Table Component
 */
function StandingsTable({
  standings,
  onTeamSelect,
}: {
  standings: WorldCupStandingEntry[];
  onTeamSelect?: (teamId: number) => void;
}) {
  if (!standings || standings.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No standings available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 px-2 text-xs font-medium text-muted-foreground w-8">#</th>
            <th className="text-left py-2 px-2 text-xs font-medium text-muted-foreground">Team</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">P</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">W</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">D</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">L</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-12">GD</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">Pts</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((entry) => {
            const canClick = entry.team_id !== null && onTeamSelect;
            return (
              <tr key={entry.team_id || entry.name} className="border-b border-border last:border-0">
                <td className="py-2 px-2 text-muted-foreground">{entry.position}</td>
                <td className="py-2 px-2">
                  {canClick ? (
                    <button
                      onClick={() => onTeamSelect!(entry.team_id!)}
                      className="flex items-center gap-2 hover:text-primary transition-colors"
                    >
                      <TeamFlag name={entry.name} logoUrl={entry.logo_url} size={16} />
                      <span className="font-medium text-foreground hover:underline">
                        {entry.name}
                      </span>
                    </button>
                  ) : (
                    <div className="flex items-center gap-2">
                      <TeamFlag name={entry.name} logoUrl={entry.logo_url} size={16} />
                      <span className="font-medium text-foreground">{entry.name}</span>
                    </div>
                  )}
                </td>
                <td className="py-2 px-2 text-center text-muted-foreground">{entry.played}</td>
                <td className="py-2 px-2 text-center text-muted-foreground">{entry.won}</td>
                <td className="py-2 px-2 text-center text-muted-foreground">{entry.drawn}</td>
                <td className="py-2 px-2 text-center text-muted-foreground">{entry.lost}</td>
                <td className={`py-2 px-2 text-center ${entry.goal_diff > 0 ? "text-green-500" : entry.goal_diff < 0 ? "text-red-400" : "text-muted-foreground"}`}>
                  {entry.goal_diff > 0 ? `+${entry.goal_diff}` : entry.goal_diff}
                </td>
                <td className="py-2 px-2 text-center font-semibold text-foreground">{entry.points}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Matches List Component
 */
function MatchesList({
  matches,
  onTeamSelect,
}: {
  matches: WorldCupGroupMatch[];
  onTeamSelect?: (teamId: number) => void;
}) {
  if (!matches || matches.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No matches available
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {matches.map((match) => {
        const matchDate = match.date ? new Date(match.date) : null;
        const dateStr = matchDate
          ? matchDate.toLocaleDateString([], { month: "short", day: "numeric" })
          : "TBD";
        const timeStr = matchDate
          ? matchDate.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
          : "--:--";

        return (
          <div key={match.match_id} className="py-2 px-3 flex items-center gap-3">
            <div className="w-16 shrink-0">
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

interface WorldCup2026GroupDetailProps {
  group: string;
  onBack: () => void;
  onTeamSelect: (teamId: number) => void;
}

/**
 * WorldCup2026GroupDetail Component (Col 4)
 *
 * Displays a specific World Cup 2026 group:
 * - Group standings table (clickable teams if team_id available)
 * - Group matches list
 */
export function WorldCup2026GroupDetail({
  group,
  onBack,
  onTeamSelect,
}: WorldCup2026GroupDetailProps) {
  const { data, isLoading, error, refetch } = useWorldCupGroup(group);

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
              Group Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || `Unable to fetch Group ${group} data`}
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
              <h1 className="text-lg font-semibold text-foreground">
                {data.group}
              </h1>
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              FIFA World Cup 2026
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        {/* Status Badge */}
        {data.status === "not_ready" && (
          <div className="flex items-center gap-2 px-3 py-2 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <Clock className="h-4 w-4 text-yellow-500" />
            <span className="text-sm text-yellow-500">
              Group data not yet available
            </span>
          </div>
        )}
        {data.status === "disabled" && (
          <div className="flex items-center gap-2 px-3 py-2 bg-gray-500/10 border border-gray-500/20 rounded-lg">
            <XCircle className="h-4 w-4 text-gray-400" />
            <span className="text-sm text-gray-400">
              Group feature disabled
            </span>
          </div>
        )}

        {/* Standings Table */}
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <Trophy className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Standings</h2>
          </div>
          <StandingsTable standings={data.standings} onTeamSelect={onTeamSelect} />
        </div>

        {/* Matches List */}
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">Matches</h2>
          </div>
          <MatchesList matches={data.matches} onTeamSelect={onTeamSelect} />
        </div>

        {/* Empty state for not_ready with no data */}
        {data.status === "not_ready" && data.standings.length === 0 && data.matches.length === 0 && (
          <div className="text-center py-8 bg-muted/30 rounded-lg">
            <Clock className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">
              Group composition will be available after the draw
            </p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
