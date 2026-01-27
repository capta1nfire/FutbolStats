"use client";

import { useFootballTeam } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import {
  RefreshCw,
  AlertTriangle,
  Users,
  Globe,
  MapPin,
  Trophy,
  TrendingUp,
  BarChart3,
} from "lucide-react";
import type {
  TeamInfo,
  TeamStats,
  TeamLeague,
  TeamFormMatch,
} from "@/lib/types";

interface TeamDrawerProps {
  teamId: number | null;
  open: boolean;
  onClose: () => void;
}

/**
 * Team Info Section
 */
function TeamInfoSection({ team }: { team: TeamInfo }) {
  return (
    <div className="space-y-4">
      {/* Team Header */}
      <div className="flex items-start gap-4">
        {team.logo_url ? (
          <img
            src={team.logo_url}
            alt={team.name}
            className="w-16 h-16 object-contain rounded-lg bg-muted p-2"
          />
        ) : (
          <div className="w-16 h-16 rounded-lg bg-muted flex items-center justify-center">
            <Users className="h-8 w-8 text-muted-foreground" />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <h3 className="text-base font-semibold text-foreground">{team.name}</h3>
          {team.short_name && (
            <p className="text-sm text-muted-foreground">{team.short_name}</p>
          )}
          <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
            <Globe className="h-3 w-3" />
            <span>{team.country}</span>
            {team.founded && (
              <>
                <span>·</span>
                <span>Est. {team.founded}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Venue Info */}
      {(team.venue_name || team.venue_city) && (
        <div className="bg-muted/50 rounded-lg p-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
            <MapPin className="h-3 w-3" />
            <span>Home Venue</span>
          </div>
          <p className="text-sm text-foreground">
            {team.venue_name || "Unknown venue"}
          </p>
          {team.venue_city && (
            <p className="text-xs text-muted-foreground">{team.venue_city}</p>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Stats Summary Section
 */
function StatsSummarySection({ stats }: { stats: TeamStats }) {
  const winRate = stats.total_matches > 0
    ? ((stats.wins / stats.total_matches) * 100).toFixed(1)
    : "0.0";
  const goalDiff = stats.goals_for - stats.goals_against;

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
        <BarChart3 className="h-4 w-4 text-muted-foreground" />
        Statistics
      </h4>
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-card border border-border rounded-lg p-2 text-center">
          <p className="text-lg font-semibold text-foreground">{stats.total_matches}</p>
          <p className="text-xs text-muted-foreground">Matches</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-2 text-center">
          <p className="text-lg font-semibold text-green-500">{stats.wins}</p>
          <p className="text-xs text-muted-foreground">Wins</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-2 text-center">
          <p className="text-lg font-semibold text-foreground">{winRate}%</p>
          <p className="text-xs text-muted-foreground">Win Rate</p>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2 text-center">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{stats.draws}</p>
          <p className="text-xs text-muted-foreground">Draws</p>
        </div>
        <div>
          <p className="text-sm font-medium text-red-400">{stats.losses}</p>
          <p className="text-xs text-muted-foreground">Losses</p>
        </div>
        <div>
          <p className="text-sm font-medium text-foreground">{stats.goals_for}</p>
          <p className="text-xs text-muted-foreground">GF</p>
        </div>
        <div>
          <p className={`text-sm font-medium ${goalDiff >= 0 ? "text-green-500" : "text-red-400"}`}>
            {goalDiff >= 0 ? `+${goalDiff}` : goalDiff}
          </p>
          <p className="text-xs text-muted-foreground">GD</p>
        </div>
      </div>
    </div>
  );
}

/**
 * Leagues Section
 */
function LeaguesSection({ leagues }: { leagues: TeamLeague[] }) {
  if (!leagues || leagues.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
        <Trophy className="h-4 w-4 text-amber-500" />
        Competitions ({leagues.length})
      </h4>
      <div className="space-y-2">
        {leagues.slice(0, 5).map((league) => (
          <div
            key={league.league_id}
            className="bg-muted/50 rounded-lg p-2 flex items-center justify-between gap-2"
          >
            <div className="flex-1 min-w-0">
              <p className="text-sm text-foreground truncate">{league.name}</p>
              <p className="text-xs text-muted-foreground">{league.country}</p>
            </div>
            <span className="text-xs text-muted-foreground shrink-0">
              {league.seasons.length} season{league.seasons.length !== 1 ? "s" : ""}
            </span>
          </div>
        ))}
        {leagues.length > 5 && (
          <p className="text-xs text-muted-foreground text-center">
            +{leagues.length - 5} more competitions
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * Recent Form Section
 */
function RecentFormSection({ form }: { form: TeamFormMatch[] }) {
  if (!form || form.length === 0) {
    return null;
  }

  const resultColors = {
    W: "bg-green-500",
    D: "bg-yellow-500",
    L: "bg-red-500",
  };

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
        <TrendingUp className="h-4 w-4 text-muted-foreground" />
        Recent Form
      </h4>
      {/* Form badges */}
      <div className="flex items-center gap-1">
        {form.slice(0, 5).map((match, idx) => (
          <span
            key={match.match_id}
            className={`w-6 h-6 rounded flex items-center justify-center text-xs font-medium text-white ${resultColors[match.result]}`}
            title={`${match.home ? "H" : "A"} vs ${match.opponent}: ${match.score}`}
          >
            {match.result}
          </span>
        ))}
      </div>
      {/* Recent matches list */}
      <div className="space-y-1">
        {form.slice(0, 5).map((match) => {
          const matchDate = new Date(match.date);
          const dateStr = matchDate.toLocaleDateString([], {
            month: "short",
            day: "numeric",
          });

          return (
            <div
              key={match.match_id}
              className="flex items-center gap-2 text-xs py-1"
            >
              <span className="w-12 text-muted-foreground">{dateStr}</span>
              <span
                className={`w-4 h-4 rounded text-center text-white text-[10px] leading-4 ${resultColors[match.result]}`}
              >
                {match.result}
              </span>
              <span className="flex-1 text-foreground truncate">
                {match.home ? "vs" : "@"} {match.opponent}
              </span>
              <span className="text-muted-foreground">{match.score}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * TeamDrawer Component
 *
 * Displays Team 360 information:
 * - Team info (name, country, founded, venue)
 * - Stats summary
 * - Competitions
 * - Recent form
 */
export function TeamDrawer({ teamId, open, onClose }: TeamDrawerProps) {
  const { data, isLoading, error, refetch } = useFootballTeam(teamId);

  // Content based on state
  let content: React.ReactNode;

  if (isLoading) {
    content = (
      <div className="flex items-center justify-center py-12">
        <Loader size="md" />
      </div>
    );
  } else if (error || !data) {
    content = (
      <div className="flex flex-col items-center gap-4 text-center py-12">
        <AlertTriangle className="h-10 w-10 text-yellow-400" />
        <div>
          <p className="text-sm font-medium text-foreground mb-1">
            Team Data Unavailable
          </p>
          <p className="text-xs text-muted-foreground">
            {error?.message || `Unable to fetch team ${teamId}`}
          </p>
        </div>
        <Button onClick={() => refetch()} variant="secondary" size="sm">
          <RefreshCw className="h-3 w-3 mr-1" />
          Retry
        </Button>
      </div>
    );
  } else {
    content = (
      <div className="space-y-6">
        {/* Team Info */}
        <TeamInfoSection team={data.team} />

        {/* Stats Summary */}
        {data.stats && <StatsSummarySection stats={data.stats} />}

        {/* Competitions */}
        {data.leagues && <LeaguesSection leagues={data.leagues} />}

        {/* Recent Form */}
        {data.recent_form && <RecentFormSection form={data.recent_form} />}

        {/* Footer with cache info */}
        <div className="pt-4 border-t border-border">
          <p className="text-xs text-muted-foreground text-center">
            Team ID: {teamId}
          </p>
        </div>
      </div>
    );
  }

  return (
    <DetailDrawer
      open={open}
      onClose={onClose}
      title={data?.team?.name || `Team ${teamId || ""}`}
      variant="overlay"
    >
      {content}
    </DetailDrawer>
  );
}
