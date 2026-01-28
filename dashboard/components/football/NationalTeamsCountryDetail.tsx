"use client";

import Image from "next/image";
import { useNationalsCountry } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  RefreshCw,
  AlertTriangle,
  Flag,
  Trophy,
  Calendar,
  Users,
  Globe,
  BarChart3,
} from "lucide-react";
import type { NationalsTeamItem, NationalsRecentMatch } from "@/lib/types";

/**
 * Team Logo Component
 */
function TeamLogo({ name, logoUrl, size = 24 }: { name: string; logoUrl: string | null; size?: number }) {
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
 * Country Flag for header
 */
function CountryHeader({ country }: { country: string }) {
  const isoCode = getCountryIsoCode(country);

  return (
    <div className="flex items-center gap-3">
      {isoCode ? (
        <Image
          src={`/flags/${isoCode}.svg`}
          alt={`${country} flag`}
          width={28}
          height={28}
          className="rounded-full object-cover"
        />
      ) : (
        <Flag className="h-7 w-7 text-muted-foreground" />
      )}
      <div>
        <h1 className="text-lg font-semibold text-foreground">{country}</h1>
        <p className="text-sm text-muted-foreground">National Teams</p>
      </div>
    </div>
  );
}

/**
 * Stats Summary Cards
 */
function StatsSummary({ stats }: { stats: { total_matches: number; wins: number; draws: number; losses: number } }) {
  const winPct = stats.total_matches > 0 ? ((stats.wins / stats.total_matches) * 100).toFixed(0) : "0";

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-xs text-muted-foreground">Matches</p>
        <p className="text-xl font-semibold text-foreground">{stats.total_matches}</p>
      </div>
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-xs text-muted-foreground">Wins</p>
        <p className="text-xl font-semibold text-[var(--status-success-text)]">{stats.wins}</p>
      </div>
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-xs text-muted-foreground">Draws</p>
        <p className="text-xl font-semibold text-muted-foreground">{stats.draws}</p>
      </div>
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-xs text-muted-foreground">Losses</p>
        <p className="text-xl font-semibold text-[var(--status-error-text)]">{stats.losses}</p>
      </div>
    </div>
  );
}

/**
 * Teams List Component
 */
function TeamsList({
  teams,
  onTeamSelect,
}: {
  teams: NationalsTeamItem[];
  onTeamSelect: (teamId: number) => void;
}) {
  if (teams.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No teams found
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {teams.map((team) => (
        <button
          key={team.team_id}
          onClick={() => onTeamSelect(team.team_id)}
          className="w-full text-left py-3 px-3 flex items-center gap-3 hover:bg-muted/50 transition-colors"
        >
          <TeamLogo name={team.name} logoUrl={team.logo_url} size={28} />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-foreground">{team.name}</p>
            <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
              <span>{team.total_matches} matches</span>
              {team.matches_25_26 > 0 && (
                <span className="text-[var(--status-success-text)]">{team.matches_25_26} in 25/26</span>
              )}
            </div>
            {team.competitions.length > 0 && (
              <p className="text-xs text-muted-foreground mt-0.5 truncate">
                {team.competitions.join(", ")}
              </p>
            )}
          </div>
        </button>
      ))}
    </div>
  );
}

/**
 * Recent Matches List
 */
function RecentMatchesList({ matches }: { matches: NationalsRecentMatch[] }) {
  if (matches.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No recent matches
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {matches.map((match) => {
        const matchDate = match.date ? new Date(match.date) : null;

        return (
          <div key={match.match_id} className="py-2 px-3 flex items-center gap-3">
            <div className="w-16 shrink-0">
              {matchDate ? (
                <>
                  <p className="text-xs text-muted-foreground">
                    {matchDate.toLocaleDateString([], { month: "short", day: "numeric" })}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {matchDate.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </p>
                </>
              ) : (
                <p className="text-xs text-muted-foreground">TBD</p>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-foreground truncate">
                {match.home_team} vs {match.away_team}
              </p>
              <p className="text-xs text-muted-foreground truncate">
                {match.competition_name}
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

interface NationalTeamsCountryDetailProps {
  country: string;
  onTeamSelect: (teamId: number) => void;
}

/**
 * NationalTeamsCountryDetail Component (Col 4)
 *
 * Displays national teams for a specific country:
 * - Stats summary (matches, wins, draws, losses)
 * - Teams list (clickable to open TeamDrawer)
 * - Competitions
 * - Recent matches
 */
export function NationalTeamsCountryDetail({
  country,
  onTeamSelect,
}: NationalTeamsCountryDetailProps) {
  const { data, isLoading, error, refetch } = useNationalsCountry(country);

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
              National Teams Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || `Unable to fetch national teams for ${country}`}
            </p>
          </div>
          <Button onClick={() => refetch()} variant="secondary">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <CountryHeader country={data.country} />
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Stats Summary */}
        {data.stats.total_matches > 0 && (
          <StatsSummary stats={data.stats} />
        )}

        {/* Teams */}
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold text-foreground">
              Teams ({data.teams.length})
            </h2>
          </div>
          <TeamsList teams={data.teams} onTeamSelect={onTeamSelect} />
        </div>

        {/* Competitions */}
        {data.competitions.length > 0 && (
          <div className="bg-card border border-border rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-border flex items-center gap-2">
              <Trophy className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold text-foreground">
                Competitions ({data.competitions.length})
              </h2>
            </div>
            <div className="divide-y divide-border">
              {data.competitions.map((comp) => (
                <div key={comp.league_id} className="py-2 px-4 flex items-center justify-between">
                  <span className="text-sm text-foreground">{comp.name}</span>
                  <span className="text-xs text-muted-foreground">{comp.matches_count} matches</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent Matches */}
        {data.recent_matches.length > 0 && (
          <div className="bg-card border border-border rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-border flex items-center gap-2">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold text-foreground">
                Recent Matches ({data.recent_matches.length})
              </h2>
            </div>
            <RecentMatchesList matches={data.recent_matches} />
          </div>
        )}

        {/* Empty state */}
        {data.teams.length === 0 && data.recent_matches.length === 0 && (
          <div className="text-center py-12 bg-muted/30 rounded-lg">
            <Flag className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-sm font-medium text-foreground mb-2">
              No Data Available
            </h3>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              National team data for {country} is not available yet.
            </p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
