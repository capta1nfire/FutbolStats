"use client";

import Image from "next/image";
import { useFootballCountry } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  RefreshCw,
  AlertTriangle,
  Trophy,
  Users,
  Calendar,
  BarChart3,
  ChevronRight,
  Globe,
} from "lucide-react";
import type { CompetitionEntry } from "@/lib/types";

/**
 * Country Flag Component
 */
function CountryFlag({ country, size = 20 }: { country: string; size?: number }) {
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

interface CountryCompetitionsProps {
  country: string;
  onLeagueSelect: (leagueId: number) => void;
  onGroupSelect: (groupId: number) => void;
}

/**
 * Competition Card
 */
function CompetitionCard({
  competition,
  onLeagueSelect,
  onGroupSelect,
}: {
  competition: CompetitionEntry;
  onLeagueSelect: (leagueId: number) => void;
  onGroupSelect: (groupId: number) => void;
}) {
  const isGroup = competition.type === "group";
  const hasStats = competition.stats.total_matches > 0;

  const handleClick = () => {
    if (isGroup && competition.group_id) {
      onGroupSelect(competition.group_id);
    } else if (competition.league_id) {
      onLeagueSelect(competition.league_id);
    }
  };

  return (
    <button
      onClick={handleClick}
      className="w-full text-left bg-card border border-border rounded-lg p-4 hover:border-border-hover hover:bg-muted/50 transition-colors group"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-2 mb-1">
            {isGroup ? (
              <Users className="h-4 w-4 text-[var(--tag-purple-text)] shrink-0" />
            ) : (
              <Trophy className="h-4 w-4 text-amber-500 shrink-0" />
            )}
            <h3 className="text-sm font-medium text-foreground truncate">
              {competition.name}
            </h3>
          </div>

          {/* Meta info */}
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {competition.kind && (
              <span className="capitalize">{competition.kind}</span>
            )}
            {competition.priority && (
              <span className="px-1.5 py-0.5 bg-muted rounded text-xs">
                {competition.priority}
              </span>
            )}
            {isGroup && competition.member_count && (
              <span>{competition.member_count} leagues</span>
            )}
          </div>

          {/* Stats */}
          {hasStats && (
            <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                {competition.stats.total_matches.toLocaleString()} matches
              </span>
              {competition.stats.seasons_range && (
                <span>
                  {competition.stats.seasons_range[0]}-{competition.stats.seasons_range[1]}
                </span>
              )}
              {competition.stats.with_stats_pct !== null && (
                <span className="flex items-center gap-1">
                  <BarChart3 className="h-3 w-3" />
                  {competition.stats.with_stats_pct.toFixed(0)}% stats
                </span>
              )}
            </div>
          )}

          {/* TITAN badge */}
          {competition.titan && competition.titan.tier1_pct !== null && (
            <div className="mt-2 inline-flex items-center gap-1 px-2 py-0.5 bg-[var(--tag-purple-bg)] border border-[var(--tag-purple-border)] rounded text-xs text-[var(--tag-purple-text)]">
              TITAN: {competition.titan.tier1_pct.toFixed(0)}% Tier 1
              <span className="text-muted-foreground">
                ({competition.titan.tier1}/{competition.titan.total})
              </span>
            </div>
          )}

          {/* Group members preview */}
          {isGroup && competition.members && competition.members.length > 0 && (
            <div className="mt-2 text-xs text-muted-foreground">
              <span className="font-medium">Members: </span>
              {competition.members
                .slice(0, 3)
                .map((m) => m.name)
                .join(", ")}
              {competition.members.length > 3 && ` +${competition.members.length - 3} more`}
            </div>
          )}
        </div>

        {/* Arrow */}
        <ChevronRight className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors shrink-0" />
      </div>
    </button>
  );
}

/**
 * CountryCompetitions Component
 *
 * Displays all competitions for a country:
 * - Groups (with member leagues)
 * - Standalone leagues
 * - Stats and TITAN coverage for each
 */
export function CountryCompetitions({
  country,
  onLeagueSelect,
  onGroupSelect,
}: CountryCompetitionsProps) {
  const { data, isLoading, error, refetch } = useFootballCountry(country);

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
              Country Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || `Unable to fetch data for ${country}`}
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

  // Separate groups and standalone leagues
  const groups = data.competitions.filter((c) => c.type === "group");
  const leagues = data.competitions.filter((c) => c.type === "league");

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CountryFlag country={data.country} size={32} />
            <div>
              <h1 className="text-lg font-semibold text-foreground">{data.country}</h1>
              <p className="text-sm text-muted-foreground">
                {data.total} competition{data.total !== 1 ? "s" : ""}
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Groups Section */}
        {groups.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Users className="h-4 w-4 text-[var(--tag-purple-text)]" />
              Competition Groups ({groups.length})
            </h2>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {groups.map((competition) => (
                <CompetitionCard
                  key={`group-${competition.group_id}`}
                  competition={competition}
                  onLeagueSelect={onLeagueSelect}
                  onGroupSelect={onGroupSelect}
                />
              ))}
            </div>
          </div>
        )}

        {/* Standalone Leagues Section */}
        {leagues.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Trophy className="h-4 w-4 text-amber-500" />
              Leagues ({leagues.length})
            </h2>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {leagues.map((competition) => (
                <CompetitionCard
                  key={`league-${competition.league_id}`}
                  competition={competition}
                  onLeagueSelect={onLeagueSelect}
                  onGroupSelect={onGroupSelect}
                />
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {data.competitions.length === 0 && (
          <div className="text-center py-12">
            <Trophy className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm text-muted-foreground">
              No competitions found for {country}
            </p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
