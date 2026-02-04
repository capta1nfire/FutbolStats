"use client";

import Image from "next/image";
import { useFootballTournaments } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  RefreshCw,
  AlertTriangle,
  Trophy,
  Globe,
  Calendar,
  Users,
  BarChart3,
} from "lucide-react";
import type { TournamentEntry } from "@/lib/types";

/**
 * Country/International Flag Component
 */
function TournamentFlag({ country, size = 16 }: { country: string | null; size?: number }) {
  if (!country) {
    // International tournament - show globe
    return <Globe className="text-[var(--tag-blue-text)]" style={{ width: size, height: size }} />;
  }

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

/**
 * Kind Badge Component
 */
function KindBadge({ kind }: { kind: string }) {
  const colors: Record<string, string> = {
    international: "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)] border-[var(--tag-blue-border)]",
    cup: "bg-[var(--tag-orange-bg)] text-[var(--tag-orange-text)] border-[var(--tag-orange-border)]",
    friendly: "bg-[var(--tag-gray-bg)] text-[var(--tag-gray-text)] border-[var(--tag-gray-border)]",
  };

  return (
    <span className={`text-xs px-1.5 py-0.5 rounded border capitalize ${colors[kind] || colors.friendly}`}>
      {kind}
    </span>
  );
}

/**
 * Priority Badge Component
 */
function PriorityBadge({ priority }: { priority: string }) {
  const colors: Record<string, string> = {
    high: "bg-[var(--status-success-bg)] text-[var(--status-success-text)]",
    medium: "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]",
    low: "bg-[var(--tag-gray-bg)] text-[var(--tag-gray-text)]",
  };

  return (
    <span className={`text-xs px-1.5 py-0.5 rounded capitalize ${colors[priority] || colors.low}`}>
      {priority}
    </span>
  );
}

/**
 * Tournament Row Component
 */
function TournamentRow({
  tournament,
  onSelect,
}: {
  tournament: TournamentEntry;
  onSelect: (leagueId: number) => void;
}) {
  const { stats } = tournament;
  const lastMatch = stats.last_match ? new Date(stats.last_match) : null;
  const nextMatch = stats.next_match ? new Date(stats.next_match) : null;

  return (
    <button
      onClick={() => onSelect(tournament.league_id)}
      className="w-full text-left bg-card border border-border rounded-lg p-4 hover:border-border-hover hover:bg-muted/50 transition-colors"
    >
      <div className="flex items-start gap-3">
        {/* Flag */}
        <div className="shrink-0 mt-0.5">
          <TournamentFlag country={tournament.country} size={20} />
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-medium text-foreground truncate">
              {tournament.name}
            </h3>
            <KindBadge kind={tournament.kind} />
            <PriorityBadge priority={tournament.priority} />
          </div>

          {/* Stats row */}
          <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {stats.total_matches.toLocaleString()} matches
            </span>
            {stats.matches_30d > 0 && (
              <span className="text-[var(--status-success-text)]">{stats.matches_30d} in 30d</span>
            )}
            {stats.participants_count > 0 && (
              <span className="flex items-center gap-1">
                <Users className="h-3 w-3" />
                {stats.participants_count} teams
              </span>
            )}
            {stats.with_stats_pct !== null && stats.with_stats_pct > 0 && (
              <span className="flex items-center gap-1">
                <BarChart3 className="h-3 w-3" />
                {stats.with_stats_pct.toFixed(0)}% stats
              </span>
            )}
          </div>

          {/* Dates row */}
          {(lastMatch || nextMatch) && (
            <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
              {lastMatch && (
                <span>
                  Last: {lastMatch.toLocaleDateString([], { month: "short", day: "numeric" })}
                </span>
              )}
              {nextMatch && (
                <span className="text-[var(--status-success-text)]">
                  Next: {nextMatch.toLocaleDateString([], { month: "short", day: "numeric" })}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

interface TournamentsListProps {
  onLeagueSelect: (leagueId: number) => void;
}

/**
 * TournamentsList Component (Col 4)
 *
 * Displays all tournaments and cups organized by kind:
 * - International tournaments (Champions League, World Cup, etc.)
 * - National cups (FA Cup, Copa del Rey, etc.)
 * - Friendly matches
 */
export function TournamentsList({ onLeagueSelect }: TournamentsListProps) {
  const { data, isLoading, error, refetch } = useFootballTournaments();

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
              Tournaments Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || "Unable to fetch tournaments"}
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

  // Separate by kind
  const international = data.tournaments.filter((t) => t.kind === "international");
  const cups = data.tournaments.filter((t) => t.kind === "cup");
  const friendly = data.tournaments.filter((t) => t.kind === "friendly");

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Trophy className="h-6 w-6 text-primary" />
            <div>
              <h1 className="text-lg font-semibold text-foreground">Tournaments & Cups</h1>
              <p className="text-sm text-muted-foreground">
                {data.totals.tournaments_count} tournaments, {data.totals.cups_count} cups
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* International Tournaments */}
        {international.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Globe className="h-4 w-4 text-[var(--tag-blue-text)]" />
              International Tournaments ({international.length})
            </h2>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {international.map((tournament) => (
                <TournamentRow
                  key={tournament.league_id}
                  tournament={tournament}
                  onSelect={onLeagueSelect}
                />
              ))}
            </div>
          </div>
        )}

        {/* National Cups */}
        {cups.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Trophy className="h-4 w-4 text-primary" />
              National Cups ({cups.length})
            </h2>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {cups.map((tournament) => (
                <TournamentRow
                  key={tournament.league_id}
                  tournament={tournament}
                  onSelect={onLeagueSelect}
                />
              ))}
            </div>
          </div>
        )}

        {/* Friendly Matches */}
        {friendly.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Users className="h-4 w-4 text-muted-foreground" />
              Friendly Matches ({friendly.length})
            </h2>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {friendly.map((tournament) => (
                <TournamentRow
                  key={tournament.league_id}
                  tournament={tournament}
                  onSelect={onLeagueSelect}
                />
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {data.tournaments.length === 0 && (
          <div className="text-center py-12">
            <Trophy className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm text-muted-foreground">
              No tournaments available
            </p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
