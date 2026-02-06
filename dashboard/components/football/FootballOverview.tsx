"use client";

import { useFootballOverview } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";
import {
  RefreshCw,
  AlertTriangle,
  Trophy,
  Globe,
  Calendar,
  Radio,
  CheckCircle,
  Users,
  TrendingUp,
  Clock,
} from "lucide-react";

/**
 * Summary Card Component
 */
function SummaryCard({
  icon: Icon,
  label,
  value,
  variant = "default",
}: {
  icon: React.ElementType;
  label: string;
  value: number;
  variant?: "default" | "live" | "success";
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "p-2 rounded-md",
            variant === "live" && "bg-[var(--status-error-bg)]",
            variant === "success" && "bg-[var(--status-success-bg)]",
            variant === "default" && "bg-muted"
          )}
        >
          <Icon
            className={cn(
              "h-5 w-5",
              variant === "live" && "text-[var(--status-error-text)]",
              variant === "success" && "text-[var(--status-success-text)]",
              variant === "default" && "text-muted-foreground"
            )}
            strokeWidth={1.5}
          />
        </div>
        <div>
          <p className="text-2xl font-semibold text-foreground">{value.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">{label}</p>
        </div>
      </div>
    </div>
  );
}

/**
 * Upcoming Match Row
 */
function UpcomingMatchRow({
  match,
}: {
  match: {
    match_id: number;
    date: string | null;
    league_name: string;
    home_team: string;
    away_team: string;
    home_display_name?: string;
    away_display_name?: string;
    status: string;
    has_prediction: boolean;
  };
}) {
  // Handle null date
  const dateStr = match.date
    ? new Date(match.date).toLocaleDateString([], { month: "short", day: "numeric" })
    : "TBD";
  const timeStr = match.date
    ? new Date(match.date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "--:--";

  return (
    <div className="flex items-center gap-3 py-2 px-3 hover:bg-muted/50 rounded-md transition-colors">
      <div className="w-16 text-center shrink-0">
        <p className="text-xs text-muted-foreground">{dateStr}</p>
        <p className="text-sm font-medium text-foreground">{timeStr}</p>
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-foreground truncate">
          {match.home_display_name ?? match.home_team} vs {match.away_display_name ?? match.away_team}
        </p>
        <p className="text-xs text-muted-foreground truncate">{match.league_name}</p>
      </div>
      {match.has_prediction && (
        <span title="Has prediction">
          <TrendingUp className="h-4 w-4 text-[var(--status-success-text)] shrink-0" />
        </span>
      )}
    </div>
  );
}

/**
 * Top League Row
 */
function TopLeagueRow({
  league,
}: {
  league: {
    league_id: number;
    name: string;
    country: string;
    matches_30d: number;
    matches_total: number;
    with_stats_pct: number | null;
    with_odds_pct: number | null;
  };
}) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 hover:bg-muted/50 rounded-md transition-colors">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-foreground truncate">{league.name}</p>
        <p className="text-xs text-muted-foreground truncate">{league.country}</p>
      </div>
      <div className="text-right shrink-0">
        <p className="text-sm font-medium text-foreground">{league.matches_30d}</p>
        <p className="text-xs text-muted-foreground">last 30d</p>
      </div>
      {league.with_stats_pct !== null && (
        <div className="w-12 text-right shrink-0">
          <p className="text-xs text-muted-foreground">{league.with_stats_pct.toFixed(0)}% stats</p>
        </div>
      )}
    </div>
  );
}

/**
 * Alert Card
 */
function AlertCard({
  alert,
}: {
  alert: {
    type: string;
    league_name: string;
    message: string;
    value: number;
  };
}) {
  return (
    <div className="flex items-start gap-3 p-3 bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)] rounded-lg">
      <AlertTriangle className="h-4 w-4 text-[var(--status-warning-text)] shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-foreground">{alert.league_name}</p>
        <p className="text-xs text-muted-foreground">{alert.message}</p>
      </div>
    </div>
  );
}

/**
 * TITAN Coverage Badge
 */
function TitanBadge({
  titan,
}: {
  titan: {
    total: number;
    tier1: number;
    tier1b: number;
    tier1_pct: number;
    tier1b_pct: number;
  };
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 rounded-md bg-[var(--tag-purple-bg)]">
          <TrendingUp className="h-4 w-4 text-[var(--tag-purple-text)]" />
        </div>
        <h3 className="text-sm font-semibold text-foreground">TITAN Coverage</h3>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-2xl font-semibold text-foreground">{titan.tier1_pct.toFixed(1)}%</p>
          <p className="text-xs text-muted-foreground">Tier 1 ({titan.tier1.toLocaleString()})</p>
        </div>
        <div>
          <p className="text-2xl font-semibold text-foreground">{titan.tier1b_pct.toFixed(1)}%</p>
          <p className="text-xs text-muted-foreground">Tier 1b ({titan.tier1b.toLocaleString()})</p>
        </div>
      </div>
      <p className="text-xs text-muted-foreground mt-2">
        Total matches: {titan.total.toLocaleString()}
      </p>
    </div>
  );
}

/**
 * FootballOverview Component
 *
 * Displays overview data:
 * - Summary cards (leagues, countries, matches live, etc.)
 * - Upcoming matches list
 * - Top leagues table
 * - Alerts
 * - TITAN coverage badge
 */
export function FootballOverview() {
  const { data, isLoading, error, refetch } = useFootballOverview();

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
              Football Overview Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || "Unable to fetch football overview data"}
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

  const { summary, upcoming, leagues, alerts, titan } = data;

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-foreground">Football Overview</h1>
            <p className="text-sm text-muted-foreground">
              Summary of football data and upcoming matches
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          <SummaryCard
            icon={Trophy}
            label="Active Leagues"
            value={summary.leagues_active_count}
          />
          <SummaryCard
            icon={Globe}
            label="Countries"
            value={summary.countries_active_count}
          />
          <SummaryCard
            icon={Calendar}
            label="Next 7 Days"
            value={summary.matches_next_7d_count}
          />
          <SummaryCard
            icon={Radio}
            label="Live Now"
            value={summary.matches_live_count}
            variant="live"
          />
          <SummaryCard
            icon={CheckCircle}
            label="Finished 24h"
            value={summary.matches_finished_24h_count}
            variant="success"
          />
          <SummaryCard
            icon={Users}
            label="Active Teams"
            value={summary.teams_active_count}
          />
        </div>

        {/* Alerts */}
        {alerts && alerts.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-[var(--status-warning-text)]" />
              Alerts ({alerts.length})
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {alerts.map((alert, idx) => (
                <AlertCard key={`${alert.league_id}-${idx}`} alert={alert} />
              ))}
            </div>
          </div>
        )}

        {/* Two Column Layout: Upcoming + Top Leagues */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upcoming Matches */}
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              Upcoming Matches
            </h2>
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              {upcoming && upcoming.length > 0 ? (
                <div className="divide-y divide-border">
                  {upcoming.slice(0, 10).map((match) => (
                    <UpcomingMatchRow key={match.match_id} match={match} />
                  ))}
                </div>
              ) : (
                <div className="p-4 text-center text-sm text-muted-foreground">
                  No upcoming matches
                </div>
              )}
            </div>
          </div>

          {/* Top Leagues */}
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Trophy className="h-4 w-4 text-muted-foreground" />
              Top Leagues (by activity)
            </h2>
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              {leagues && leagues.length > 0 ? (
                <div className="divide-y divide-border">
                  {leagues.slice(0, 10).map((league) => (
                    <TopLeagueRow key={league.league_id} league={league} />
                  ))}
                </div>
              ) : (
                <div className="p-4 text-center text-sm text-muted-foreground">
                  No league data
                </div>
              )}
            </div>
          </div>
        </div>

        {/* TITAN Coverage */}
        {titan && <TitanBadge titan={titan} />}
      </div>
    </ScrollArea>
  );
}
