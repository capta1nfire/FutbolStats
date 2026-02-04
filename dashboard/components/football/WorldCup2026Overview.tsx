"use client";

import { useWorldCupOverview } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
import {
  RefreshCw,
  AlertTriangle,
  Trophy,
  Calendar,
  Users,
  Clock,
  AlertCircle,
  XCircle,
  CheckCircle,
} from "lucide-react";

interface WorldCup2026OverviewProps {
  onGroupsClick: () => void;
}

/**
 * Status Badge Component
 */
function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    not_ready: "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]",
    ok: "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]",
    disabled: "bg-[var(--tag-gray-bg)] text-[var(--tag-gray-text)] border-[var(--tag-gray-border)]",
  };

  const labels: Record<string, string> = {
    not_ready: "Not Ready",
    ok: "Active",
    disabled: "Disabled",
  };

  return (
    <span className={`text-xs px-2 py-1 rounded border ${colors[status] || colors.not_ready}`}>
      {labels[status] || status}
    </span>
  );
}

/**
 * Alert Card Component
 */
function AlertCard({ alert }: { alert: { type: string; message: string; value: number | null } }) {
  return (
    <div className="flex items-start gap-3 bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)] rounded-lg p-3">
      <AlertCircle className="h-4 w-4 text-[var(--status-warning-text)] shrink-0 mt-0.5" />
      <div>
        <p className="text-sm text-foreground">{alert.message}</p>
        <p className="text-xs text-muted-foreground capitalize mt-0.5">
          {alert.type.replace(/_/g, " ")}
        </p>
      </div>
    </div>
  );
}

/**
 * WorldCup2026Overview Component (Col 4)
 *
 * Displays World Cup 2026 overview:
 * - Status badge (not_ready, active, completed)
 * - Summary cards (groups, teams, matches)
 * - Alerts (missing data, etc.)
 * - Upcoming matches
 */
export function WorldCup2026Overview({ onGroupsClick }: WorldCup2026OverviewProps) {
  const { data, isLoading, error, refetch } = useWorldCupOverview();

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
              World Cup 2026 Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || "Unable to fetch World Cup 2026 data"}
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

  const { summary, alerts, upcoming } = data;

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Trophy className="h-6 w-6 text-primary" />
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-foreground">
                  FIFA World Cup 2026
                </h1>
                <StatusBadge status={data.status} />
              </div>
              <p className="text-sm text-muted-foreground">
                USA, Mexico & Canada
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Users className="h-4 w-4" />
              <span className="text-xs">Groups</span>
            </div>
            <p className="text-2xl font-semibold text-foreground">
              {summary.groups_count}
            </p>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Trophy className="h-4 w-4" />
              <span className="text-xs">Teams</span>
            </div>
            <p className="text-2xl font-semibold text-foreground">
              {summary.teams_count}
            </p>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Calendar className="h-4 w-4" />
              <span className="text-xs">Matches</span>
            </div>
            <p className="text-2xl font-semibold text-foreground">
              {summary.matches_played}/{summary.matches_total}
            </p>
          </div>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Clock className="h-4 w-4" />
              <span className="text-xs">Upcoming</span>
            </div>
            <p className="text-2xl font-semibold text-foreground">
              {summary.matches_upcoming}
            </p>
          </div>
        </div>

        {/* View Groups Button */}
        {summary.groups_count > 0 && (
          <Button onClick={onGroupsClick} className="w-full">
            View All Groups
          </Button>
        )}

        {/* Alerts Section */}
        {alerts.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-[var(--status-warning-text)]" />
              Alerts ({alerts.length})
            </h2>
            <div className="space-y-2">
              {alerts.map((alert, idx) => (
                <AlertCard key={idx} alert={alert} />
              ))}
            </div>
          </div>
        )}

        {/* Upcoming Matches */}
        {upcoming.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              Upcoming Matches ({upcoming.length})
            </h2>
            <div className="space-y-2">
              {upcoming.map((match) => {
                const matchDate = match.date ? new Date(match.date) : null;
                return (
                  <div
                    key={match.match_id}
                    className="bg-card border border-border rounded-lg p-3 flex items-center gap-3"
                  >
                    <div className="w-16 shrink-0">
                      {matchDate && (
                        <>
                          <p className="text-xs text-muted-foreground">
                            {matchDate.toLocaleDateString([], { month: "short", day: "numeric" })}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {matchDate.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </p>
                        </>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-foreground truncate">
                        {match.home_team} vs {match.away_team}
                      </p>
                      {match.group && (
                        <p className="text-xs text-muted-foreground">
                          {match.group}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Not Ready Message */}
        {data.status === "not_ready" && alerts.length === 0 && upcoming.length === 0 && (
          <div className="text-center py-12 bg-muted/30 rounded-lg">
            <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-sm font-medium text-foreground mb-2">
              Tournament Not Yet Started
            </h3>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              World Cup 2026 data will be available once fixtures and standings are announced.
              Check back later for updates.
            </p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
