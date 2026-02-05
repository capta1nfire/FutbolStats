"use client";

import { useState } from "react";
import { useFootballTeam } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { IconTabs } from "@/components/ui/icon-tabs";
import { toast } from "sonner";
import {
  RefreshCw,
  AlertTriangle,
  Users,
  MapPin,
  Trophy,
  TrendingUp,
  BarChart3,
  Image as ImageIcon,
  Info,
  Settings,
} from "lucide-react";
import type {
  TeamInfo,
  TeamStats,
  TeamLeague,
  TeamFormMatch,
} from "@/lib/types";
import { TeamLogoSettings } from "./TeamLogoSettings";
import { TeamEnrichmentSettings } from "./TeamEnrichmentSettings";
import { TeamWikiSettings } from "./TeamWikiSettings";

interface TeamDrawerProps {
  teamId: number | null;
  open: boolean;
  onClose: () => void;
  /** Persistent mode: always visible, no close functionality */
  persistent?: boolean;
}

/**
 * Team Info Section
 */
function TeamInfoSection({ team }: { team: TeamInfo }) {
  return (
    <div className="space-y-3">
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
          <p className="text-lg font-semibold text-[var(--status-success-text)]">{stats.wins}</p>
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
          <p className="text-sm font-medium text-[var(--status-error-text)]">{stats.losses}</p>
          <p className="text-xs text-muted-foreground">Losses</p>
        </div>
        <div>
          <p className="text-sm font-medium text-foreground">{stats.goals_for}</p>
          <p className="text-xs text-muted-foreground">GF</p>
        </div>
        <div>
          <p className={`text-sm font-medium ${goalDiff >= 0 ? "text-[var(--status-success-text)]" : "text-[var(--status-error-text)]"}`}>
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
        <Trophy className="h-4 w-4 text-primary" />
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
              <p className="text-xs text-muted-foreground">
                {league.matches} matches · {league.seasons_range[0]}-{league.seasons_range[1]}
              </p>
            </div>
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
    W: "bg-[var(--status-success-text)]",
    D: "bg-muted-foreground",
    L: "bg-[var(--status-error-text)]",
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

/** Tab definitions for team drawer */
const TEAM_TABS = [
  { id: "overview", icon: <Info />, label: "Overview" },
  { id: "multimedia", icon: <ImageIcon />, label: "Multimedia" },
  { id: "settings", icon: <Settings />, label: "Settings" },
];

/**
 * TeamDrawer Component
 *
 * Displays Team 360 information:
 * - Team info (name, country, founded, venue)
 * - Stats summary
 * - Competitions
 * - Recent form
 */
export function TeamDrawer({ teamId, open, onClose, persistent = false }: TeamDrawerProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const { data, isLoading, error, refetch } = useFootballTeam(teamId);

  // Content based on state
  let content: React.ReactNode;

  // No team selected - show placeholder (only relevant in persistent mode)
  if (!teamId) {
    content = (
      <div className="flex flex-col items-center justify-center h-full py-12 px-4">
        <Users className="h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-sm font-medium text-foreground mb-1">Team 360</h3>
        <p className="text-xs text-muted-foreground text-center">
          Select a team from the list to view detailed information, statistics, and settings.
        </p>
      </div>
    );
  } else if (isLoading) {
    content = (
      <div className="flex items-center justify-center py-12">
        <Loader size="md" />
      </div>
    );
  } else if (error || !data) {
    content = (
      <div className="flex flex-col items-center gap-4 text-center py-12">
        <AlertTriangle className="h-10 w-10 text-[var(--status-warning-text)]" />
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

        {/* Tabs */}
        <IconTabs
          tabs={TEAM_TABS}
          value={activeTab}
          onValueChange={setActiveTab}
          className="w-full"
        />

        {/* Tab Content */}
        {activeTab === "overview" && (
          <>
            {/* Stats Summary */}
            {data.stats && <StatsSummarySection stats={data.stats} />}

            {/* Competitions */}
            {data.leagues_played && <LeaguesSection leagues={data.leagues_played} />}

            {/* Recent Form */}
            {data.recent_form && <RecentFormSection form={data.recent_form} />}
          </>
        )}

        {activeTab === "multimedia" && (
          <div className="space-y-4 pb-4">
            <TeamLogoSettings
              teamId={data.team.team_id}
              teamName={data.team.name}
              fallbackLogoUrl={data.team.logo_url || undefined}
            />
          </div>
        )}

        {activeTab === "settings" && (
          <div className="space-y-4 pb-4">
            <TeamEnrichmentSettings
              teamId={data.team.team_id}
              teamName={data.team.name}
              enrichment={data.wikidata_enrichment}
            />
            <TeamWikiSettings
              teamId={data.team.team_id}
              teamName={data.team.name}
              wiki={data.team.wiki}
            />
          </div>
        )}
      </div>
    );
  }

  const handleCopyId = () => {
    if (teamId) {
      navigator.clipboard.writeText(String(teamId));
      toast.success(`Team ID ${teamId} copied`);
    }
  };

  // Title: display name (clickable to copy ID) or fallback
  const displayName = data?.wikidata_enrichment?.short_name ?? data?.team?.name;
  const drawerTitle = teamId ? (
    displayName ? (
      <button
        onClick={handleCopyId}
        className="text-foreground hover:text-primary transition-colors cursor-pointer"
        title={`Click to copy ID: ${teamId}`}
      >
        {displayName}
      </button>
    ) : isLoading ? (
      <span className="text-muted-foreground">Loading...</span>
    ) : (
      <span>Team {teamId}</span>
    )
  ) : (
    "Team 360"
  );

  return (
    <DetailDrawer
      open={open}
      onClose={onClose}
      title={drawerTitle}
      variant={persistent ? "inline" : "overlay"}
      persistent={persistent}
    >
      {content}
    </DetailDrawer>
  );
}
