"use client";

import { useState, useEffect, useRef } from "react";
import { useFootballTeam, useTeamSquad, useTeamEnrichmentDeleteMutation } from "@/lib/hooks";
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
  Loader2,
  Settings,
  Trash2,
} from "lucide-react";
import type {
  TeamInfo,
  TeamStats,
  TeamLeague,
  TeamFormMatch,
} from "@/lib/types";
import { SurfaceCard } from "@/components/ui/surface-card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TeamLogoSettings } from "./TeamLogoSettings";
import { TeamEnrichmentSettings } from "./TeamEnrichmentSettings";
import type { TeamEnrichmentHandle, EnrichmentFormState } from "./TeamEnrichmentSettings";
import { TeamWikiSettings } from "./TeamWikiSettings";
import { ManagerCard, InjuryList } from "@/components/squad";

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
  const [enrichmentNotes, setEnrichmentNotes] = useState("");
  const [enrichmentForm, setEnrichmentForm] = useState<EnrichmentFormState>({ isDirty: false, canSave: false, isPending: false });
  const enrichmentRef = useRef<TeamEnrichmentHandle>(null);
  const { data, isLoading, error, refetch } = useFootballTeam(teamId);
  const deleteMutation = useTeamEnrichmentDeleteMutation();

  // Sync notes state when enrichment data changes
  useEffect(() => {
    setEnrichmentNotes(data?.wikidata_enrichment?.override?.notes ?? "");
  }, [data?.wikidata_enrichment?.override?.notes]);

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

            {/* Squad: Manager + Injuries */}
            {data.team?.team_id && (
              <TeamSquadOverview teamId={data.team.team_id} />
            )}
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
          <div className="pb-4">
            <SurfaceCard className="space-y-4">
              <h4 className="text-sm font-medium flex items-center gap-1.5">
                Team Enrichment
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="h-4 w-4 text-primary" />
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={6}>
                    Override values (leave empty to use Wikidata)
                  </TooltipContent>
                </Tooltip>
              </h4>
              <TeamEnrichmentSettings
                ref={enrichmentRef}
                teamId={data.team.team_id}
                teamName={data.team.name}
                enrichment={data.wikidata_enrichment}
                notes={enrichmentNotes}
                onNotesChange={setEnrichmentNotes}
                onFormStateChange={setEnrichmentForm}
              />
              <TeamWikiSettings
                teamId={data.team.team_id}
                teamName={data.team.name}
                wiki={data.team.wiki}
              />
              {/* Notes - last field in the card */}
              <div className="space-y-1">
                <Label htmlFor={`team-${data.team.team_id}-notes`} className="text-xs">
                  Notes <span className="opacity-50">(optional)</span>
                </Label>
                <Input
                  id={`team-${data.team.team_id}-notes`}
                  type="text"
                  placeholder="Reason for override..."
                  value={enrichmentNotes}
                  onChange={(e) => setEnrichmentNotes(e.target.value)}
                  className="h-8 text-sm"
                />
              </div>

              {/* Save/Cancel buttons for enrichment */}
              {enrichmentForm.isDirty && (
                <div className="flex items-center justify-end gap-3 pt-2">
                  <button
                    type="button"
                    onClick={() => enrichmentRef.current?.handleReset()}
                    disabled={enrichmentForm.isPending}
                    className="text-sm px-3 py-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-[color:var(--field-bg-hover)] transition-colors disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <Button
                    size="sm"
                    onClick={() => enrichmentRef.current?.handleSave()}
                    disabled={!enrichmentForm.canSave}
                  >
                    {enrichmentForm.isPending ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      "Apply Changes"
                    )}
                  </Button>
                </div>
              )}

              {/* Remove override + info */}
              {data.wikidata_enrichment?.has_override && !enrichmentForm.isDirty && (
                <div className="pt-2 space-y-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      deleteMutation.mutate(
                        { teamId: data.team.team_id },
                        {
                          onSuccess: () => toast.success("Override removed"),
                          onError: (err) => toast.error(err.message),
                        }
                      )
                    }
                    disabled={deleteMutation.isPending}
                    className="text-destructive hover:text-destructive"
                  >
                    {deleteMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4 mr-1" />
                    )}
                    Remove override
                  </Button>
                  {data.wikidata_enrichment.override?.updated_at && (
                    <div className="text-xs text-muted-foreground">
                      Override by {data.wikidata_enrichment.override.source || "manual"} ·{" "}
                      {new Date(data.wikidata_enrichment.override.updated_at).toLocaleDateString()}
                    </div>
                  )}
                </div>
              )}
            </SurfaceCard>
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

/**
 * Squad overview for TeamDrawer: current manager + active injuries.
 * Renders nothing if no squad data available (no placeholder).
 */
function TeamSquadOverview({ teamId }: { teamId: number }) {
  const { data, isLoading } = useTeamSquad(teamId);

  if (isLoading || !data) return null;

  const hasManager = !!data.current_manager;
  const hasInjuries = data.current_injuries.length > 0;

  if (!hasManager && !hasInjuries) return null;

  return (
    <div className="space-y-3">
      {hasManager && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-1.5">Manager</h4>
          <ManagerCard manager={data.current_manager!} />
        </div>
      )}
      {hasInjuries && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-1.5">
            Absences ({data.current_injuries.length})
          </h4>
          <InjuryList injuries={data.current_injuries} compact />
        </div>
      )}
    </div>
  );
}
