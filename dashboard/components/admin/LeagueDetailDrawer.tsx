"use client";

import { useState } from "react";
import { DetailDrawer } from "@/components/shell/DetailDrawer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader } from "@/components/ui/loader";
import { IconTabs } from "@/components/ui/icon-tabs";
import { Pencil, BarChart3, Calendar } from "lucide-react";
import { useAdminLeague, useAdminLeagueDetail } from "@/lib/hooks";
import { useAdminLeagueMutation } from "@/lib/hooks/use-admin-league-mutation";
import { cn } from "@/lib/utils";
import type {
  AdminLeagueDetailCore,
  AdminLeagueDetailData,
  AdminLeagueSeasonStats,
  AdminLeagueTitanCoverage,
  AdminLeagueRecentMatch,
} from "@/lib/types";

interface LeagueDetailDrawerProps {
  leagueId: number | null;
  onClose: () => void;
}

const PRIORITY_OPTIONS = ["high", "medium", "low"] as const;
const KIND_OPTIONS = ["league", "cup", "international", "friendly"] as const;

type DrawerTab = "edit" | "stats" | "matches";

const LEAGUE_TABS = [
  { id: "edit", icon: <Pencil />, label: "Edit" },
  { id: "stats", icon: <BarChart3 />, label: "Stats" },
  { id: "matches", icon: <Calendar />, label: "Matches" },
];

export function LeagueDetailDrawer({ leagueId, onClose }: LeagueDetailDrawerProps) {
  const { data, isLoading } = useAdminLeague(leagueId);
  const [activeTab, setActiveTab] = useState<DrawerTab>("edit");

  const tabBar = data ? (
    <IconTabs
      tabs={LEAGUE_TABS}
      value={activeTab}
      onValueChange={(v) => setActiveTab(v as DrawerTab)}
      className="w-full"
    />
  ) : undefined;

  return (
    <DetailDrawer
      open={leagueId !== null}
      onClose={onClose}
      title={data?.name ?? "League Details"}
      variant="overlay"
      fixedContent={tabBar}
    >
      {isLoading && (
        <div className="flex items-center justify-center h-48">
          <Loader size="md" />
        </div>
      )}
      {!isLoading && !data && leagueId !== null && (
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          Failed to load league
        </div>
      )}
      {data && activeTab === "edit" && (
        <LeagueForm key={data.league_id} league={data} />
      )}
      {data && activeTab === "stats" && (
        <StatsTab leagueId={data.league_id} />
      )}
      {data && activeTab === "matches" && (
        <MatchesTab leagueId={data.league_id} />
      )}
    </DetailDrawer>
  );
}

// =============================================================================
// Edit Tab (existing form, unchanged)
// =============================================================================

function LeagueForm({ league }: { league: AdminLeagueDetailCore }) {
  const mutation = useAdminLeagueMutation();

  const [isActive, setIsActive] = useState(league.is_active);
  const [kind, setKind] = useState(league.kind);
  const [priority, setPriority] = useState(league.priority);
  const [matchWeight, setMatchWeight] = useState(
    league.match_weight != null ? String(league.match_weight) : ""
  );
  const [lastAuditId, setLastAuditId] = useState<number | null>(null);
  const [lastChanges, setLastChanges] = useState<string[]>([]);

  const isDirty =
    isActive !== league.is_active ||
    kind !== league.kind ||
    priority !== league.priority ||
    matchWeight !== (league.match_weight != null ? String(league.match_weight) : "");

  function handleSave() {
    const body: Record<string, unknown> = {};

    if (isActive !== league.is_active) body.is_active = isActive;
    if (kind !== league.kind) body.kind = kind;
    if (priority !== league.priority) body.priority = priority;
    if (matchWeight !== (league.match_weight != null ? String(league.match_weight) : "")) {
      body.match_weight = matchWeight ? parseFloat(matchWeight) : null;
    }

    if (Object.keys(body).length === 0) return;

    mutation.mutate(
      { id: league.league_id, body },
      {
        onSuccess: (data) => {
          if (data?.audit_id) {
            setLastAuditId(data.audit_id);
          }
          setLastChanges(data?.changes_applied ?? []);
        },
      }
    );
  }

  return (
    <div className="space-y-3 text-sm">
      {/* Read-only info */}
      <div className="bg-card border border-border rounded-lg p-3">
        <div className="grid grid-cols-2 gap-3">
          <ReadOnlyField label="ID" value={String(league.league_id)} />
          <ReadOnlyField label="Source" value={league.source} />
          <ReadOnlyField label="Country" value={league.country} />
          <ReadOnlyField label="Match Type" value={league.match_type} />
          <ReadOnlyField label="Observed" value={league.observed ? "Yes" : "No"} />
          <ReadOnlyField label="Configured" value={league.configured ? "Yes" : "No"} />
        </div>
      </div>

      {/* Editable fields */}
      <div className="bg-card border border-border rounded-lg p-3 space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="is-active" className="text-sm">Active</Label>
          <Switch
            id="is-active"
            checked={isActive}
            onCheckedChange={setIsActive}
          />
        </div>

        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Kind</Label>
          <Select value={kind} onValueChange={setKind}>
            <SelectTrigger className="h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {KIND_OPTIONS.map((k) => (
                <SelectItem key={k} value={k}>{k}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Priority</Label>
          <Select value={priority} onValueChange={setPriority}>
            <SelectTrigger className="h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PRIORITY_OPTIONS.map((p) => (
                <SelectItem key={p} value={p}>{p}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1">
          <Label htmlFor="match-weight" className="text-xs text-muted-foreground">
            Match Weight
          </Label>
          <Input
            id="match-weight"
            type="number"
            step="0.1"
            value={matchWeight}
            onChange={(e) => setMatchWeight(e.target.value)}
            placeholder="null"
            className="h-8 text-sm"
          />
        </div>

        {/* Save button */}
        <div className="flex items-center gap-3 pt-2">
          <Button
            onClick={handleSave}
            disabled={!isDirty || mutation.isPending}
            size="sm"
          >
            {mutation.isPending ? "Saving…" : "Save"}
          </Button>
          {mutation.isSuccess && (
            <span className="text-xs text-green-600">
              Saved{lastAuditId ? ` (audit #${lastAuditId})` : ""}
              {lastChanges.length > 0 && ` — ${lastChanges.join(", ")}`}
            </span>
          )}
          {mutation.isError && (
            <span className="text-xs text-destructive">
              Error: {(mutation.error as Error)?.message ?? "Save failed"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Stats Tab
// =============================================================================

function StatsTab({ leagueId }: { leagueId: number }) {
  const { data, isLoading } = useAdminLeagueDetail(leagueId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        No data available
      </div>
    );
  }

  return (
    <div className="space-y-3 text-sm">
      {/* TITAN Coverage */}
      {data.titan_coverage && (
        <div className="bg-card border border-border rounded-lg p-3">
          <TitanCoverageCard coverage={data.titan_coverage} />
        </div>
      )}

      {/* Stats by season */}
      {data.stats_by_season.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-3 space-y-2">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Stats by Season
          </h3>
          <div className="rounded-md border border-border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-muted/50">
                  <th className="text-left px-2 py-1.5 font-medium">Season</th>
                  <th className="text-right px-2 py-1.5 font-medium">Matches</th>
                  <th className="text-right px-2 py-1.5 font-medium">Finished</th>
                  <th className="text-right px-2 py-1.5 font-medium">Stats%</th>
                  <th className="text-right px-2 py-1.5 font-medium">Odds%</th>
                </tr>
              </thead>
              <tbody>
                {data.stats_by_season.map((s) => (
                  <SeasonRow key={s.season} stats={s} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Teams count */}
      <div className="bg-card border border-border rounded-lg p-3 text-xs text-muted-foreground">
        Teams: <span className="font-mono font-medium text-foreground">{data.teams.length}</span>
      </div>
    </div>
  );
}

function TitanCoverageCard({ coverage }: { coverage: AdminLeagueTitanCoverage }) {
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        TITAN Coverage
      </h3>
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-md border border-border p-2">
          <p className="text-[10px] text-muted-foreground">Tier 1</p>
          <p className="text-lg font-semibold">{coverage.tier1_pct}%</p>
          <p className="text-[10px] text-muted-foreground">{coverage.tier1}/{coverage.total}</p>
        </div>
        <div className="rounded-md border border-border p-2">
          <p className="text-[10px] text-muted-foreground">Tier 1b</p>
          <p className="text-lg font-semibold">{coverage.tier1b_pct}%</p>
          <p className="text-[10px] text-muted-foreground">{coverage.tier1b}/{coverage.total}</p>
        </div>
      </div>
    </div>
  );
}

function SeasonRow({ stats }: { stats: AdminLeagueSeasonStats }) {
  return (
    <tr className="border-t border-border">
      <td className="px-2 py-1.5 font-mono">{stats.season}</td>
      <td className="px-2 py-1.5 text-right font-mono">{stats.total_matches}</td>
      <td className="px-2 py-1.5 text-right font-mono">{stats.finished}</td>
      <td className="px-2 py-1.5 text-right font-mono">{stats.with_stats_pct}%</td>
      <td className="px-2 py-1.5 text-right font-mono">{stats.with_odds_pct}%</td>
    </tr>
  );
}

// =============================================================================
// Matches Tab
// =============================================================================

function MatchesTab({ leagueId }: { leagueId: number }) {
  const { data, isLoading } = useAdminLeagueDetail(leagueId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        No data available
      </div>
    );
  }

  return (
    <div className="space-y-3 text-sm">
      {/* Recent matches */}
      <div className="bg-card border border-border rounded-lg p-3">
        {data.recent_matches.length > 0 ? (
          <div className="space-y-2">
            <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Recent Matches
            </h3>
            <div className="space-y-1">
              {data.recent_matches.map((m) => (
                <RecentMatchRow key={m.match_id} match={m} />
              ))}
            </div>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground text-center py-2">
            No recent matches
          </div>
        )}
      </div>

      {/* Teams */}
      {data.teams.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-3 space-y-2">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Teams ({data.teams.length})
          </h3>
          <div className="space-y-0.5 max-h-48 overflow-y-auto">
            {data.teams.map((t) => (
              <div key={t.team_id} className="flex items-center justify-between text-xs">
                <span>{t.name}</span>
                <span className="font-mono text-muted-foreground">{t.matches_in_league} matches</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RecentMatchRow({ match }: { match: AdminLeagueRecentMatch }) {
  return (
    <div className="flex items-center justify-between text-xs py-0.5">
      <span className="truncate flex-1">
        {match.home} vs {match.away}
      </span>
      <div className="flex items-center gap-2 shrink-0 ml-2">
        <Badge
          variant={match.status === "FT" ? "default" : "outline"}
          className="text-[10px] px-1.5"
        >
          {match.status}
        </Badge>
        {match.has_stats && (
          <span className="text-[10px] text-green-600" title="Has stats">S</span>
        )}
        {match.has_prediction && (
          <span className="text-[10px] text-blue-600" title="Has prediction">P</span>
        )}
        <span className="text-muted-foreground text-[10px]">
          {new Date(match.date).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
}

// =============================================================================
// Shared
// =============================================================================

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium">{value}</p>
    </div>
  );
}
