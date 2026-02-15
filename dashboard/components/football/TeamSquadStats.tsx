"use client";

import { useMemo, useState } from "react";
import { useTeamSquadStats } from "@/lib/hooks";
import type { TeamSquadPlayerSeasonStats } from "@/lib/types/squad";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { SurfaceCard } from "@/components/ui/surface-card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertTriangle, RefreshCw, Users } from "lucide-react";

const EMPTY_PLAYERS: TeamSquadPlayerSeasonStats[] = [];

function formatRating(r: number | null): string {
  if (r === null || Number.isNaN(r)) return "—";
  return `★${r.toFixed(1)}`;
}

function posGroup(pos: string | null | undefined): "G" | "D" | "M" | "F" | "U" {
  const p = (pos || "U").toUpperCase();
  if (p === "G") return "G";
  if (p === "D") return "D";
  if (p === "M") return "M";
  if (p === "F") return "F";
  return "U";
}

const GROUP_LABEL: Record<string, string> = {
  G: "GOALKEEPERS",
  D: "DEFENDERS",
  M: "MIDFIELDERS",
  F: "FORWARDS",
  U: "OTHER",
};

export function TeamSquadStats({ teamId }: { teamId: number }) {
  const [season, setSeason] = useState<number | null>(null);
  const { data, isLoading, error, refetch, isFetching } = useTeamSquadStats(teamId, season);

  const selectedSeason = season ?? data?.season ?? null;
  const seasons = data?.available_seasons ?? [];
  const players = data?.players ?? EMPTY_PLAYERS;

  const { grouped, teamAvgRating, totalMinutes } = useMemo(() => {
    const groups: Record<string, TeamSquadPlayerSeasonStats[]> = { G: [], D: [], M: [], F: [], U: [] };

    let sumWeighted = 0;
    let sumMinutes = 0;
    for (const p of players) {
      groups[posGroup(p.position)].push(p);
      if (p.avg_rating !== null && p.total_minutes > 0) {
        sumWeighted += p.avg_rating * p.total_minutes;
        sumMinutes += p.total_minutes;
      }
    }

    // Sort within each group
    for (const k of Object.keys(groups)) {
      groups[k].sort((a, b) => {
        if (b.appearances !== a.appearances) return b.appearances - a.appearances;
        if (b.total_minutes !== a.total_minutes) return b.total_minutes - a.total_minutes;
        const ar = a.avg_rating ?? -1;
        const br = b.avg_rating ?? -1;
        return br - ar;
      });
    }

    return {
      grouped: groups,
      teamAvgRating: sumMinutes > 0 ? sumWeighted / sumMinutes : null,
      totalMinutes: sumMinutes,
    };
  }, [players]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-10">
        <Loader size="md" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center gap-3 text-center py-10">
        <AlertTriangle className="h-10 w-10 text-[var(--status-warning-text)]" />
        <div>
          <p className="text-sm font-medium text-foreground mb-1">Squad stats unavailable</p>
          <p className="text-xs text-muted-foreground">{error.message}</p>
        </div>
        <Button onClick={() => refetch()} variant="secondary" size="sm">
          <RefreshCw className="h-3 w-3 mr-1" />
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4 pb-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Users className="h-4 w-4 text-muted-foreground" />
          <h4 className="text-sm font-medium text-foreground">Squad</h4>
          {isFetching && <span className="text-xs text-muted-foreground">(updating)</span>}
        </div>

        <div className="flex items-center gap-2">
          <Select
            value={selectedSeason ? String(selectedSeason) : ""}
            onValueChange={(v) => setSeason(Number(v))}
            disabled={!seasons || seasons.length === 0}
          >
            <SelectTrigger size="sm" className="min-w-[120px]">
              <SelectValue placeholder="Season" />
            </SelectTrigger>
            <SelectContent align="end">
              {seasons.map((s) => (
                <SelectItem key={s} value={String(s)}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {players.length === 0 ? (
        <SurfaceCard className="py-10 text-center">
          <p className="text-sm text-muted-foreground">No player stats for this season.</p>
        </SurfaceCard>
      ) : (
        <>
          {(["G", "D", "M", "F", "U"] as const).map((k) => {
            const groupPlayers = grouped[k];
            if (!groupPlayers || groupPlayers.length === 0) return null;

            return (
              <div key={k} className="space-y-2">
                <div className="text-xs font-medium text-muted-foreground">
                  {GROUP_LABEL[k]} ({groupPlayers.length})
                </div>
                <SurfaceCard className="p-0 overflow-hidden">
                  <div className="divide-y divide-border">
                    {groupPlayers.map((p) => (
                      <div key={p.player_external_id} className="px-3 py-2 flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <div className="text-sm font-medium text-foreground truncate">
                              {p.player_name}
                            </div>
                            {p.ever_captain && (
                              <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">
                                C
                              </Badge>
                            )}
                          </div>
                          <div className="mt-0.5 text-xs text-muted-foreground flex items-center gap-2 flex-wrap">
                            <span className="tabular-nums">
                              {p.goals}G · {p.assists}A
                              {posGroup(p.position) === "G" ? ` · ${p.saves}sv` : ""}
                            </span>
                            {(p.yellows > 0 || p.reds > 0) && (
                              <span className="tabular-nums">
                                {p.yellows > 0 ? `🟨${p.yellows}` : ""}
                                {p.yellows > 0 && p.reds > 0 ? " " : ""}
                                {p.reds > 0 ? `🟥${p.reds}` : ""}
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="shrink-0 text-right">
                          <div className="text-sm font-semibold text-foreground tabular-nums">
                            {formatRating(p.avg_rating)}
                          </div>
                          <div className="text-xs text-muted-foreground tabular-nums">
                            {p.appearances} app · {p.total_minutes} min
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </SurfaceCard>
              </div>
            );
          })}

          <div className="text-xs text-muted-foreground text-center pt-1">
            {players.length} players · avg{" "}
            <span className="font-medium text-foreground tabular-nums">
              {teamAvgRating !== null ? `★${teamAvgRating.toFixed(1)}` : "—"}
            </span>
            {totalMinutes > 0 ? (
              <span className="opacity-70"> · {totalMinutes.toLocaleString()} min</span>
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}

