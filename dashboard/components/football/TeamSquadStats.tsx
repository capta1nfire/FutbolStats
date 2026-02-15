"use client";

import Image from "next/image";
import { useMemo } from "react";
import { useTeamSquadStats } from "@/lib/hooks";
import type { TeamSquadPlayerSeasonStats } from "@/lib/types/squad";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { AlertTriangle, RefreshCw } from "lucide-react";

function playerPhotoUrl(externalId: number): string {
  return `https://media.api-sports.io/football/players/${externalId}.png`;
}

const EMPTY_PLAYERS: TeamSquadPlayerSeasonStats[] = [];

function formatRating(r: number | null): string {
  if (r === null || Number.isNaN(r)) return "—";
  return r.toFixed(1);
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
  G: "Goalkeepers",
  D: "Defenders",
  M: "Midfielders",
  F: "Forwards",
  U: "Other",
};

interface TeamSquadStatsProps {
  teamId: number;
  season: number | null;
}

export function TeamSquadStats({ teamId, season }: TeamSquadStatsProps) {
  const { data, isLoading, error, refetch } = useTeamSquadStats(teamId, season);

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

  // Check if any group has goalkeepers (for conditional Sv column)
  const hasGoalkeepers = (grouped["G"]?.length ?? 0) > 0;

  return (
    <div className="flex flex-col max-h-[600px] rounded-lg border border-border overflow-hidden">
      {players.length === 0 ? (
        <div className="py-10 text-center">
          <p className="text-sm text-muted-foreground">No player stats for this season.</p>
        </div>
      ) : (
        <div className="overflow-auto flex-1 min-h-0">
          <table className="w-full table-fixed">
            <colgroup><col style={{ width: "38%" }} /><col style={{ width: "7%" }} /><col style={{ width: "9%" }} /><col style={{ width: "10%" }} /><col style={{ width: "9%" }} /><col style={{ width: "7%" }} /><col style={{ width: "8%" }} /><col style={{ width: "12%" }} /></colgroup>
            {/* Sticky header */}
            <thead className="sticky top-0 z-10 bg-background">
              <tr className="border-b border-border">
                <th className="px-3 py-2.5 text-left text-xs font-semibold text-muted-foreground">Player</th>
                <th className="px-1 py-2.5 text-center text-xs font-semibold text-muted-foreground">Pos</th>
                <th className="px-1 py-2.5 text-center text-xs font-semibold text-muted-foreground">App</th>
                <th className="px-1 py-2.5 text-center text-xs font-semibold text-muted-foreground">Min</th>
                <th className="px-1 py-2.5 text-center text-xs font-semibold text-muted-foreground">Rtg</th>
                <th className="px-1 py-2.5 text-center text-xs font-semibold text-muted-foreground">G</th>
                <th className="px-1 py-2.5 text-center text-xs font-semibold text-muted-foreground">A</th>
                <th className="px-1 py-2.5 text-center text-xs font-semibold text-muted-foreground"></th>
              </tr>
            </thead>
            <tbody>
              {(["G", "D", "M", "F", "U"] as const).map((k) => {
                const groupPlayers = grouped[k];
                if (!groupPlayers || groupPlayers.length === 0) return null;

                return groupPlayers.map((p, idx) => (
                  <tr
                    key={p.player_external_id}
                    className="border-b border-border transition-colors hover:bg-accent/50"
                  >
                    {/* Player cell */}
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <Image
                          src={playerPhotoUrl(p.player_external_id)}
                          alt=""
                          width={28}
                          height={28}
                          className="rounded-full shrink-0 object-cover"
                          unoptimized
                        />
                        {p.jersey_number != null && (
                          <span className="text-[10px] tabular-nums text-muted-foreground shrink-0 w-4 text-right">
                            {p.jersey_number}
                          </span>
                        )}
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="text-sm font-medium text-foreground truncate">
                            {p.player_name}
                          </span>
                          {p.ever_captain && (
                            <span className="text-[9px] font-medium text-muted-foreground bg-muted px-1 py-px rounded shrink-0">
                              c
                            </span>
                          )}
                        </div>
                      </div>
                    </td>
                    {/* Position */}
                    <td className="px-1 py-2.5 text-center text-xs text-muted-foreground">
                      {k}
                    </td>
                    {/* App */}
                    <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                      {p.appearances}
                    </td>
                    {/* Min */}
                    <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                      {p.total_minutes}
                    </td>
                    {/* Rating */}
                    <td className="px-1 py-2.5 text-center text-sm font-semibold text-foreground tabular-nums">
                      {formatRating(p.avg_rating)}
                    </td>
                    {/* Goals */}
                    <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                      {p.goals || "—"}
                    </td>
                    {/* Assists / Saves for GK */}
                    <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                      {k === "G" ? (p.saves || "—") : (p.assists || "—")}
                    </td>
                    {/* Cards */}
                    <td className="px-1 py-2.5 text-center tabular-nums whitespace-nowrap">
                      {p.yellows > 0 && (
                        <span className="inline-flex items-center gap-px text-[11px]">
                          <span className="inline-block w-2.5 h-3 rounded-[1px] bg-yellow-400" />
                          <span className="text-muted-foreground">{p.yellows}</span>
                        </span>
                      )}
                      {p.yellows > 0 && p.reds > 0 && <span className="mx-0.5" />}
                      {p.reds > 0 && (
                        <span className="inline-flex items-center gap-px text-[11px]">
                          <span className="inline-block w-2.5 h-3 rounded-[1px] bg-red-500" />
                          <span className="text-muted-foreground">{p.reds}</span>
                        </span>
                      )}
                    </td>
                  </tr>
                ));
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
