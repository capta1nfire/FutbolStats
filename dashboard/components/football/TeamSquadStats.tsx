"use client";

import Image from "next/image";
import { useCallback, useMemo, useState } from "react";
import { useTeamSquadStats } from "@/lib/hooks";
import type { TeamSquadPlayerSeasonStats } from "@/lib/types/squad";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { AlertTriangle, RefreshCw, ChevronUp, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

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

// Position sort order: G=0, D=1, M=2, F=3, U=4
const POS_ORDER: Record<string, number> = { G: 0, D: 1, M: 2, F: 3, U: 4 };

type SortKey = "name" | "pos" | "app" | "min" | "rtg" | "g" | "a";
type SortDir = "asc" | "desc";

interface PlayerRow extends TeamSquadPlayerSeasonStats {
  posKey: "G" | "D" | "M" | "F" | "U";
}

function getSortValue(p: PlayerRow, key: SortKey): number | string {
  switch (key) {
    case "name": return p.player_name.toLowerCase();
    case "pos": return POS_ORDER[p.posKey] ?? 4;
    case "app": return p.appearances;
    case "min": return p.total_minutes;
    case "rtg": return p.avg_rating ?? -1;
    case "g": return p.goals;
    case "a": return p.posKey === "G" ? p.saves : p.assists;
  }
}

interface TeamSquadStatsProps {
  teamId: number;
  season: number | null;
}

export function TeamSquadStats({ teamId, season }: TeamSquadStatsProps) {
  const { data, isLoading, error, refetch } = useTeamSquadStats(teamId, season);
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const players = data?.players ?? EMPTY_PLAYERS;

  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      if (sortDir === "desc") setSortDir("asc");
      else { setSortKey(null); setSortDir("desc"); } // third click resets
    } else {
      setSortKey(key);
      setSortDir(key === "name" ? "asc" : "desc");
    }
  }, [sortKey, sortDir]);

  const sortedPlayers = useMemo(() => {
    const rows: PlayerRow[] = players.map((p) => ({
      ...p,
      posKey: posGroup(p.position),
    }));

    if (!sortKey) {
      // Default: group by position, then by appearances desc
      rows.sort((a, b) => {
        const posA = POS_ORDER[a.posKey] ?? 4;
        const posB = POS_ORDER[b.posKey] ?? 4;
        if (posA !== posB) return posA - posB;
        if (b.appearances !== a.appearances) return b.appearances - a.appearances;
        if (b.total_minutes !== a.total_minutes) return b.total_minutes - a.total_minutes;
        return (b.avg_rating ?? -1) - (a.avg_rating ?? -1);
      });
    } else {
      const dir = sortDir === "asc" ? 1 : -1;
      rows.sort((a, b) => {
        const va = getSortValue(a, sortKey);
        const vb = getSortValue(b, sortKey);
        if (va < vb) return -1 * dir;
        if (va > vb) return 1 * dir;
        return 0;
      });
    }

    return rows;
  }, [players, sortKey, sortDir]);

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

  const columns: { key: SortKey; label: string; align: "left" | "center" }[] = [
    { key: "name", label: "Player", align: "left" },
    { key: "pos", label: "Pos", align: "center" },
    { key: "app", label: "App", align: "center" },
    { key: "min", label: "Min", align: "center" },
    { key: "rtg", label: "Rtg", align: "center" },
    { key: "g", label: "G", align: "center" },
    { key: "a", label: "A", align: "center" },
  ];

  return (
    <div className="flex flex-col max-h-[calc(100dvh-300px)] overflow-hidden">
      {sortedPlayers.length === 0 ? (
        <div className="py-10 text-center">
          <p className="text-sm text-muted-foreground">No player stats for this season.</p>
        </div>
      ) : (
        <div className="overflow-auto flex-1 min-h-0">
          <table className="w-full table-fixed">
            <colgroup><col style={{ width: "6%" }} /><col style={{ width: "42%" }} /><col style={{ width: "7%" }} /><col style={{ width: "9%" }} /><col style={{ width: "11%" }} /><col style={{ width: "10%" }} /><col style={{ width: "7%" }} /><col style={{ width: "8%" }} /></colgroup>
            <thead className="sticky top-0 z-10 bg-background">
              <tr className="border-b border-border">
                <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground">#</th>
                {columns.map((col) => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className={cn(
                      "py-2 text-xs font-medium select-none cursor-pointer transition-colors hover:text-foreground",
                      col.align === "left" ? "px-3 text-left" : "px-1 text-center",
                      sortKey === col.key ? "text-foreground" : "text-muted-foreground"
                    )}
                  >
                    <span className="inline-flex items-center gap-0.5">
                      {col.label}
                      {sortKey === col.key && (
                        sortDir === "asc"
                          ? <ChevronUp className="w-3 h-3" />
                          : <ChevronDown className="w-3 h-3" />
                      )}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedPlayers.map((p, idx) => (
                <tr
                  key={p.player_external_id}
                  className="border-b border-border transition-colors hover:bg-accent/50"
                >
                  <td className="text-center py-1.5 px-2 text-[11px] text-muted-foreground/50 tabular-nums">
                    {idx + 1}
                  </td>
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
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="text-sm font-medium text-foreground truncate">
                            {p.player_name}
                          </span>
                          {p.jersey_number != null && (
                            <span className="text-[10px] tabular-nums text-muted-foreground bg-muted px-1 py-px rounded shrink-0">
                              {p.jersey_number}
                            </span>
                          )}
                          {p.ever_captain && (
                            <span className="text-[9px] font-medium text-muted-foreground bg-muted px-1 py-px rounded shrink-0">
                              c
                            </span>
                          )}
                        </div>
                        {(p.yellows > 0 || p.reds > 0) && (
                          <div className="flex items-center gap-0.5 mt-0.5">
                            {Array.from({ length: p.yellows }, (_, i) => (
                              <span key={`y${i}`} className="inline-block w-2 h-2 rounded-full bg-yellow-400" />
                            ))}
                            {Array.from({ length: p.reds }, (_, i) => (
                              <span key={`r${i}`} className="inline-block w-2 h-2 rounded-full bg-red-500" />
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="px-1 py-2.5 text-center text-xs text-muted-foreground">
                    {p.posKey}
                  </td>
                  <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                    {p.appearances}
                  </td>
                  <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                    {p.total_minutes}
                  </td>
                  <td className="px-1 py-2.5 text-center text-sm font-semibold text-foreground tabular-nums">
                    {formatRating(p.avg_rating)}
                  </td>
                  <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                    {p.goals || "—"}
                  </td>
                  <td className="px-1 py-2.5 text-center text-sm text-muted-foreground tabular-nums">
                    {p.posKey === "G" ? (p.saves || "—") : (p.assists || "—")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
