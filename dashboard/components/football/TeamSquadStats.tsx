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

const POS_LABEL: Record<string, string> = {
  G: "Goalkeeper",
  D: "Defender",
  M: "Midfielder",
  F: "Forward",
  U: "—",
};

// Position sort order: G=0, D=1, M=2, F=3, U=4
const POS_ORDER: Record<string, number> = { G: 0, D: 1, M: 2, F: 3, U: 4 };

type SortKey = "name" | "pos" | "app" | "min" | "rtg";
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
  }
}

interface ColDef {
  key: SortKey;
  label: string;
  align: "left" | "center";
}

const COLUMNS: ColDef[] = [
  { key: "name", label: "Player", align: "left" },
  { key: "pos", label: "Position", align: "left" },
  { key: "app", label: "App", align: "center" },
  { key: "min", label: "Min", align: "center" },
  { key: "rtg", label: "Rtg", align: "center" },
];

interface TeamSquadStatsProps {
  teamId: number;
  season: number | null;
  onPlayerSelect?: (player: TeamSquadPlayerSeasonStats) => void;
}

export function TeamSquadStats({ teamId, season, onPlayerSelect }: TeamSquadStatsProps) {
  const { data, isLoading, error, refetch } = useTeamSquadStats(teamId, season);
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const players = data?.players ?? EMPTY_PLAYERS;
  const teamMatchesPlayed = data?.team_matches_played ?? 0;

  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      if (sortDir === "desc") setSortDir("asc");
      else { setSortKey(null); setSortDir("desc"); }
    } else {
      setSortKey(key);
      setSortDir(key === "name" || key === "pos" ? "asc" : "desc");
    }
  }, [sortKey, sortDir]);

  const sortedPlayers = useMemo(() => {
    const rows: PlayerRow[] = players.map((p) => ({
      ...p,
      posKey: posGroup(p.position),
    }));

    if (!sortKey) {
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

  return (
    <div className="flex flex-col max-h-[calc(100dvh-300px)] overflow-hidden">
      {sortedPlayers.length === 0 ? (
        <div className="py-10 text-center">
          <p className="text-sm text-muted-foreground">No player stats for this season.</p>
        </div>
      ) : (
        <div className="overflow-auto flex-1 min-h-0">
          <table className="w-full">
            <thead className="sticky top-0 z-10 bg-background">
              <tr className="border-b border-border">
                <th className="text-center py-2 text-xs font-medium text-muted-foreground sticky left-0 z-20 bg-background" style={{ width: 28, minWidth: 28, maxWidth: 28 }}>#</th>
                {COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className={cn(
                      "py-2 text-xs font-medium select-none cursor-pointer transition-colors hover:text-foreground whitespace-nowrap",
                      col.align === "left" ? "px-3 text-left" : "px-2 text-center",
                      col.key === "name" && "min-w-[180px] sticky left-[28px] z-20 bg-background",
                      sortKey === col.key ? "text-foreground" : "text-muted-foreground"
                    )}
                  >
                    <span className="inline-flex items-center gap-0.5">
                      {col.label}
                      {col.key === "app" && teamMatchesPlayed > 0 && (
                        <span className="text-[9px] tabular-nums text-muted-foreground bg-muted px-1 py-px rounded ml-0.5">
                          /{teamMatchesPlayed}
                        </span>
                      )}
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
                  role={onPlayerSelect ? "button" : undefined}
                  tabIndex={onPlayerSelect ? 0 : undefined}
                  onClick={() => onPlayerSelect?.(p)}
                  onKeyDown={onPlayerSelect ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPlayerSelect(p); } } : undefined}
                  className={cn(
                    "group/row border-b border-border",
                    onPlayerSelect && "cursor-pointer"
                  )}
                >
                  <td className="text-center py-1.5 text-[11px] text-muted-foreground/50 tabular-nums sticky left-0 z-[5] bg-background group-hover/row:bg-accent/50 transition-colors" style={{ width: 28, minWidth: 28, maxWidth: 28 }}>
                    {idx + 1}
                  </td>
                  {/* Player name — sticky */}
                  <td className="px-3 py-2 min-w-[180px] sticky left-[28px] z-[5] bg-background group-hover/row:bg-accent/50 transition-colors">
                    <div className="flex items-center gap-2">
                      <Image
                        src={p.photo_url_thumb_hq || p.photo_url || playerPhotoUrl(p.player_external_id)}
                        alt=""
                        width={26}
                        height={26}
                        className="rounded-full shrink-0 object-cover"
                        unoptimized={!(p.photo_url_thumb_hq)}
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
                  {/* Position */}
                  <td className="px-3 py-2 text-left text-xs text-muted-foreground group-hover/row:bg-accent/50 transition-colors">
                    {POS_LABEL[p.posKey]}
                  </td>
                  {/* App */}
                  <td className="px-2 py-2 text-center text-sm text-muted-foreground tabular-nums group-hover/row:bg-accent/50 transition-colors">
                    {p.appearances}
                  </td>
                  {/* Min */}
                  <td className="px-2 py-2 text-center text-sm text-muted-foreground tabular-nums group-hover/row:bg-accent/50 transition-colors">
                    {p.total_minutes}
                  </td>
                  {/* Rtg */}
                  <td className="px-2 py-2 text-center text-sm font-semibold text-foreground tabular-nums group-hover/row:bg-accent/50 transition-colors">
                    {formatRating(p.avg_rating)}
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
