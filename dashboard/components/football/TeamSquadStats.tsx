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

type SortKey =
  | "name" | "pos" | "app" | "min" | "rtg" | "g" | "a" | "sv"
  | "kp" | "tkl" | "int" | "blk"
  | "sh" | "sot" | "pas" | "pacc"
  | "dtot" | "dwon" | "drba" | "drbs"
  | "fld" | "fc";

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
    case "a": return p.assists;
    case "sv": return p.saves;
    case "kp": return p.key_passes;
    case "tkl": return p.tackles;
    case "int": return p.interceptions;
    case "blk": return p.blocks;
    case "sh": return p.shots_total;
    case "sot": return p.shots_on_target;
    case "pas": return p.passes_total;
    case "pacc": return p.passes_accuracy ?? -1;
    case "dtot": return p.duels_total;
    case "dwon": return p.duels_won;
    case "drba": return p.dribbles_attempts;
    case "drbs": return p.dribbles_success;
    case "fld": return p.fouls_drawn;
    case "fc": return p.fouls_committed;
  }
}

interface ColDef {
  key: SortKey;
  label: string;
  group: "core" | "attack" | "passing" | "defense" | "duels" | "discipline";
}

const COLUMNS: ColDef[] = [
  // Core
  { key: "name", label: "Player", group: "core" },
  { key: "pos", label: "Pos", group: "core" },
  { key: "app", label: "App", group: "core" },
  { key: "min", label: "Min", group: "core" },
  { key: "rtg", label: "Rtg", group: "core" },
  // Attack
  { key: "g", label: "G", group: "attack" },
  { key: "a", label: "A", group: "attack" },
  { key: "sh", label: "Sh", group: "attack" },
  { key: "sot", label: "SoT", group: "attack" },
  // Passing
  { key: "pas", label: "Pas", group: "passing" },
  { key: "kp", label: "KP", group: "passing" },
  { key: "pacc", label: "Acc%", group: "passing" },
  // Defense
  { key: "tkl", label: "Tkl", group: "defense" },
  { key: "int", label: "Int", group: "defense" },
  { key: "blk", label: "Blk", group: "defense" },
  { key: "sv", label: "Sv", group: "defense" },
  // Duels
  { key: "dtot", label: "Duel", group: "duels" },
  { key: "dwon", label: "DW", group: "duels" },
  { key: "drba", label: "Drb", group: "duels" },
  { key: "drbs", label: "DrS", group: "duels" },
  // Discipline
  { key: "fld", label: "FD", group: "discipline" },
  { key: "fc", label: "FC", group: "discipline" },
];

const GROUP_LABELS: Record<string, string> = {
  core: "",
  attack: "Attack",
  passing: "Passing",
  defense: "Defense",
  duels: "Duels",
  discipline: "Disc.",
};

const GROUP_COLORS: Record<string, string> = {
  core: "",
  attack: "text-emerald-500",
  passing: "text-blue-500",
  defense: "text-amber-500",
  duels: "text-purple-500",
  discipline: "text-red-400",
};

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

  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      if (sortDir === "desc") setSortDir("asc");
      else { setSortKey(null); setSortDir("desc"); }
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

  // Build group header spans
  const groupSpans: { group: string; span: number }[] = [];
  for (const col of COLUMNS) {
    const last = groupSpans[groupSpans.length - 1];
    if (last && last.group === col.group) {
      last.span++;
    } else {
      groupSpans.push({ group: col.group, span: 1 });
    }
  }

  function renderCellValue(p: PlayerRow, key: SortKey): React.ReactNode {
    switch (key) {
      case "rtg": return formatRating(p.avg_rating);
      case "pacc": return p.passes_accuracy != null ? `${p.passes_accuracy}%` : "—";
      case "g": return p.goals || "—";
      case "a": return p.assists || "—";
      case "sv": return p.saves || "—";
      case "sh": return p.shots_total || "—";
      case "sot": return p.shots_on_target || "—";
      case "pas": return p.passes_total || "—";
      case "kp": return p.key_passes || "—";
      case "tkl": return p.tackles || "—";
      case "int": return p.interceptions || "—";
      case "blk": return p.blocks || "—";
      case "dtot": return p.duels_total || "—";
      case "dwon": return p.duels_won || "—";
      case "drba": return p.dribbles_attempts || "—";
      case "drbs": return p.dribbles_success || "—";
      case "fld": return p.fouls_drawn || "—";
      case "fc": return p.fouls_committed || "—";
      default: return "—";
    }
  }

  return (
    <div className="flex flex-col max-h-[calc(100dvh-300px)] overflow-hidden">
      {sortedPlayers.length === 0 ? (
        <div className="py-10 text-center">
          <p className="text-sm text-muted-foreground">No player stats for this season.</p>
        </div>
      ) : (
        <div className="overflow-auto flex-1 min-h-0">
          <table className="min-w-[1100px]">
            <thead className="sticky top-0 z-10 bg-background">
              {/* Group header row */}
              <tr className="border-b border-border/50">
                <th className="sticky left-0 z-20 bg-background" style={{ width: 28, minWidth: 28, maxWidth: 28 }} />
                {groupSpans.map(({ group, span }) => (
                  <th
                    key={group}
                    colSpan={group === "core" ? span : span}
                    className={cn(
                      "py-1 text-[10px] font-semibold uppercase tracking-wider",
                      group !== "core" && "border-l border-border/30",
                      GROUP_COLORS[group] || "text-muted-foreground/50"
                    )}
                  >
                    {GROUP_LABELS[group]}
                  </th>
                ))}
              </tr>
              {/* Column header row */}
              <tr className="border-b border-border">
                <th className="text-center py-2 text-xs font-medium text-muted-foreground sticky left-0 z-20 bg-background" style={{ width: 28, minWidth: 28, maxWidth: 28 }}>#</th>
                {COLUMNS.map((col, ci) => {
                  const isGroupStart = ci > 0 && COLUMNS[ci - 1].group !== col.group;
                  return (
                    <th
                      key={col.key}
                      onClick={() => handleSort(col.key)}
                      className={cn(
                        "py-2 text-xs font-medium select-none cursor-pointer transition-colors hover:text-foreground whitespace-nowrap",
                        col.key === "name" ? "px-3 text-left min-w-[180px] sticky left-[28px] z-20 bg-background" : "px-2 text-center min-w-[48px]",
                        sortKey === col.key ? "text-foreground" : "text-muted-foreground",
                        isGroupStart && "border-l border-border/30"
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
                  );
                })}
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
                        src={playerPhotoUrl(p.player_external_id)}
                        alt=""
                        width={26}
                        height={26}
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
                  {/* Pos */}
                  <td className="px-2 py-2 text-center text-xs text-muted-foreground group-hover/row:bg-accent/50 transition-colors">
                    {p.posKey}
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
                  {/* Attack, Passing, Defense, Duels, Discipline — all remaining columns */}
                  {COLUMNS.slice(5).map((col, ci) => {
                    const isGroupStart = ci > 0 && COLUMNS[5 + ci - 1].group !== col.group;
                    return (
                      <td
                        key={col.key}
                        className={cn(
                          "px-2 py-2 text-center text-sm text-muted-foreground tabular-nums group-hover/row:bg-accent/50 transition-colors",
                          isGroupStart && "border-l border-border/30"
                        )}
                      >
                        {renderCellValue(p, col.key)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
