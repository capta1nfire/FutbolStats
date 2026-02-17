"use client";

import Image from "next/image";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useFootballLeague, useFootballTeam, useStandings, useTeamSquad, useTeamSquadStats } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import type { StandingEntry, DescensoData, ReclasificacionData, AvailableTable } from "@/lib/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { QualificationBadge } from "@/components/ui/qualification-badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Loader } from "@/components/ui/loader";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  RefreshCw,
  AlertTriangle,
  ArrowLeft,
  Trophy,
  Globe,
  Calendar,
  TrendingUp,
  MapPin,
  ShieldUser,
  Settings,
  ChevronDown,
} from "lucide-react";
import { toast } from "sonner";
import { useCoverageMap } from "@/lib/hooks/use-coverage-map";
import { useLiveLeagueMatches } from "@/lib/hooks/use-live-league-matches";
import type { LiveMatch } from "@/lib/hooks/use-live-league-matches";
import { useTeamLogos } from "@/lib/hooks/use-team-logos";
import { TeamLogo } from "@/components/ui/team-logo";
import { CoverageDetailContent } from "@/components/coverage-map/CoverageDetail";
import { FeatureCoverageSection } from "./FeatureCoverageSection";
import { TeamSquadStats } from "./TeamSquadStats";
import { TeamPerformanceCharts } from "./TeamPerformanceCharts";
import type { TeamSquadPlayerSeasonStats } from "@/lib/types/squad";

const TEAM_DETAIL_TABS = [
  { id: "squad", label: "Squad" },
  { id: "matches", label: "Matches" },
  { id: "stats", label: "Stats" },
  { id: "coverage", label: "Coverage" },
  { id: "transfers", label: "Transfers" },
  { id: "club", label: "Club" },
] as const;

type TeamDetailTabId = (typeof TEAM_DETAIL_TABS)[number]["id"];

function StadiumIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect x="1" y="4" width="22" height="16" rx="3" />
      <line x1="12" y1="4" x2="12" y2="20" />
      <circle cx="12" cy="12" r="3" />
      <rect x="1" y="8" width="4" height="8" rx="1" />
      <rect x="19" y="8" width="4" height="8" rx="1" />
    </svg>
  );
}

/**
 * Country Flag Component
 */
function CountryFlag({ country, size = 14 }: { country: string; size?: number }) {
  const isoCode = getCountryIsoCode(country);

  if (!isoCode) {
    return <Globe className="text-muted-foreground" style={{ width: size, height: size }} />;
  }

  return (
    <Image
      src={`/flags/${isoCode}.svg`}
      alt={`${country} flag`}
      width={size}
      height={size}
      className="rounded-full object-cover"
    />
  );
}

interface SiblingLeague {
  league_id: number;
  name: string;
  kind: string;
}

interface LeagueDetailProps {
  leagueId: number;
  onBack: () => void;
  onTeamSelect: (teamId: number) => void;
  onSettingsClick?: () => void;
  onCoverageClick?: () => void;
  /** Initial team to show in details panel (from URL param) */
  initialTeamId?: number | null;
  /** Other leagues from the same country (sorted by priority) */
  siblingLeagues?: SiblingLeague[];
  /** Called when user switches to a different league via dropdown */
  onLeagueChange?: (leagueId: number) => void;
  /** Called when user clicks a player row in Squad tab */
  onPlayerSelect?: (player: TeamSquadPlayerSeasonStats, matchesPlayed?: number) => void;
}

/**
 * Stats Table Component
 */
function StatsTable({
  stats,
}: {
  stats: {
    season: number;
    total_matches: number;
    finished: number;
    with_stats_pct: number | null;
    with_odds_pct: number | null;
  }[];
}) {
  if (!stats || stats.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No season stats available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground">Season</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Matches</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Finished</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Stats %</th>
            <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Odds %</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((s) => (
            <tr key={s.season} className="border-b border-border last:border-0">
              <td className="py-2 px-3 font-medium text-foreground">{s.season}</td>
              <td className="py-2 px-3 text-right text-muted-foreground">{s.total_matches}</td>
              <td className="py-2 px-3 text-right text-muted-foreground">{s.finished}</td>
              <td className="py-2 px-3 text-right text-muted-foreground">
                {s.with_stats_pct !== null ? `${s.with_stats_pct.toFixed(0)}%` : "-"}
              </td>
              <td className="py-2 px-3 text-right text-muted-foreground">
                {s.with_odds_pct !== null ? `${s.with_odds_pct.toFixed(0)}%` : "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Recent Matches List
 */
function RecentMatchesList({
  matches,
  onTeamSelect,
  useShortNames = false,
}: {
  matches: {
    match_id: number;
    date: string | null;
    status: string;
    home_team: string;
    away_team: string;
    home_display_name?: string;
    away_display_name?: string;
    home_team_id?: number;
    away_team_id?: number;
    score: string | null;
  }[];
  onTeamSelect?: (teamId: number) => void;
  useShortNames?: boolean;
}) {
  if (!matches || matches.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No recent matches
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {matches.map((match) => {
        const dateStr = match.date
          ? new Date(match.date).toLocaleDateString([], { month: "short", day: "numeric" })
          : "TBD";
        const timeStr = match.date
          ? new Date(match.date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
          : "--:--";

        return (
          <div key={match.match_id} className="py-2 px-3 flex items-center gap-3">
            <div className="w-20 shrink-0">
              <p className="text-xs text-muted-foreground">{dateStr}</p>
              <p className="text-xs text-muted-foreground">{timeStr}</p>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-foreground truncate">
                {match.home_team_id && onTeamSelect ? (
                  <button
                    onClick={() => onTeamSelect(match.home_team_id!)}
                    className="text-primary hover:text-primary-hover transition-colors no-underline hover:no-underline"
                  >
                    {useShortNames ? (match.home_display_name ?? match.home_team) : match.home_team}
                  </button>
                ) : (
                  <span>{useShortNames ? (match.home_display_name ?? match.home_team) : match.home_team}</span>
                )}
                <span className="text-muted-foreground"> vs </span>
                {match.away_team_id && onTeamSelect ? (
                  <button
                    onClick={() => onTeamSelect(match.away_team_id!)}
                    className="text-primary hover:text-primary-hover transition-colors no-underline hover:no-underline"
                  >
                    {useShortNames ? (match.away_display_name ?? match.away_team) : match.away_team}
                  </button>
                ) : (
                  <span>{useShortNames ? (match.away_display_name ?? match.away_team) : match.away_team}</span>
                )}
              </p>
            </div>
            <div className="shrink-0 text-right">
              {match.score ? (
                <span className="text-sm font-medium text-foreground">{match.score}</span>
              ) : (
                <span className="text-xs px-1.5 py-0.5 bg-muted rounded text-muted-foreground capitalize">
                  {match.status}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Standings Table Component
 */
function StandingsTable({
  standings,
  onTeamSelect,
  useShortNames = false,
  selectedTeamId,
}: {
  standings: StandingEntry[];
  onTeamSelect?: (teamId: number) => void;
  useShortNames?: boolean;
  selectedTeamId?: number | null;
}) {
  if (standings.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-4">
        No standings data
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">#</th>
            <th className="text-left py-2 px-2 text-xs font-medium text-muted-foreground">Team</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">P</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">W</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">D</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">L</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">GD</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">Pts</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-20">Form</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((entry) => {
            const teamClickable = entry.teamId > 0 && onTeamSelect;
            const isSelected = selectedTeamId != null && entry.teamId === selectedTeamId;

            return (
              <tr
                key={entry.position}
                data-team-id={entry.teamId}
                className={cn(
                  "border-b border-border last:border-0",
                  isSelected ? "bg-[var(--row-selected)]" : "hover:bg-accent/50",
                  teamClickable && "cursor-pointer"
                )}
                onClick={teamClickable ? () => onTeamSelect(entry.teamId) : undefined}
              >
                <td className="text-center py-1.5 px-2">
                  <QualificationBadge value={entry.position} active={entry.position <= 8} />
                </td>
                <td className="py-1.5 px-2">
                  <div className="flex items-center gap-2">
                    {entry.teamLogo ? (
                      <Image src={entry.teamLogo} alt="" width={16} height={16} className="rounded-full object-cover" />
                    ) : (
                      <Globe className="h-4 w-4 text-muted-foreground" />
                    )}
                    <span className="truncate text-foreground">
                      {useShortNames ? entry.displayName : entry.teamName}
                    </span>
                    {entry.description && entry.position > 8 && (
                      <span className="text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground shrink-0">
                        {entry.description}
                      </span>
                    )}
                  </div>
                </td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.played}</td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.won}</td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.drawn}</td>
                <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.lost}</td>
                <td className={cn(
                  "text-center py-1.5 px-2 font-medium",
                  entry.goalDiff > 0 ? "text-[var(--status-success-text)]" : entry.goalDiff < 0 ? "text-[var(--status-error-text)]" : "text-muted-foreground"
                )}>
                  {entry.goalDiff > 0 ? `+${entry.goalDiff}` : entry.goalDiff}
                </td>
                <td className="text-center py-1.5 px-2 font-bold text-foreground">{entry.points}</td>
                <td className="text-center py-1.5 px-2">
                  {entry.form ? (
                    <div className="flex items-center justify-center gap-0.5">
                      {entry.form.split("").slice(-5).map((ch, i) => (
                        <span
                          key={i}
                          className={cn(
                            "w-2 h-2 rounded-full",
                            ch === "W" ? "bg-[var(--status-success-text)]" : ch === "D" ? "bg-muted-foreground" : ch === "L" ? "bg-[var(--status-error-text)]" : "bg-muted"
                          )}
                        />
                      ))}
                    </div>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * ABE P0-1: Sanity check for descenso data validity
 */
function isDescensoValid(descenso: DescensoData | null | undefined): boolean {
  if (!descenso?.data) return false;
  if (descenso.data.length < 10) return false;
  if (descenso.data.some((d) => !d.team_name)) return false;
  return true;
}

/**
 * Reclasificación Table — accumulated Apertura + Clausura standings
 */
function ReclasificacionTable({
  data,
  onTeamSelect,
}: {
  data: ReclasificacionData;
  onTeamSelect?: (teamId: number) => void;
}) {
  if (!data.data || data.data.length === 0) {
    return <div className="text-sm text-muted-foreground text-center py-4">No data</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">#</th>
            <th className="text-left py-2 px-2 text-xs font-medium text-muted-foreground">Team</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">P</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">W</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">D</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">L</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">GD</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">Pts</th>
          </tr>
        </thead>
        <tbody>
          {data.data.map((entry) => (
            <tr
              key={entry.team_id}
              className={cn(
                "border-b border-border last:border-0 hover:bg-accent/50",
                entry.team_id > 0 && onTeamSelect && "cursor-pointer"
              )}
              onClick={entry.team_id > 0 && onTeamSelect ? () => onTeamSelect(entry.team_id) : undefined}
            >
              <td className="text-center py-1.5 px-2">
                <QualificationBadge value={entry.position} active={entry.position <= 8} />
              </td>
              <td className="py-1.5 px-2">
                <div className="flex items-center gap-2">
                  {entry.team_logo ? (
                    <Image src={entry.team_logo} alt="" width={16} height={16} className="rounded-full object-cover" />
                  ) : (
                    <Globe className="h-4 w-4 text-muted-foreground" />
                  )}
                  <span className="truncate text-foreground">{entry.team_name}</span>
                  {entry.zone && (
                    <span
                      className={cn(
                        "text-[10px] px-1 py-0.5 rounded shrink-0",
                        entry.zone.style === "blue" && "bg-blue-500/10 text-blue-400",
                        entry.zone.style === "orange" && "bg-orange-500/10 text-orange-400",
                        entry.zone.style === "cyan" && "bg-cyan-500/10 text-cyan-400",
                      )}
                    >
                      {entry.zone.type}
                    </span>
                  )}
                </div>
              </td>
              <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.played}</td>
              <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.won}</td>
              <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.drawn}</td>
              <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.lost}</td>
              <td
                className={cn(
                  "text-center py-1.5 px-2 font-medium",
                  entry.goal_diff > 0
                    ? "text-[var(--status-success-text)]"
                    : entry.goal_diff < 0
                      ? "text-[var(--status-error-text)]"
                      : "text-muted-foreground"
                )}
              >
                {entry.goal_diff > 0 ? `+${entry.goal_diff}` : entry.goal_diff}
              </td>
              <td className="text-center py-1.5 px-2 font-bold text-foreground">{entry.points}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Descenso Table — relegation risk by 3-year average
 */
function DescensoTable({
  data,
  onTeamSelect,
}: {
  data: DescensoData;
  onTeamSelect?: (teamId: number) => void;
}) {
  if (!data.data || data.data.length === 0) {
    return <div className="text-sm text-muted-foreground text-center py-4">No data</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">#</th>
            <th className="text-left py-2 px-2 text-xs font-medium text-muted-foreground">Team</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">Pts</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-8">P</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-14">Avg</th>
            <th className="text-center py-2 px-2 text-xs font-medium text-muted-foreground w-10">GD</th>
          </tr>
        </thead>
        <tbody>
          {[...data.data].reverse().map((entry) => (
            <tr
              key={entry.team_id}
              className={cn(
                "border-b border-border last:border-0 hover:bg-accent/50",
                entry.zone?.type === "relegation_risk" && "bg-red-500/5",
                entry.team_id > 0 && onTeamSelect && "cursor-pointer"
              )}
              onClick={entry.team_id > 0 && onTeamSelect ? () => onTeamSelect(entry.team_id) : undefined}
            >
              <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.position}</td>
              <td className="py-1.5 px-2">
                <div className="flex items-center gap-2">
                  {entry.team_logo ? (
                    <Image src={entry.team_logo} alt="" width={16} height={16} className="rounded-full object-cover" />
                  ) : (
                    <Globe className="h-4 w-4 text-muted-foreground" />
                  )}
                  <span className="truncate text-foreground">{entry.display_name || entry.team_name}</span>
                  {entry.zone?.type === "relegation_risk" && (
                    <span className="text-[10px] px-1 py-0.5 rounded bg-red-500/10 text-red-400 shrink-0">
                      Relegation Risk
                    </span>
                  )}
                </div>
              </td>
              <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.points}</td>
              <td className="text-center py-1.5 px-2 text-muted-foreground">{entry.played}</td>
              <td className="text-center py-1.5 px-2 font-bold text-foreground">{entry.average.toFixed(4)}</td>
              <td
                className={cn(
                  "text-center py-1.5 px-2 font-medium",
                  entry.goal_diff > 0
                    ? "text-[var(--status-success-text)]"
                    : entry.goal_diff < 0
                      ? "text-[var(--status-error-text)]"
                      : "text-muted-foreground"
                )}
              >
                {entry.goal_diff > 0 ? `+${entry.goal_diff}` : entry.goal_diff}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.seasons && (
        <div className="px-2 py-1.5 text-[10px] text-muted-foreground">
          Seasons: {data.seasons.join(", ")} &middot; Source: {data.source}
        </div>
      )}
    </div>
  );
}

/** Coverage ring — SVG donut showing a percentage */
function CoverageRing({ pct, size = 72 }: { pct: number; size?: number }) {
  const stroke = 5;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(pct, 100) / 100);

  // Color based on percentage
  const color =
    pct >= 85 ? "var(--status-success-text)"
    : pct >= 50 ? "var(--status-info-text)"
    : pct >= 25 ? "var(--status-warning-text)"
    : "var(--status-error-text)";

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="var(--border)" strokeWidth={stroke}
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color} strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-semibold text-foreground">{pct.toFixed(0)}%</span>
      </div>
    </div>
  );
}

/** Coverage detail content — fetches coverage map data for a specific league */
export function CoverageDrawerContent({ leagueId }: { leagueId: number }) {
  const { data } = useCoverageMap("current_season");
  const leagueCoverage = useMemo(() => {
    if (!data?.leagues) return [];
    return data.leagues.filter((l) => l.league_id === leagueId);
  }, [data, leagueId]);

  if (!leagueCoverage.length) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No coverage data available
      </p>
    );
  }
  return <CoverageDetailContent leagues={leagueCoverage} />;
}

/**
 * LeagueDetail Component
 *
 * Displays detailed league information:
 * - League info (name, country, kind, priority)
 * - Group info if part of a group
 * - Stats by season table
 * - TITAN coverage
 * - Recent matches
 * - Standings (with sub-tabs for reclasificación/descenso)
 */
export function LeagueDetail({ leagueId, onBack, onTeamSelect, onSettingsClick, onCoverageClick, initialTeamId, siblingLeagues, onLeagueChange, onPlayerSelect }: LeagueDetailProps) {
  const [standingsSubTab, _setStandingsSubTab] = useState<string>(() => {
    if (typeof window === "undefined") return "standings";
    return localStorage.getItem("fs:leagueSubTab") || "standings";
  });
  const setStandingsSubTab = useCallback((v: string) => {
    _setStandingsSubTab(v);
    try { localStorage.setItem("fs:leagueSubTab", v); } catch {}
  }, []);
  const [selectedGroup, setSelectedGroup] = useState<string | undefined>(undefined);
  const [selectedTeamId, setSelectedTeamId] = useState<number | null>(initialTeamId ?? null);
  const [teamTab, _setTeamTab] = useState<TeamDetailTabId>(() => {
    if (typeof window === "undefined") return "squad";
    const stored = localStorage.getItem("fs:teamTab");
    const valid = TEAM_DETAIL_TABS.some(t => t.id === stored);
    return valid ? (stored as TeamDetailTabId) : "squad";
  });
  const setTeamTab = useCallback((v: TeamDetailTabId) => {
    _setTeamTab(v);
    try { localStorage.setItem("fs:teamTab", v); } catch {}
  }, []);
  const [selectedSeason, setSelectedSeason] = useState<number | undefined>(undefined);
  // Reset selected team when league changes (let auto-select pick first team)
  useEffect(() => {
    setSelectedTeamId(initialTeamId ?? null);
    setStandingsSubTab("standings");
    setSelectedGroup(undefined);
    setSelectedSeason(undefined);
  }, [leagueId, initialTeamId]);
  const { data, isLoading, error, refetch } = useFootballLeague(leagueId);
  const { data: standingsData, isLoading: isStandingsLoading } = useStandings(leagueId, { group: selectedGroup, season: selectedSeason });

  // Auto-select first team in standings when data loads and no team is selected
  useEffect(() => {
    if (!selectedTeamId && standingsData?.standings?.length) {
      const firstTeam = standingsData.standings[0];
      if (firstTeam.teamId > 0) {
        setSelectedTeamId(firstTeam.teamId);
        onTeamSelect(firstTeam.teamId);
      }
    }
  }, [standingsData, selectedTeamId, onTeamSelect]);
  const selectedTeam = useFootballTeam(selectedTeamId);
  const selectedTeamSquad = useTeamSquad(selectedTeamId ?? 0, !!selectedTeamId);

  // Squad stats — shares TanStack cache key with TeamSquadStats (no duplicate fetch)
  const squadStatsQuery = useTeamSquadStats(selectedTeamId, selectedSeason ?? null, teamTab === "squad");

  // MUST be called on every render (Rules of Hooks). Keep defensive for loading/error.
  const nextMatches = useMemo(() => {
    const matches = data?.recent_matches ?? [];
    // Treat "no score" as upcoming. Status may vary; keep it defensive.
    return matches.filter((m) => !m.score);
  }, [data?.recent_matches]);

  // Live matches — polls every 30s, only enabled when leagueId is set
  const { data: liveMatches } = useLiveLeagueMatches(leagueId);
  const hasLive = (liveMatches?.length ?? 0) > 0;
  const { getLogoUrl } = useTeamLogos();

  // Only auto-switch when hasLive transitions: false→true (go to live), true→false (leave live)
  // Does NOT force the user back to live if they manually picked another tab.
  const prevHasLive = useRef(hasLive);
  useEffect(() => {
    if (hasLive && !prevHasLive.current) {
      setStandingsSubTab("live");
    } else if (!hasLive && prevHasLive.current && standingsSubTab === "live") {
      setStandingsSubTab("standings");
    }
    prevHasLive.current = hasLive;
  }, [hasLive]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTeamSelect = useCallback(
    (teamId: number) => {
      setSelectedTeamId(teamId);
      onTeamSelect(teamId);
    },
    [onTeamSelect]
  );

  // Keyboard navigation: arrow up/down moves selection in standings
  useEffect(() => {
    if (standingsSubTab !== "standings") return;
    const teams = standingsData?.standings;
    if (!teams?.length) return;

    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      // Don't hijack if user is typing in an input/textarea/select
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      e.preventDefault();
      const currentIdx = teams!.findIndex((t) => t.teamId === selectedTeamId);
      let nextIdx: number;
      if (e.key === "ArrowDown") {
        nextIdx = currentIdx < teams!.length - 1 ? currentIdx + 1 : 0;
      } else {
        nextIdx = currentIdx > 0 ? currentIdx - 1 : teams!.length - 1;
      }
      const next = teams![nextIdx];
      if (next.teamId > 0) {
        handleTeamSelect(next.teamId);
        // Scroll the row into view
        const row = document.querySelector(`tr[data-team-id="${next.teamId}"]`);
        row?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [standingsSubTab, standingsData?.standings, selectedTeamId, handleTeamSelect]);

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
              League Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || `Unable to fetch league ${leagueId}`}
            </p>
          </div>
          <div className="flex gap-2">
            <Button onClick={onBack} variant="outline">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button onClick={() => refetch()} variant="secondary">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const { league, stats_by_season, recent_matches } = data;

  return (
    <ScrollArea data-dev-ref="LeagueDetail" className="h-full">
      <div className="px-6 pt-6 pb-2">
        {/* Two-column content: league + team */}
        <div className="w-full flex flex-col lg:flex-row gap-4 lg:items-start min-w-0">
          {/* League column */}
          <div
            className={cn(
              "flex flex-col gap-4 w-full min-w-0",
              selectedTeamId !== null ? "lg:w-[50%]" : "lg:max-w-[50%] lg:mr-auto"
            )}
          >
            {/* League header */}
            <div data-dev-ref="LeagueDetail:LeagueHeader" className="rounded-lg border border-border px-4 py-3 flex items-center gap-4 min-h-[148px]">
              {league.logo_url && (
                <img
                  src={league.logo_url}
                  alt=""
                  className="h-24 w-24 object-contain shrink-0"
                />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  {siblingLeagues && siblingLeagues.length > 1 && onLeagueChange ? (
                    <Select
                      value={leagueId.toString()}
                      onValueChange={(v) => onLeagueChange(parseInt(v, 10))}
                    >
                      <SelectTrigger className="text-2xl font-semibold text-foreground border-none shadow-none px-1 h-auto py-0 gap-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {siblingLeagues.map((sl) => (
                          <SelectItem key={sl.league_id} value={sl.league_id.toString()}>
                            {sl.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <h1 className="text-2xl font-semibold text-foreground cursor-help truncate">
                          {league.name}
                        </h1>
                      </TooltipTrigger>
                      <TooltipContent side="bottom" sideOffset={6}>
                        <div className="space-y-1">
                          <div className="text-foreground">
                            <span className="text-muted-foreground">Match Weight:</span>{" "}
                            <span className="font-medium text-foreground">
                              {league.match_weight ?? 1}
                            </span>
                          </div>
                          <div className="text-muted-foreground">
                            1 = peso estándar. Valores mayores aumentan la influencia de esta liga
                            en cálculos internos; menores la reducen.
                          </div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  )}
                  {onSettingsClick && (
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={onSettingsClick}
                      className="shrink-0 h-8 w-8"
                      aria-label="League settings"
                    >
                      <Settings className="h-4 w-4" />
                    </Button>
                  )}
                </div>
                <div className="flex items-center gap-3 mt-1 text-sm text-muted-foreground flex-wrap">
                  <span className="flex items-center gap-1">
                    <CountryFlag country={league.country} size={14} />
                    {league.country}
                  </span>
                  {league.kind && <span className="capitalize">{league.kind}</span>}
                  {league.priority && (
                    <span className="px-1.5 py-0.5 bg-muted rounded text-xs">
                      {league.priority}
                    </span>
                  )}
                </div>
                {/* Season coverage badges */}
                {stats_by_season && stats_by_season.length > 0 && (
                  <div className="flex items-center gap-1 mt-1.5 flex-wrap">
                    {[...stats_by_season].reverse().map((s) => {
                      const avg = ((s.with_stats_pct ?? 0) + (s.with_odds_pct ?? 0)) / 2;
                      const isActive = selectedSeason === s.season;
                      const isCurrent = selectedSeason === undefined && s.season === stats_by_season[0].season;
                      const color = isActive || isCurrent
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground";
                      return (
                        <Tooltip key={s.season}>
                          <TooltipTrigger asChild>
                            <button
                              onClick={() => {
                                const newSeason = (isActive || isCurrent) ? undefined : s.season;
                                setSelectedSeason(newSeason);
                                setStandingsSubTab("standings");
                              }}
                              className={cn(
                                "px-1.5 py-0.5 rounded text-[10px] font-medium tabular-nums transition-colors",
                                "hover:ring-1 hover:ring-primary/50",
                                color
                              )}
                            >
                              {s.season}
                            </button>
                          </TooltipTrigger>
                          <TooltipContent side="bottom" className="text-xs">
                            <div className="space-y-0.5">
                              <div>{s.finished}/{s.total_matches} matches</div>
                              <div>Stats {s.with_stats_pct?.toFixed(0) ?? 0}% · Odds {s.with_odds_pct?.toFixed(0) ?? 0}%</div>
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      );
                    })}
                  </div>
                )}
              </div>
              {(() => {
                const current = stats_by_season?.[0];
                if (!current) return null;
                const stats = current.with_stats_pct ?? 0;
                const odds = current.with_odds_pct ?? 0;
                const pct = (stats + odds) / 2;
                const ring = (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      {onCoverageClick ? (
                        <button
                          onClick={onCoverageClick}
                          className="cursor-pointer hover:opacity-80 transition-opacity"
                        >
                          <CoverageRing pct={pct} />
                        </button>
                      ) : (
                        <div><CoverageRing pct={pct} /></div>
                      )}
                    </TooltipTrigger>
                    <TooltipContent side="bottom">Coverage</TooltipContent>
                  </Tooltip>
                );
                return ring;
              })()}
            </div>

            {/* League content */}
            <div data-dev-ref="LeagueDetail:LeagueContent" className="rounded-lg border border-border overflow-hidden">
              <div>
                {/* Sub-tabs: Live + Group/Standings + Reclassification + Relegation */}
                {(() => {
                  const tables = standingsData?.meta?.available_tables ?? [];
                  const navigableGroups = tables.filter(
                    (t: AvailableTable) => t.type === "regular" || t.type === "group_stage" || t.type === "playoff"
                  );
                  const shortLabel = (name: string) => {
                    const match = name.match(/Group\s+\w+$/i);
                    return match ? match[0] : name;
                  };
                  const hasVirtualTabs = !!(standingsData?.reclasificacion || isDescensoValid(standingsData?.descenso ?? null));
                  const showGroupPills = navigableGroups.length > 1;
                  return (
                    <div className="flex gap-1 border-b border-border overflow-x-auto">
                      {/* Live tab — only when in-play */}
                      {hasLive && (
                        <button
                          onClick={() => setStandingsSubTab("live")}
                          className={cn(
                            "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors inline-flex items-center gap-1.5",
                            standingsSubTab === "live"
                              ? "text-success border-b-2 border-success"
                              : "text-success/70 hover:text-success"
                          )}
                        >
                          <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-success" />
                          </span>
                          Live
                        </button>
                      )}
                      {/* Group/table pills */}
                      {showGroupPills ? (
                        <>
                          {navigableGroups.map((t: AvailableTable) => {
                            const isActive = standingsSubTab === "standings" && (
                              selectedGroup === t.group || (!selectedGroup && t.is_current)
                            );
                            return (
                              <button
                                key={t.group}
                                onClick={() => {
                                  setSelectedGroup(t.is_current ? undefined : t.group);
                                  setStandingsSubTab("standings");
                                }}
                                className={cn(
                                  "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                                  isActive
                                    ? "text-foreground border-b-2 border-primary"
                                    : "text-muted-foreground hover:text-foreground"
                                )}
                              >
                                {shortLabel(t.group)}
                              </button>
                            );
                          })}
                        </>
                      ) : (
                        <button
                          onClick={() => setStandingsSubTab("standings")}
                          className={cn(
                            "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                            standingsSubTab === "standings"
                              ? "text-foreground border-b-2 border-primary"
                              : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Standings
                        </button>
                      )}
                      {/* Virtual table sub-tabs */}
                      {standingsData?.reclasificacion && (
                        <button
                          onClick={() => setStandingsSubTab("reclasificacion")}
                          className={cn(
                            "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                            standingsSubTab === "reclasificacion"
                              ? "text-foreground border-b-2 border-primary"
                              : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Reclassification
                        </button>
                      )}
                      {isDescensoValid(standingsData?.descenso ?? null) && (
                        <button
                          onClick={() => setStandingsSubTab("descenso")}
                          className={cn(
                            "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                            standingsSubTab === "descenso"
                              ? "text-foreground border-b-2 border-primary"
                              : "text-muted-foreground hover:text-foreground"
                          )}
                        >
                          Relegation
                        </button>
                      )}
                      {/* Next Matches */}
                      <button
                        onClick={() => setStandingsSubTab("next")}
                        className={cn(
                          "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                          standingsSubTab === "next"
                            ? "text-foreground border-b-2 border-primary"
                            : "text-muted-foreground hover:text-foreground"
                        )}
                      >
                        Next
                      </button>
                      {/* Stats by Season */}
                      <button
                        onClick={() => setStandingsSubTab("stats")}
                        className={cn(
                          "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                          standingsSubTab === "stats"
                            ? "text-foreground border-b-2 border-primary"
                            : "text-muted-foreground hover:text-foreground"
                        )}
                      >
                        Stats
                      </button>
                    </div>
                  );
                })()}

                {/* Sub-tab content */}
                {standingsSubTab === "live" && hasLive ? (
                  <div>
                    {liveMatches!.map((m: LiveMatch, idx: number) => {
                      const isHT = m.status === "HT";
                      const elapsed = m.elapsedExtra
                        ? `${m.elapsed}+${m.elapsedExtra}'`
                        : m.elapsed != null
                          ? `${m.elapsed}'`
                          : m.status;
                      const isLast = idx === liveMatches!.length - 1;
                      return (
                        <div key={m.id}>
                          <div className="px-3 py-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-[18px]">
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <div className="flex items-center justify-center">
                                      <TeamLogo src={getLogoUrl(m.home)} teamName={m.home} size={20} />
                                    </div>
                                  </TooltipTrigger>
                                  <TooltipContent side="bottom"><p>{m.home}</p></TooltipContent>
                                </Tooltip>
                                <span className="text-lg font-bold text-foreground font-condensed tabular-nums">
                                  {m.homeScore}
                                </span>
                              </div>
                              <div className="flex flex-col items-center gap-0.5">
                                <span className="text-xs font-medium text-muted-foreground tabular-nums">
                                  {elapsed}
                                </span>
                                <div className={cn(
                                  "w-8 h-0.5 animate-pulse",
                                  isHT ? "bg-[var(--status-warning-text)]" : "bg-success"
                                )} />
                              </div>
                              <div className="flex items-center gap-[18px]">
                                <span className="text-lg font-bold text-foreground font-condensed tabular-nums">
                                  {m.awayScore}
                                </span>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <div className="flex items-center justify-center">
                                      <TeamLogo src={getLogoUrl(m.away)} teamName={m.away} size={20} />
                                    </div>
                                  </TooltipTrigger>
                                  <TooltipContent side="bottom"><p>{m.away}</p></TooltipContent>
                                </Tooltip>
                              </div>
                            </div>
                          </div>
                          {!isLast && (
                            <div className="flex justify-center">
                              <div className="w-[65%] h-px bg-border" />
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : isStandingsLoading ? (
                  <div className="flex items-center justify-center py-6">
                    <Loader size="sm" />
                  </div>
                ) : standingsSubTab === "reclasificacion" && standingsData?.reclasificacion ? (
                  <ReclasificacionTable
                    data={standingsData.reclasificacion}
                    onTeamSelect={handleTeamSelect}
                  />
                ) : standingsSubTab === "descenso" && standingsData?.descenso ? (
                  <DescensoTable
                    data={standingsData.descenso}
                    onTeamSelect={handleTeamSelect}
                  />
                ) : standingsSubTab === "next" ? (
                  <RecentMatchesList
                    matches={nextMatches}
                    onTeamSelect={handleTeamSelect}
                    useShortNames={league.tags?.use_short_names ?? false}
                  />
                ) : standingsSubTab === "stats" ? (
                  <StatsTable stats={stats_by_season} />
                ) : standingsData && standingsData.standings.length > 0 ? (
                  <StandingsTable
                    standings={standingsData.standings}
                    onTeamSelect={handleTeamSelect}
                    useShortNames={league.tags?.use_short_names ?? false}
                    selectedTeamId={selectedTeamId}
                  />
                ) : (
                  <div className="px-4 py-4 text-sm text-muted-foreground">
                    No standings available for this league
                  </div>
                )}
              </div>

          </div>
          </div>

          {/* Club container (only when selected) */}
          {selectedTeamId !== null && (
            <div data-dev-ref="LeagueDetail:TeamColumn" className="flex flex-col gap-4 w-full lg:w-[50%] min-w-0">
              {/* Team Header */}
              <div data-dev-ref="LeagueDetail:TeamHeader" className="rounded-lg border border-border px-4 py-3 flex items-center gap-3 min-h-[148px]">
                {selectedTeam.data?.team?.logo_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={selectedTeam.data.team.logo_url}
                    alt=""
                    className="h-24 w-24 object-contain"
                  />
                ) : (
                  <div className="h-24 w-24 rounded-full bg-muted" />
                )}
                <div className="min-w-0 flex-1">
                  <h2 className="text-2xl font-semibold text-foreground truncate">
                    {selectedTeam.data?.team?.display_name ?? selectedTeam.data?.team?.name ?? "Team details"}
                  </h2>
                  {(selectedTeam.data?.wikidata_enrichment?.city || selectedTeam.data?.team?.country) && (
                    <div className="flex items-center gap-1.5 text-sm text-muted-foreground mt-1">
                      <MapPin className="h-3.5 w-3.5" />
                      <span>
                        {[selectedTeam.data?.wikidata_enrichment?.city, selectedTeam.data?.team?.country]
                          .filter(Boolean)
                          .join(", ")}
                      </span>
                    </div>
                  )}
                  {selectedTeam.data?.wikidata_enrichment?.stadium_name && (
                    <div className="flex items-center gap-1.5 text-sm text-muted-foreground mt-0.5">
                      <StadiumIcon className="h-3.5 w-3.5" />
                      <span>
                        {selectedTeam.data.wikidata_enrichment.stadium_name}
                        {selectedTeam.data.wikidata_enrichment.stadium_capacity && (
                          <span className="ml-1">
                            ({selectedTeam.data.wikidata_enrichment.stadium_capacity.toLocaleString()})
                          </span>
                        )}
                      </span>
                    </div>
                  )}
                  {selectedTeamSquad.data?.current_manager && (
                    <div className="flex items-center gap-1.5 text-sm text-muted-foreground mt-0.5">
                      <ShieldUser className="h-3.5 w-3.5" />
                      <span>{selectedTeamSquad.data.current_manager.name}</span>
                    </div>
                  )}
                </div>
                {selectedTeam.isLoading && <Loader size="sm" />}
                {/* Coverage ring — click navigates to Coverage tab */}
                {(() => {
                  const sources = selectedTeam.data?.feature_coverage?.sources;
                  if (!sources) return null;
                  const pcts = [sources.odds.pct, sources.xg.pct, sources.lineup.pct, sources.xi_depth.pct, sources.form.pct, sources.h2h.pct];
                  const avg = pcts.reduce((a, b) => a + b, 0) / pcts.length;
                  return (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          onClick={() => setTeamTab("coverage")}
                          className="shrink-0 hover:opacity-80 transition-opacity"
                        >
                          <CoverageRing pct={avg} />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent side="bottom">Coverage</TooltipContent>
                    </Tooltip>
                  );
                })()}
              </div>

              {/* Team Tabs + Content */}
              <div data-dev-ref="LeagueDetail:TeamTabs" className="rounded-lg border border-border overflow-hidden">
              <div className="flex gap-1 border-b border-border overflow-x-auto">
                {TEAM_DETAIL_TABS.map((tab) => {
                  const isActive = teamTab === tab.id;

                  return (
                    <button
                      key={tab.id}
                      onClick={() => setTeamTab(tab.id)}
                      className={cn(
                        "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                        isActive
                          ? "text-foreground border-b-2 border-primary"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {/* Tab Content */}
              {teamTab === "club" && (
                <div data-dev-ref="LeagueDetail:TeamOverviewTab">
                  {!selectedTeam.isLoading && selectedTeam.data?.wikidata_enrichment ? (
                    <div>
                      <table className="w-full text-sm">
                        <tbody className="divide-y divide-border">
                          {[
                            ["Team ID", selectedTeamId?.toString()],
                            ["Full name", selectedTeam.data.wikidata_enrichment.full_name],
                            ["Short name", selectedTeam.data.wikidata_enrichment.short_name],
                            ["City", selectedTeam.data.wikidata_enrichment.city],
                            ["Stadium", selectedTeam.data.wikidata_enrichment.stadium_name],
                            [
                              "Capacity",
                              selectedTeam.data.wikidata_enrichment.stadium_capacity != null
                                ? selectedTeam.data.wikidata_enrichment.stadium_capacity.toLocaleString()
                                : null,
                            ],
                            [
                              "Altitude (m)",
                              selectedTeam.data.wikidata_enrichment.stadium_altitude_m != null
                                ? selectedTeam.data.wikidata_enrichment.stadium_altitude_m.toLocaleString()
                                : null,
                            ],
                            [
                              "Coords",
                              selectedTeam.data.wikidata_enrichment.lat != null &&
                              selectedTeam.data.wikidata_enrichment.lon != null
                                ? `${selectedTeam.data.wikidata_enrichment.lat.toFixed(6)}, ${selectedTeam.data.wikidata_enrichment.lon.toFixed(6)}`
                                : null,
                            ],
                          ].map(([label, value]) => (
                            <tr key={label}>
                              <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                                {label}
                              </td>
                              <td className="py-2 px-3 text-sm text-foreground">
                                {value ? (
                                  label === "Team ID" ? (
                                    <button
                                      onClick={() => {
                                        navigator.clipboard.writeText(value);
                                        toast.success("Team ID copied");
                                      }}
                                      className="text-primary hover:opacity-80 transition-opacity"
                                    >
                                      {value}
                                    </button>
                                  ) : label === "Coords" ? (
                                    <a
                                      href={`https://www.google.com/maps?q=${value}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-primary hover:opacity-80 transition-opacity"
                                    >
                                      {value}
                                    </a>
                                  ) : (
                                    <span className="break-words">{value}</span>
                                  )
                                ) : (
                                  <span className="text-muted-foreground">-</span>
                                )}
                              </td>
                            </tr>
                          ))}
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Website
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.website ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={selectedTeam.data.wikidata_enrichment.website}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    {selectedTeam.data.wikidata_enrichment.website}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              X.com (Twitter)
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.twitter ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={`https://x.com/${selectedTeam.data.wikidata_enrichment.twitter}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    @{selectedTeam.data.wikidata_enrichment.twitter}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Instagram
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.instagram ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={`https://www.instagram.com/${selectedTeam.data.wikidata_enrichment.instagram}/`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    @{selectedTeam.data.wikidata_enrichment.instagram}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Wikipedia
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.team.wiki?.wiki_url_cached ||
                              selectedTeam.data.team.wiki?.wiki_url ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={
                                      selectedTeam.data.team.wiki?.wiki_url_cached ||
                                      selectedTeam.data.team.wiki?.wiki_url ||
                                      "#"
                                    }
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    {selectedTeam.data.team.wiki?.wiki_url_cached ||
                                      selectedTeam.data.team.wiki?.wiki_url}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Wikidata ID
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.wikidata_id ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={`https://www.wikidata.org/wiki/${selectedTeam.data.wikidata_enrichment.wikidata_id}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    {selectedTeam.data.wikidata_enrichment.wikidata_id}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Stadium Wikidata
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.stadium_wikidata_id ? (
                                <Button variant="link" size="sm" className="px-0 h-auto" asChild>
                                  <a
                                    href={`https://www.wikidata.org/wiki/${selectedTeam.data.wikidata_enrichment.stadium_wikidata_id}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    {selectedTeam.data.wikidata_enrichment.stadium_wikidata_id}
                                  </a>
                                </Button>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                          <tr>
                            <td className="py-2 px-3 text-[11px] text-muted-foreground whitespace-nowrap">
                              Updated
                            </td>
                            <td className="py-2 px-3 text-sm text-foreground">
                              {selectedTeam.data.wikidata_enrichment.wikidata_updated_at ? (
                                <span>
                                  {new Date(selectedTeam.data.wikidata_enrichment.wikidata_updated_at).toLocaleDateString(undefined, {
                                    year: "numeric",
                                    month: "short",
                                    day: "numeric",
                                  })}
                                </span>
                              ) : (
                                <span className="text-muted-foreground">-</span>
                              )}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  ) : !selectedTeam.isLoading ? (
                    <div className="px-4 py-4 text-sm text-muted-foreground">
                      No enrichment data for this team
                    </div>
                  ) : null}
                </div>
              )}

              {teamTab === "squad" && selectedTeamId && (
                <TeamSquadStats teamId={selectedTeamId} season={selectedSeason ?? null} onPlayerSelect={onPlayerSelect} />
              )}

              {teamTab === "matches" && (
                <div className="px-4 py-8 text-center">
                  <Calendar className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Matches coming soon</p>
                </div>
              )}

              {teamTab === "stats" && selectedTeamId && (
                <TeamPerformanceCharts teamId={selectedTeamId} leagueId={leagueId} season={selectedSeason ?? null} />
              )}

              {teamTab === "coverage" && (
                <div data-dev-ref="LeagueDetail:TeamCoverageTab" className="space-y-4 min-w-0 overflow-hidden p-4">
                  {selectedTeam.isLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader size="sm" />
                    </div>
                  ) : selectedTeam.data?.feature_coverage ? (
                    <FeatureCoverageSection coverage={selectedTeam.data.feature_coverage} />
                  ) : (
                    <div className="px-4 py-8 text-center">
                      <p className="text-sm text-muted-foreground">No coverage data available</p>
                    </div>
                  )}
                </div>
              )}

              {teamTab === "transfers" && (
                <div className="px-4 py-8 text-center">
                  <TrendingUp className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Transfers coming soon</p>
                </div>
              )}
              </div>
            </div>
          )}
        </div>

      </div>
    </ScrollArea>

  );
}
