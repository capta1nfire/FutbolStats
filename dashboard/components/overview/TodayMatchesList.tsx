"use client";

import { useMemo } from "react";
import { TodayMatchItem } from "@/lib/hooks/use-today-matches";
import { useTeamLogos } from "@/lib/hooks";
import { useOverviewDrawer } from "@/lib/hooks/use-overview-drawer";
import { cn } from "@/lib/utils";
import { Calendar, AlertCircle, Play } from "lucide-react";
import { TeamLogo } from "@/components/ui/team-logo";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface TodayMatchesListProps {
  matches: TodayMatchItem[];
  className?: string;
}

const LIVE_STATUSES = ["LIVE", "1H", "2H", "HT"];
const FINISHED_STATUSES = ["FT", "AET", "PEN"];

function formatKickoff(isoDate: string): string {
  const date = new Date(isoDate);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatElapsed(elapsed?: number, elapsedExtra?: number): string {
  if (elapsed == null) return "";
  if (elapsedExtra && elapsedExtra > 0) {
    return `${elapsed}+${elapsedExtra}'`;
  }
  return `${elapsed}'`;
}

function getStatusBadge(status: string) {
  const s = status?.toUpperCase();
  switch (s) {
    case "LIVE":
    case "1H":
    case "2H":
      return null;
    case "HT":
      return null;
    case "FT":
    case "AET":
    case "PEN":
      return (
        <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground">
          {s}
        </span>
      );
    case "PST":
    case "CANC":
    case "ABD":
      return (
        <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-muted text-muted-foreground">
          {s}
        </span>
      );
    case "NS":
    case "TBD":
    default:
      return null;
  }
}

interface LeagueGroup {
  leagueName: string;
  matches: TodayMatchItem[];
  hasLive: boolean;
}

export function TodayMatchesList({
  matches,
  className,
}: TodayMatchesListProps) {
  const { getLogoUrl } = useTeamLogos();
  const { openDrawer } = useOverviewDrawer();

  // Group matches by league and sort
  const leagueGroups = useMemo(() => {
    // First sort all matches: live first, then by kickoff
    const sorted = [...matches].sort((a, b) => {
      const aLive = LIVE_STATUSES.includes(a.status?.toUpperCase() || "");
      const bLive = LIVE_STATUSES.includes(b.status?.toUpperCase() || "");
      if (aLive && !bLive) return -1;
      if (!aLive && bLive) return 1;
      return new Date(a.kickoffISO).getTime() - new Date(b.kickoffISO).getTime();
    });

    // Group by league
    const groups = new Map<string, TodayMatchItem[]>();
    for (const match of sorted) {
      const league = match.leagueName || "Unknown";
      if (!groups.has(league)) {
        groups.set(league, []);
      }
      groups.get(league)!.push(match);
    }

    // Convert to array and sort leagues: those with live matches first
    const result: LeagueGroup[] = [];
    for (const [leagueName, leagueMatches] of groups) {
      const hasLive = leagueMatches.some((m) =>
        LIVE_STATUSES.includes(m.status?.toUpperCase() || "")
      );
      result.push({ leagueName, matches: leagueMatches, hasLive });
    }

    // Sort: leagues with live matches first, then alphabetically
    result.sort((a, b) => {
      if (a.hasLive && !b.hasLive) return -1;
      if (!a.hasLive && b.hasLive) return 1;
      return a.leagueName.localeCompare(b.leagueName);
    });

    return result;
  }, [matches]);

  const handleMatchClick = (matchId: number) => {
    openDrawer({ panel: "match", matchId });
  };

  if (matches.length === 0) {
    return (
      <div className={cn("text-center py-8", className)}>
        <Calendar className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No matches today</p>
      </div>
    );
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className={cn("space-y-2", className)}>
        {leagueGroups.map((group) => (
          <div
            key={group.leagueName}
            className={cn(
              "rounded-lg bg-background border border-border overflow-hidden",
              group.hasLive && "border-success/50"
            )}
          >
            {/* League header */}
            <div
              className={cn(
                "px-3 py-1.5 flex items-center gap-2",
                group.hasLive && "bg-gradient-to-b from-success/20 to-transparent"
              )}
            >
              {group.hasLive && (
                <Play className="h-3 w-3 text-success" />
              )}
              <span className="text-xs font-medium text-muted-foreground truncate">
                {group.leagueName}
              </span>
              <span className="text-[10px] text-muted-foreground/70">
                ({group.matches.length})
              </span>
            </div>

            {/* Matches */}
            <div>
              {group.matches.map((match, idx) => {
                const statusUpper = match.status?.toUpperCase() || "";
                const isLive = LIVE_STATUSES.includes(statusUpper);
                const isHT = statusUpper === "HT";
                const isFinished = FINISHED_STATUSES.includes(statusUpper);
                const hasScore =
                  match.homeScore != null && match.awayScore != null;
                const isLast = idx === group.matches.length - 1;

                return (
                  <div key={match.id}>
                    <button
                      type="button"
                      onClick={() => handleMatchClick(match.id)}
                      className="w-full text-left px-3 py-2 hover:bg-tile transition-colors cursor-pointer"
                    >
                    {/* Teams row with logos */}
                    <div className="flex items-center justify-between">
                      {/* Home team + score */}
                      <div className="flex items-center gap-[18px]">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-center">
                              <TeamLogo
                                src={getLogoUrl(match.home)}
                                teamName={match.home}
                                size={20}
                              />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent side="bottom">
                            <p>{match.home}</p>
                          </TooltipContent>
                        </Tooltip>
                        {hasScore && (
                          <span className="text-lg font-bold text-foreground font-condensed tabular-nums">
                            {match.homeScore}
                          </span>
                        )}
                      </div>

                      {/* Center: elapsed time / kickoff + status */}
                      <div className="flex flex-col items-center gap-0.5">
                        {isLive ? (
                          <>
                            {match.elapsed != null && (
                              <span className="text-xs font-medium text-muted-foreground tabular-nums">
                                {formatElapsed(match.elapsed, match.elapsedExtra)}
                              </span>
                            )}
                            {isHT ? (
                              <div className="w-8 h-0.5 bg-[var(--status-warning-text)] animate-pulse" />
                            ) : (
                              <div className="w-8 h-0.5 bg-success animate-pulse" />
                            )}
                          </>
                        ) : hasScore ? (
                          <span className="text-xs font-medium text-muted-foreground tabular-nums">
                            {getStatusBadge(match.status || "")}
                          </span>
                        ) : (
                          <span className="text-xs font-medium text-muted-foreground tabular-nums">
                            {formatKickoff(match.kickoffISO)}
                          </span>
                        )}
                        {!hasScore && (
                          <div className="flex items-center gap-1">
                            {getStatusBadge(match.status || "")}
                            {!isFinished && !match.hasPrediction && (
                              <AlertCircle
                                className="h-3 w-3 text-[var(--status-warning-text)]"
                                aria-label="Missing prediction"
                              />
                            )}
                          </div>
                        )}
                      </div>

                      {/* Away team + score */}
                      <div className="flex items-center gap-[18px]">
                        {hasScore && (
                          <span className="text-lg font-bold text-foreground font-condensed tabular-nums">
                            {match.awayScore}
                          </span>
                        )}
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-center">
                              <TeamLogo
                                src={getLogoUrl(match.away)}
                                teamName={match.away}
                                size={20}
                              />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent side="bottom">
                            <p>{match.away}</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </div>
                    </button>
                    {!isLast && (
                      <div className="flex justify-center">
                        <div className="w-[65%] h-px bg-border" />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </TooltipProvider>
  );
}
