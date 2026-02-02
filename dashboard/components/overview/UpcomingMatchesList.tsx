"use client";

import { UpcomingMatch } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Calendar, CheckCircle2, AlertCircle } from "lucide-react";
import Link from "next/link";

interface UpcomingMatchesListProps {
  matches: UpcomingMatch[];
  className?: string;
}

function formatKickoff(isoDate: string): string {
  const date = new Date(isoDate);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function UpcomingMatchesList({
  matches,
  className,
}: UpcomingMatchesListProps) {
  if (matches.length === 0) {
    return (
      <div className={cn("text-center py-8", className)}>
        <Calendar className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No upcoming matches</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {matches.map((match) => (
        <Link
          key={match.id}
          href={`/matches?id=${match.id}`}
          className="block p-3 rounded-lg bg-background hover:bg-tile border border-border transition-colors"
        >
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Calendar className="h-3 w-3" />
              <span>{formatKickoff(match.kickoffISO)}</span>
              <span>·</span>
              <span>{match.leagueName}</span>
            </div>

            {match.hasPrediction ? (
              <CheckCircle2
                className="h-4 w-4 text-[var(--status-success-text)]"
                aria-label="Has prediction"
              />
            ) : (
              <AlertCircle
                className="h-4 w-4 text-[var(--status-warning-text)]"
                aria-label="Missing prediction"
              />
            )}
          </div>

          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">
              {match.home}
            </span>
            <span className="text-xs text-muted-foreground">vs</span>
            <span className="text-sm font-medium text-foreground text-right">
              {match.away}
            </span>
          </div>
        </Link>
      ))}
    </div>
  );
}
