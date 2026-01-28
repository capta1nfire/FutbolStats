"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { Calendar, Clock } from "lucide-react";
import { useUpcomingMatches } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";

interface OverviewDrawerUpcomingProps {
  tab: OverviewTab;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars -- tab prop reserved for future use
export function OverviewDrawerUpcoming({ tab }: OverviewDrawerUpcomingProps) {
  // Only summary tab for now
  return <UpcomingSummaryTab />;
}

function UpcomingSummaryTab() {
  const { matches, isDegraded, isLoading } = useUpcomingMatches();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!matches || isDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Upcoming matches data unavailable
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Calendar className="h-4 w-4 text-primary" />
        <span>Upcoming Matches ({matches.length})</span>
      </div>

      <div className="space-y-2">
        {matches.slice(0, 10).map((match) => (
          <div
            key={match.id}
            className="bg-muted/30 rounded-lg p-3 space-y-1"
          >
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-foreground truncate max-w-[60%]">
                {match.home} vs {match.away}
              </span>
              <span className="text-xs text-muted-foreground flex items-center gap-1 shrink-0">
                <Clock className="h-3 w-3" />
                {new Date(match.kickoffISO).toLocaleString(undefined, {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{match.leagueName}</span>
              {match.hasPrediction && (
                <span className="px-1.5 py-0.5 bg-[var(--status-success-bg)] text-[var(--status-success-text)] rounded text-[10px]">
                  Pred
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {matches.length > 10 && (
        <p className="text-xs text-muted-foreground text-center">
          +{matches.length - 10} more matches
        </p>
      )}
    </div>
  );
}
