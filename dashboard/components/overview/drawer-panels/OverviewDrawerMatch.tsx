"use client";

import { useMatchApi } from "@/lib/hooks/use-matches";
import { useOverviewDrawer } from "@/lib/hooks/use-overview-drawer";
import { OverviewTab } from "@/lib/overview-drawer";
import { Loader } from "@/components/ui/loader";
import { MatchDetailContent } from "@/components/matches/MatchDetailContent";
import { AlertCircle } from "lucide-react";

interface OverviewDrawerMatchProps {
  tab: OverviewTab;
}

/**
 * Match Detail Panel for Overview Drawer
 *
 * Shows full match details including:
 * - Match header with teams and score
 * - Tabs: Overview, Predictions, Standings
 *
 * Uses matchId from URL params to fetch match data.
 */
export function OverviewDrawerMatch({ tab }: OverviewDrawerMatchProps) {
  const { matchId } = useOverviewDrawer();
  const { match, isLoading, isDegraded } = useMatchApi(matchId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader size="md" />
      </div>
    );
  }

  if (!matchId) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No match selected</p>
      </div>
    );
  }

  if (isDegraded || !match) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="h-8 w-8 mx-auto mb-2 text-[var(--status-warning-text)]" />
        <p className="text-sm text-muted-foreground">
          Unable to load match {matchId}
        </p>
      </div>
    );
  }

  return <MatchDetailContent match={match} />;
}
