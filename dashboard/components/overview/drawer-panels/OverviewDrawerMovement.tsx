"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { AlertCircle, ArrowUpDown, Users, TrendingUp } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";

interface OverviewDrawerMovementProps {
  tab: OverviewTab;
}

/**
 * Movement panel content for overview drawer
 *
 * Tabs:
 * - summary: Counts from ops/rollup
 * - movers: Top movers list (from /api/movement/top)
 */
export function OverviewDrawerMovement({ tab }: OverviewDrawerMovementProps) {
  if (tab === "movers") {
    return <MovementMoversTab />;
  }

  return <MovementSummaryTab />;
}

/**
 * Summary tab - uses existing ops data
 */
function MovementSummaryTab() {
  const { movement, isMovementDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!movement || isMovementDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Movement data unavailable
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ArrowUpDown className="h-4 w-4" />
        <span>Movement detected in last 24h</span>
      </div>

      {/* Lineup Movement */}
      <div className="bg-muted/30 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-2">
          <Users className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium text-foreground">Lineup Changes</span>
        </div>
        <div className="text-3xl font-bold text-foreground tabular-nums">
          {movement.lineup_movement_24h}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Teams with lineup updates
        </p>
      </div>

      {/* Market Movement */}
      <div className="bg-muted/30 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-2">
          <TrendingUp className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium text-foreground">Market Movement</span>
        </div>
        <div className="text-3xl font-bold text-foreground tabular-nums">
          {movement.market_movement_24h}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Matches with odds changes
        </p>
      </div>
    </div>
  );
}

/**
 * Movers tab - top movers from /api/movement/top
 *
 * HOLD: Backend contract pending confirmation.
 * ABE spec: movers[] with value:number (magnitude) + type=lineup|market
 * Master changed semántica to "recent activity" with value→captured_at
 * Awaiting final shape confirmation.
 */
function MovementMoversTab() {
  return (
    <div className="p-4">
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <AlertCircle className="h-8 w-8 text-orange-400 mb-2" />
        <p className="text-sm text-orange-400 font-medium">
          Awaiting backend contract
        </p>
        <p className="text-xs text-muted-foreground mt-1 max-w-[280px]">
          Top movers API shape pending confirmation. Check with backend team for final contract.
        </p>
        <span className="mt-3 px-2 py-1 text-[10px] bg-orange-500/10 text-orange-400 rounded border border-orange-500/30">
          HOLD
        </span>
      </div>
    </div>
  );
}
