"use client";

import { useState } from "react";
import { MatchSummary } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { IconTabs } from "@/components/ui/icon-tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatusDot } from "./StatusDot";
import { Calendar, TrendingUp, Radio, Info } from "lucide-react";

interface MatchDetailDrawerProps {
  match: MatchSummary | null;
  open: boolean;
  onClose: () => void;
}

/** Tab definitions for match detail drawer */
const MATCH_TABS = [
  { id: "overview", icon: <Info />, label: "Overview" },
  { id: "predictions", icon: <TrendingUp />, label: "Predictions" },
  { id: "live", icon: <Radio />, label: "Live Data" },
];

/**
 * Match Detail Content - shared between desktop drawer and mobile sheet
 */
function MatchDetailContent({ match }: { match: MatchSummary }) {
  const [activeTab, setActiveTab] = useState("overview");

  const kickoffDate = new Date(match.kickoffISO);
  const formattedDate = kickoffDate.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
  const formattedTime = kickoffDate.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="w-full space-y-4">
      <IconTabs
        tabs={MATCH_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />

      {/* Overview Tab */}
      {activeTab === "overview" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <StatusDot status={match.status} />
            <Badge variant="secondary" className="text-xs">
              {match.leagueName}
            </Badge>
          </div>

          {/* Score (if available) */}
          {match.score && (
            <div className="bg-background rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-foreground">
                {match.score.home} - {match.score.away}
              </div>
              {match.elapsed && (
                <div className="text-sm text-muted-foreground mt-1">
                  {match.elapsed.min}&apos;
                  {match.elapsed.extra ? ` +${match.elapsed.extra}` : ""}
                </div>
              )}
            </div>
          )}

          {/* Match info */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">
                {formattedDate} at {formattedTime}
              </span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Country:</span>{" "}
              <span className="text-foreground">{match.leagueCountry}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Match ID:</span>{" "}
              <span className="text-foreground font-mono">{match.id}</span>
            </div>
          </div>
        </div>
      )}

      {/* Predictions Tab */}
      {activeTab === "predictions" && (
        <>
          {match.prediction ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Model {match.prediction.model}</span>
              </div>

              <div className="bg-background rounded-lg p-4 space-y-3">
                <div className="text-center">
                  <div className="text-lg font-semibold text-primary capitalize">
                    {match.prediction.pick === "home"
                      ? match.home
                      : match.prediction.pick === "away"
                      ? match.away
                      : "Draw"}
                  </div>
                  <div className="text-xs text-muted-foreground">Predicted winner</div>
                </div>

                {match.prediction.probs && (
                  <div className="space-y-2 pt-2 border-t border-border">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">{match.home}</span>
                      <span className="text-foreground font-medium">
                        {(match.prediction.probs.home * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Draw</span>
                      <span className="text-foreground font-medium">
                        {(match.prediction.probs.draw * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">{match.away}</span>
                      <span className="text-foreground font-medium">
                        {(match.prediction.probs.away * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <TrendingUp className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No prediction available</p>
            </div>
          )}
        </>
      )}

      {/* Live Data Tab */}
      {activeTab === "live" && (
        <>
          <div className="flex items-center gap-2 mb-4">
            <Radio className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Live Data</span>
          </div>

          {match.status === "live" || match.status === "ht" ? (
            <div className="space-y-3">
              <div className="bg-background rounded-lg p-4">
                <div className="text-xs text-muted-foreground mb-2">Events</div>
                <p className="text-sm text-muted-foreground">
                  Live events feed coming soon
                </p>
              </div>
              <div className="bg-background rounded-lg p-4">
                <div className="text-xs text-muted-foreground mb-2">Statistics</div>
                <p className="text-sm text-muted-foreground">
                  Match statistics coming soon
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <Radio className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                Match is not live
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/**
 * Responsive Match Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer (no reflow, ~400px)
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function MatchDetailDrawer({
  match,
  open,
  onClose,
}: MatchDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const matchTitle = match ? `${match.home} vs ${match.away}` : "Match Details";

  // Desktop: overlay drawer
  if (isDesktop) {
    return (
      <DetailDrawer open={open} onClose={onClose} title={matchTitle}>
        {match ? (
          <MatchDetailContent match={match} />
        ) : (
          <p className="text-muted-foreground text-sm">Select a match to view details</p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {matchTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {match ? (
              <MatchDetailContent match={match} />
            ) : (
              <p className="text-muted-foreground text-sm">Select a match to view details</p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
