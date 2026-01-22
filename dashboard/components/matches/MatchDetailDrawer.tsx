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
 * Tab content only - without tabs component (for desktop drawer with fixedContent)
 */
function MatchTabContent({ match, activeTab }: { match: MatchSummary; activeTab: string }) {
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
    <div className="w-full">
      {/* Overview Tab */}
      {activeTab === "overview" && (
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <StatusDot status={match.status} />
            <Badge variant="secondary" className="text-xs">
              {match.leagueName}
            </Badge>
          </div>

          {/* Score (if available) */}
          {match.score && (
            <div className="text-center py-2">
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
        <div className="bg-surface rounded-lg p-4">
          {match.prediction ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Model {match.prediction.model}</span>
              </div>

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
                <div className="space-y-2 pt-3 border-t border-border">
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
          ) : (
            <div className="text-center py-8">
              <TrendingUp className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No prediction available</p>
            </div>
          )}
        </div>
      )}

      {/* Live Data Tab */}
      {activeTab === "live" && (
        <div className="bg-surface rounded-lg p-4">
          <div className="flex items-center gap-2 mb-4">
            <Radio className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Live Data</span>
          </div>

          {match.status === "live" || match.status === "ht" ? (
            <div className="space-y-3">
              <div>
                <div className="text-xs text-muted-foreground mb-1">Events</div>
                <p className="text-sm text-muted-foreground">
                  Live events feed coming soon
                </p>
              </div>
              <div className="pt-3 border-t border-border">
                <div className="text-xs text-muted-foreground mb-1">Statistics</div>
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
        </div>
      )}
    </div>
  );
}

/**
 * Match Detail Content - used for mobile sheet (tabs + content together)
 */
function MatchDetailContentMobile({ match }: { match: MatchSummary }) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="w-full space-y-3">
      <IconTabs
        tabs={MATCH_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />
      <MatchTabContent match={match} activeTab={activeTab} />
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
  const [activeTab, setActiveTab] = useState("overview");
  const matchTitle = match ? `${match.home} vs ${match.away}` : "Match Details";

  // Desktop: overlay drawer with tabs in fixedContent (prevents tooltip clipping)
  if (isDesktop) {
    return (
      <DetailDrawer
        open={open}
        onClose={onClose}
        title={matchTitle}
        fixedContent={
          match && (
            <IconTabs
              tabs={MATCH_TABS}
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            />
          )
        }
      >
        {match ? (
          <MatchTabContent match={match} activeTab={activeTab} />
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
              <MatchDetailContentMobile match={match} />
            ) : (
              <p className="text-muted-foreground text-sm">Select a match to view details</p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
