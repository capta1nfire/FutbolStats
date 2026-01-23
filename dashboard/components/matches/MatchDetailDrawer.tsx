"use client";

import { useState } from "react";
import { MatchSummary, ProbabilitySet } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import { Loader } from "@/components/ui/loader";
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

/** Section for displaying a single prediction model */
function PredictionSection({
  label,
  probs,
  home,
  away,
}: {
  label: string;
  probs: ProbabilitySet;
  home: string;
  away: string;
}) {
  const maxProb = Math.max(probs.home, probs.draw, probs.away);
  const pick =
    probs.home === maxProb ? home : probs.draw === maxProb ? "Draw" : away;

  return (
    <div className="space-y-2 pb-3 border-b border-border last:border-b-0">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <Badge variant="secondary" className="text-xs">
          {pick}
        </Badge>
      </div>
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <span className={probs.home === maxProb ? "text-foreground" : "text-muted-foreground"}>
            {home}
          </span>
          <span className={probs.home === maxProb ? "text-foreground font-medium" : "text-muted-foreground"}>
            {(probs.home * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className={probs.draw === maxProb ? "text-foreground" : "text-muted-foreground"}>
            Draw
          </span>
          <span className={probs.draw === maxProb ? "text-foreground font-medium" : "text-muted-foreground"}>
            {(probs.draw * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className={probs.away === maxProb ? "text-foreground" : "text-muted-foreground"}>
            {away}
          </span>
          <span className={probs.away === maxProb ? "text-foreground font-medium" : "text-muted-foreground"}>
            {(probs.away * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}

interface MatchDetailDrawerProps {
  match: MatchSummary | null;
  /** True when match is being fetched for deep-link / pagination fallback */
  isLoading?: boolean;
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
        <div className="bg-surface rounded-lg p-4 space-y-4">
          {match.modelA || match.shadow || match.sensorB || match.market ? (
            <>
              {/* Model A */}
              {match.modelA && (
                <PredictionSection
                  label="Model A"
                  probs={match.modelA}
                  home={match.home}
                  away={match.away}
                />
              )}

              {/* Shadow */}
              {match.shadow && (
                <PredictionSection
                  label="Shadow"
                  probs={match.shadow}
                  home={match.home}
                  away={match.away}
                />
              )}

              {/* Sensor B */}
              {match.sensorB && (
                <PredictionSection
                  label="Sensor B"
                  probs={match.sensorB}
                  home={match.home}
                  away={match.away}
                />
              )}

              {/* Market */}
              {match.market && (
                <PredictionSection
                  label="Market"
                  probs={match.market}
                  home={match.home}
                  away={match.away}
                />
              )}
            </>
          ) : (
            <div className="text-center py-8">
              <TrendingUp className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No predictions available</p>
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
  isLoading = false,
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
        ) : isLoading ? (
          <div className="h-full flex items-center justify-center py-10">
            <Loader size="md" />
          </div>
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
            ) : isLoading ? (
              <div className="h-full flex items-center justify-center py-10">
                <Loader size="md" />
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Select a match to view details</p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
