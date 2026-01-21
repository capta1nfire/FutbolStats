"use client";

import { MatchSummary } from "@/lib/types";
import { DetailDrawer } from "@/components/shell";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { StatusDot } from "./StatusDot";
import { Calendar, TrendingUp, Radio } from "lucide-react";

interface MatchDetailDrawerProps {
  match: MatchSummary | null;
  open: boolean;
  onClose: () => void;
}

export function MatchDetailDrawer({
  match,
  open,
  onClose,
}: MatchDetailDrawerProps) {
  if (!match) {
    return (
      <DetailDrawer open={open} onClose={onClose} title="Match Details">
        <p className="text-muted-foreground text-sm">Select a match to view details</p>
      </DetailDrawer>
    );
  }

  const matchTitle = `${match.home} vs ${match.away}`;
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
    <DetailDrawer open={open} onClose={onClose} title={matchTitle}>
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="w-full grid grid-cols-3 mb-4">
          <TabsTrigger value="overview" className="rounded-full text-xs">
            Overview
          </TabsTrigger>
          <TabsTrigger value="predictions" className="rounded-full text-xs">
            Predictions
          </TabsTrigger>
          <TabsTrigger value="live" className="rounded-full text-xs">
            Live Data
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
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
        </TabsContent>

        {/* Predictions Tab */}
        <TabsContent value="predictions" className="space-y-4">
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
        </TabsContent>

        {/* Live Data Tab */}
        <TabsContent value="live" className="space-y-4">
          <div className="flex items-center gap-2 mb-4">
            <Radio className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Live Data</span>
          </div>

          {match.status === "live" || match.status === "ht" ? (
            <div className="space-y-3">
              <div className="bg-background rounded-lg p-4">
                <div className="text-xs text-muted-foreground mb-2">Events</div>
                <p className="text-sm text-muted-foreground">
                  Live events feed coming in Phase 1
                </p>
              </div>
              <div className="bg-background rounded-lg p-4">
                <div className="text-xs text-muted-foreground mb-2">Statistics</div>
                <p className="text-sm text-muted-foreground">
                  Match statistics coming in Phase 1
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
        </TabsContent>
      </Tabs>
    </DetailDrawer>
  );
}
