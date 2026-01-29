"use client";

import { useState } from "react";
import { LogosLeagueSelector } from "./LogosLeagueSelector";
import { LogosGeneratePanel } from "./LogosGeneratePanel";
import { LogosBatchProgress } from "./LogosBatchProgress";
import { LogosReviewGrid } from "./LogosReviewGrid";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getStoredActiveBatch, setStoredActiveBatch } from "@/lib/hooks";
import { ImageIcon, ListChecks, Settings2 } from "lucide-react";

type LogosView = "generate" | "review" | "config";

export function LogosTab() {
  const [selectedLeagueId, setSelectedLeagueId] = useState<number | null>(null);
  // Use lazy initializer to read from localStorage once on mount
  const [activeBatchId, setActiveBatchId] = useState<string | null>(() => getStoredActiveBatch());
  const [activeView, setActiveView] = useState<LogosView>("generate");

  const handleBatchStarted = (batchId: string) => {
    setActiveBatchId(batchId);
    setStoredActiveBatch(batchId);
  };

  const handleBatchComplete = () => {
    setActiveBatchId(null);
    setStoredActiveBatch(null);
    setActiveView("review");
  };

  const handleBatchClear = () => {
    setActiveBatchId(null);
    setStoredActiveBatch(null);
  };

  return (
    <div className="space-y-6">
      <Tabs value={activeView} onValueChange={(v) => setActiveView(v as LogosView)}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="generate" className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Generate
          </TabsTrigger>
          <TabsTrigger value="review" className="flex items-center gap-2">
            <ListChecks className="h-4 w-4" />
            Review
          </TabsTrigger>
          <TabsTrigger value="config" className="flex items-center gap-2">
            <Settings2 className="h-4 w-4" />
            Config
          </TabsTrigger>
        </TabsList>

        {/* Generate Tab */}
        <TabsContent value="generate" className="space-y-4 mt-4">
          <LogosLeagueSelector
            selectedLeagueId={selectedLeagueId}
            onSelect={setSelectedLeagueId}
          />

          {selectedLeagueId && !activeBatchId && (
            <LogosGeneratePanel
              leagueId={selectedLeagueId}
              onBatchStarted={handleBatchStarted}
            />
          )}

          {activeBatchId && (
            <LogosBatchProgress
              batchId={activeBatchId}
              onComplete={handleBatchComplete}
              onClear={handleBatchClear}
            />
          )}
        </TabsContent>

        {/* Review Tab */}
        <TabsContent value="review" className="space-y-4 mt-4">
          <LogosLeagueSelector
            selectedLeagueId={selectedLeagueId}
            onSelect={setSelectedLeagueId}
          />

          {selectedLeagueId && <LogosReviewGrid leagueId={selectedLeagueId} />}

          {!selectedLeagueId && (
            <div className="text-center py-8 text-sm text-muted-foreground">
              Select a league to review generated logos
            </div>
          )}
        </TabsContent>

        {/* Config Tab */}
        <TabsContent value="config" className="space-y-4 mt-4">
          <div className="bg-surface/50 rounded-lg p-4 border border-border space-y-3">
            <h4 className="text-sm font-medium">Current Configuration</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Default Model:</span>
                <span>Imagen 3 (Google AI Studio)</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Pricing Tier:</span>
                <span className="text-[var(--status-success-text)]">Free</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Daily Limit:</span>
                <span>~50 images/day</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Max Batch Cost:</span>
                <span>$50.00</span>
              </div>
            </div>
          </div>

          <div className="bg-surface/50 rounded-lg p-4 border border-border">
            <p className="text-xs text-muted-foreground">
              Configuration is managed via environment variables. To change these
              settings, update the Railway environment and redeploy.
            </p>
            <ul className="mt-2 text-xs text-muted-foreground space-y-1">
              <li>
                <code className="bg-muted px-1 rounded">LOGOS_IA_MODEL</code> - Default model
              </li>
              <li>
                <code className="bg-muted px-1 rounded">LOGOS_USE_FREE_TIER</code> - Free vs paid
              </li>
              <li>
                <code className="bg-muted px-1 rounded">LOGOS_MAX_BATCH_COST_USD</code> - Cost limit
              </li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
