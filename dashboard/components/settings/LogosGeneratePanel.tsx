"use client";

import { useState } from "react";
import { useLogosLeagues, useStartBatch } from "@/lib/hooks";
import { estimateCost } from "@/lib/api/logos";
import {
  GenerationMode,
  IAModel,
  IA_MODEL_LABELS,
  GENERATION_MODE_LABELS,
  FREE_TIER_DAILY_LIMIT,
} from "@/lib/types/logos";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { AlertTriangle, Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";

interface LogosGeneratePanelProps {
  leagueId: number;
  onBatchStarted: (batchId: string) => void;
}

export function LogosGeneratePanel({
  leagueId,
  onBatchStarted,
}: LogosGeneratePanelProps) {
  const [generationMode, setGenerationMode] = useState<GenerationMode>("full_3d");
  const [iaModel, setIaModel] = useState<IAModel>("imagen-3");

  const { data: leagues, isLoading: isLoadingLeagues } = useLogosLeagues();
  const startBatch = useStartBatch();

  const selectedLeague = leagues?.find((l) => l.leagueId === leagueId);
  const teamCount = selectedLeague?.pendingCount || 0;

  const costEstimate = estimateCost(teamCount, generationMode, iaModel);
  const totalImages = teamCount * costEstimate.imagesPerTeam;

  const exceedsFreeTierLimit = costEstimate.isFree && totalImages > FREE_TIER_DAILY_LIMIT;

  const handleStartGeneration = async () => {
    try {
      const result = await startBatch.mutateAsync({
        leagueId,
        request: {
          generation_mode: generationMode,
          ia_model: iaModel,
        },
      });

      toast.success(`Batch started: ${result.batch_id}`);
      onBatchStarted(result.batch_id);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to start batch"
      );
    }
  };

  // Show loading state while leagues are loading
  if (isLoadingLeagues) {
    return (
      <div className="bg-surface/50 rounded-lg p-4 border border-border flex items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">Loading league data...</span>
      </div>
    );
  }

  if (teamCount === 0) {
    return (
      <div className="bg-surface/50 rounded-lg p-4 border border-border">
        <p className="text-sm text-muted-foreground">
          No pending teams in this league. All teams are either ready or need to
          have their original logos uploaded first.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-lg p-4 space-y-4 border border-border">
      <h4 className="text-sm font-medium">Generation Settings</h4>

      <div className="grid grid-cols-2 gap-4">
        {/* Generation Mode */}
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground">Mode</Label>
          <Select
            value={generationMode}
            onValueChange={(v) => setGenerationMode(v as GenerationMode)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(GENERATION_MODE_LABELS) as GenerationMode[])
                .filter((m) => m !== "manual")
                .map((mode) => (
                  <SelectItem key={mode} value={mode}>
                    {GENERATION_MODE_LABELS[mode]}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>

        {/* IA Model */}
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground">Model</Label>
          <Select
            value={iaModel}
            onValueChange={(v) => setIaModel(v as IAModel)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(IA_MODEL_LABELS) as IAModel[]).map((model) => (
                <SelectItem key={model} value={model}>
                  {IA_MODEL_LABELS[model]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Cost Estimate */}
      <div className="bg-background rounded-lg p-3 space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Teams to process:</span>
          <span className="font-medium">{teamCount}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Images to generate:</span>
          <span className="font-medium">{totalImages}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Estimated cost:</span>
          <span className="font-medium">
            {costEstimate.isFree ? (
              <span className="text-[var(--status-success-text)]">$0.00 (Free Tier)</span>
            ) : (
              `$${costEstimate.totalCost.toFixed(2)}`
            )}
          </span>
        </div>
      </div>

      {/* Free Tier Warning */}
      {exceedsFreeTierLimit && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)]">
          <AlertTriangle className="h-4 w-4 text-[var(--status-warning-text)] mt-0.5 shrink-0" />
          <p className="text-xs text-[var(--status-warning-text)]">
            Free tier limit is ~{FREE_TIER_DAILY_LIMIT} images/day. You are requesting{" "}
            {totalImages} images. Consider processing in batches or using a paid model.
          </p>
        </div>
      )}

      {/* Start Button */}
      <Button
        className="w-full"
        onClick={handleStartGeneration}
        disabled={startBatch.isPending || teamCount === 0}
      >
        {startBatch.isPending ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Starting...
          </>
        ) : (
          <>
            <Sparkles className="h-4 w-4 mr-2" />
            Start Generation
          </>
        )}
      </Button>
    </div>
  );
}
