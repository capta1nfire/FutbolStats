"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  useLogosLeagues,
  useStartBatch,
  useTeamsReadyForTest,
  useGenerateSingleTeam,
} from "@/lib/hooks";
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
import { AlertTriangle, Loader2, Sparkles, FlaskConical, Users } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

type BatchMode = "league" | "single_team";

interface LogosGeneratePanelProps {
  leagueId: number;
  onBatchStarted: (batchId: string) => void;
}

export function LogosGeneratePanel({
  leagueId,
  onBatchStarted,
}: LogosGeneratePanelProps) {
  const router = useRouter();
  const [batchMode, setBatchMode] = useState<BatchMode>("league");
  const [generationMode, setGenerationMode] = useState<GenerationMode>("full_3d");
  const [iaModel, setIaModel] = useState<IAModel>("imagen-3");
  const [selectedTeamId, setSelectedTeamId] = useState<number | null>(null);

  const { data: leagues, isLoading: isLoadingLeagues } = useLogosLeagues();
  const { data: teamsReady, isLoading: isLoadingTeams } = useTeamsReadyForTest();
  const startBatch = useStartBatch();
  const generateSingle = useGenerateSingleTeam();

  const selectedLeague = leagues?.find((l) => l.leagueId === leagueId);
  const teamCount = batchMode === "league" ? (selectedLeague?.pendingCount || 0) : 1;
  const teamsWithOriginal = teamsReady?.teams || [];

  const costEstimate = estimateCost(teamCount, generationMode, iaModel);
  const totalImages = teamCount * costEstimate.imagesPerTeam;

  const exceedsFreeTierLimit = costEstimate.isFree && totalImages > FREE_TIER_DAILY_LIMIT;

  const handleStartGeneration = async () => {
    if (batchMode === "single_team") {
      if (!selectedTeamId) {
        toast.error("Please select a team first");
        return;
      }

      try {
        // useGenerateSingleTeam now returns TeamLogoStatus after async polling
        const finalStatus = await generateSingle.mutateAsync({
          teamId: selectedTeamId,
          request: {
            generation_mode: generationMode,
            ia_model: iaModel,
            prompt_version: "v1",
          },
        });

        // Check final status and show appropriate toast
        if (finalStatus.status === "error") {
          toast.error(
            finalStatus.error?.message || "Generation failed"
          );
        } else if (finalStatus.status === "pending_resize" || finalStatus.status === "ready") {
          // Count variants by checking r2Keys
          const variantCount = [
            finalStatus.r2Keys.front,
            finalStatus.r2Keys.right,
            finalStatus.r2Keys.left,
          ].filter(Boolean).length;

          const teamIdForDrawer = selectedTeamId;
          toast.success(
            `Generated ${variantCount} variant${variantCount !== 1 ? "s" : ""} for ${finalStatus.teamName}`,
            {
              action: {
                label: "Ver equipo",
                onClick: () => router.push(`/football?team=${teamIdForDrawer}`),
              },
            }
          );
        } else {
          toast.info(`Generation completed with status: ${finalStatus.status}`);
        }
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Generation failed"
        );
      }
    } else {
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
    }
  };

  const isPending = startBatch.isPending || generateSingle.isPending;

  // Show loading state while leagues are loading
  if (isLoadingLeagues) {
    return (
      <div className="bg-surface/50 rounded-lg p-4 border border-border flex items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">Loading league data...</span>
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-lg p-4 space-y-4 border border-border">
      <h4 className="text-sm font-medium">Generation Settings</h4>

      {/* Batch Mode Toggle */}
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          className={cn(
            "flex-1",
            batchMode === "league" && "bg-primary/10 border-primary"
          )}
          onClick={() => setBatchMode("league")}
        >
          <Users className="h-4 w-4 mr-2" />
          Full League
        </Button>
        <Button
          variant="outline"
          size="sm"
          className={cn(
            "flex-1",
            batchMode === "single_team" && "bg-primary/10 border-primary"
          )}
          onClick={() => setBatchMode("single_team")}
        >
          <FlaskConical className="h-4 w-4 mr-2" />
          Single Team Test
        </Button>
      </div>

      {/* Single Team Selector (only in single_team mode) */}
      {batchMode === "single_team" && (
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground">
            Team with Original Logo ({teamsWithOriginal.length} available)
          </Label>
          {isLoadingTeams ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading teams...
            </div>
          ) : teamsWithOriginal.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No teams have original logos uploaded. Upload a logo first.
            </p>
          ) : (
            <Select
              value={selectedTeamId?.toString() || ""}
              onValueChange={(v) => setSelectedTeamId(Number(v))}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a team..." />
              </SelectTrigger>
              <SelectContent>
                {teamsWithOriginal.map((team) => (
                  <SelectItem key={team.teamId} value={team.teamId.toString()}>
                    <div className="flex items-center gap-2">
                      <span>{team.teamName}</span>
                      {team.hasVariants && (
                        <span className="text-xs text-muted-foreground">(has variants)</span>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      )}

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

      {/* Free Tier Warning (only for league batch) */}
      {batchMode === "league" && exceedsFreeTierLimit && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)]">
          <AlertTriangle className="h-4 w-4 text-[var(--status-warning-text)] mt-0.5 shrink-0" />
          <p className="text-xs text-[var(--status-warning-text)]">
            Free tier limit is ~{FREE_TIER_DAILY_LIMIT} images/day. You are requesting{" "}
            {totalImages} images. Consider processing in batches or using a paid model.
          </p>
        </div>
      )}

      {/* Single Team Test Info */}
      {batchMode === "single_team" && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-primary/5 border border-primary/20">
          <FlaskConical className="h-4 w-4 text-primary mt-0.5 shrink-0" />
          <p className="text-xs text-muted-foreground">
            Test mode: Generate variants for one team to test prompts before running a full batch.
            Only {totalImages} image{totalImages > 1 ? "s" : ""} will be generated.
          </p>
        </div>
      )}

      {/* Start Button */}
      <Button
        className="w-full"
        onClick={handleStartGeneration}
        disabled={
          isPending ||
          (batchMode === "league" && teamCount === 0) ||
          (batchMode === "single_team" && !selectedTeamId)
        }
      >
        {isPending ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            {batchMode === "single_team" ? "Generating..." : "Starting..."}
          </>
        ) : (
          <>
            <Sparkles className="h-4 w-4 mr-2" />
            {batchMode === "single_team" ? "Generate Test" : "Start Generation"}
          </>
        )}
      </Button>
    </div>
  );
}
