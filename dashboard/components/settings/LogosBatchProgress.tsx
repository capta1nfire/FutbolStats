"use client";

import { useEffect } from "react";
import {
  useLogosBatchStatus,
  usePauseBatch,
  useResumeBatch,
  useCancelBatch,
  clearStoredActiveBatch,
} from "@/lib/hooks";
import { BATCH_STATUS_LABELS, IA_MODEL_LABELS, GENERATION_MODE_LABELS } from "@/lib/types/logos";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  Pause,
  Play,
  XCircle,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface LogosBatchProgressProps {
  batchId: string;
  onComplete: () => void;
  onClear: () => void;
}

export function LogosBatchProgress({
  batchId,
  onComplete,
  onClear,
}: LogosBatchProgressProps) {
  const { data: batch, isLoading, error } = useLogosBatchStatus(batchId);
  const pauseBatch = usePauseBatch();
  const resumeBatch = useResumeBatch();
  const cancelBatch = useCancelBatch();

  // Auto-clear when completed
  useEffect(() => {
    if (batch?.status === "completed" || batch?.status === "pending_review") {
      clearStoredActiveBatch();
      // Give user a moment to see completion before switching to review
      const timer = setTimeout(() => {
        onComplete();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [batch?.status, onComplete]);

  const handlePause = async () => {
    try {
      await pauseBatch.mutateAsync(batchId);
      toast.success("Batch paused");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to pause");
    }
  };

  const handleResume = async () => {
    try {
      await resumeBatch.mutateAsync(batchId);
      toast.success("Batch resumed");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to resume");
    }
  };

  const handleCancel = async () => {
    try {
      await cancelBatch.mutateAsync(batchId);
      clearStoredActiveBatch();
      toast.success("Batch cancelled");
      onClear();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to cancel");
    }
  };

  if (isLoading) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-border">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Loading batch status...</span>
        </div>
      </div>
    );
  }

  if (error || !batch) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-border">
        <div className="flex items-center gap-2 text-sm text-[var(--status-error-text)]">
          <AlertTriangle className="h-4 w-4" />
          <span>Failed to load batch status</span>
        </div>
        <Button variant="outline" size="sm" className="mt-2" onClick={onClear}>
          Clear
        </Button>
      </div>
    );
  }

  const isRunning = batch.status === "running";
  const isPaused = batch.status === "paused";
  const isCompleted = batch.status === "completed" || batch.status === "pending_review";
  const isCancelled = batch.status === "cancelled";
  const isError = batch.status === "error";

  const statusColor = isCompleted
    ? "text-[var(--status-success-text)]"
    : isError || isCancelled
      ? "text-[var(--status-error-text)]"
      : isPaused
        ? "text-[var(--status-warning-text)]"
        : "text-primary";

  return (
    <div className="bg-surface rounded-lg p-4 space-y-4 border border-border">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-medium">Batch Progress</h4>
          <p className="text-xs text-muted-foreground">
            {IA_MODEL_LABELS[batch.iaModel]} | {GENERATION_MODE_LABELS[batch.generationMode]}
          </p>
        </div>
        <Badge
          variant="outline"
          className={cn(
            statusColor,
            isCompleted && "bg-[var(--status-success-bg)] border-[var(--status-success-border)]",
            isError && "bg-[var(--status-error-bg)] border-[var(--status-error-border)]",
            isPaused && "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)]"
          )}
        >
          {isRunning && <Loader2 className="h-3 w-3 mr-1 animate-spin" />}
          {isCompleted && <CheckCircle className="h-3 w-3 mr-1" />}
          {isError && <AlertTriangle className="h-3 w-3 mr-1" />}
          {BATCH_STATUS_LABELS[batch.status]}
        </Badge>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="h-2 bg-background rounded-full overflow-hidden">
          <div
            className={cn(
              "h-full transition-all duration-500",
              isCompleted ? "bg-[var(--status-success-text)]" : "bg-primary"
            )}
            style={{ width: `${batch.progress}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{batch.progress.toFixed(0)}%</span>
          <span>
            {batch.processedTeams}/{batch.totalTeams} teams
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <p className="text-xs text-muted-foreground">Images</p>
          <p className="font-medium">{batch.processedImages}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Failed</p>
          <p className={cn("font-medium", batch.failedTeams > 0 && "text-[var(--status-error-text)]")}>
            {batch.failedTeams}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Cost</p>
          <p className="font-medium">
            ${batch.actualCostUsd.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Actions */}
      {(isRunning || isPaused) && (
        <div className="flex items-center gap-2 pt-2 border-t border-border">
          {isRunning && (
            <Button
              variant="outline"
              size="sm"
              onClick={handlePause}
              disabled={pauseBatch.isPending}
            >
              {pauseBatch.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <Pause className="h-4 w-4 mr-1" />
                  Pause
                </>
              )}
            </Button>
          )}
          {isPaused && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleResume}
              disabled={resumeBatch.isPending}
            >
              {resumeBatch.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <Play className="h-4 w-4 mr-1" />
                  Resume
                </>
              )}
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleCancel}
            disabled={cancelBatch.isPending}
            className="text-[var(--status-error-text)] hover:bg-[var(--status-error-bg)]"
          >
            {cancelBatch.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <XCircle className="h-4 w-4 mr-1" />
                Cancel
              </>
            )}
          </Button>
        </div>
      )}

      {/* Completed Actions */}
      {(isCompleted || isCancelled || isError) && (
        <div className="flex items-center gap-2 pt-2 border-t border-border">
          <Button variant="outline" size="sm" onClick={onClear}>
            Clear
          </Button>
          {isCompleted && (
            <Button size="sm" onClick={onComplete}>
              View Results
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
