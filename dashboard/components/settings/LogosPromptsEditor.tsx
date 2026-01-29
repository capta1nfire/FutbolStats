"use client";

import { useState, useCallback } from "react";
import { usePromptTemplates, useUpdatePrompt } from "@/lib/hooks/use-logos";
import type { PromptTemplate } from "@/lib/api/logos";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Save,
  Loader2,
  MessageSquare,
  Check,
  X,
  RefreshCw,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const VARIANT_LABELS: Record<string, string> = {
  front: "Front (Frontal)",
  right: "Right (Facing Right)",
  left: "Left (Facing Left)",
  main: "Main (Leagues/Tournaments)",
};

const VARIANT_DESCRIPTIONS: Record<string, string> = {
  front: "Vista frontal del escudo 3D",
  right: "Escudo rotado 45° mirando a la derecha",
  left: "Escudo rotado 45° mirando a la izquierda",
  main: "Para logos de ligas y torneos",
};

/**
 * Single Prompt Editor Card
 */
function PromptCard({
  prompt,
  onSave,
  isSaving,
}: {
  prompt: PromptTemplate;
  onSave: (promptId: number, newText: string) => void;
  isSaving: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [editedText, setEditedText] = useState(prompt.promptTemplate);
  const [hasChanges, setHasChanges] = useState(false);

  const handleTextChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newText = e.target.value;
      setEditedText(newText);
      setHasChanges(newText !== prompt.promptTemplate);
    },
    [prompt.promptTemplate]
  );

  const handleSave = useCallback(() => {
    onSave(prompt.id, editedText);
    setHasChanges(false);
  }, [prompt.id, editedText, onSave]);

  const handleReset = useCallback(() => {
    setEditedText(prompt.promptTemplate);
    setHasChanges(false);
  }, [prompt.promptTemplate]);

  return (
    <div
      className={cn(
        "bg-surface rounded-lg border transition-colors",
        hasChanges ? "border-primary" : "border-border",
        isExpanded ? "shadow-sm" : ""
      )}
    >
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <MessageSquare className="h-4 w-4 text-muted-foreground" />
          <div className="text-left">
            <p className="text-sm font-medium">{VARIANT_LABELS[prompt.variant]}</p>
            <p className="text-xs text-muted-foreground">
              {VARIANT_DESCRIPTIONS[prompt.variant]}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {prompt.isActive ? (
            <Badge
              variant="outline"
              className="text-xs bg-[var(--status-success-bg)] text-[var(--status-success-text)]"
            >
              Active
            </Badge>
          ) : (
            <Badge variant="outline" className="text-xs">
              Inactive
            </Badge>
          )}
          {hasChanges && (
            <Badge variant="outline" className="text-xs text-primary">
              Modified
            </Badge>
          )}
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-border/50">
          {/* Stats */}
          <div className="flex items-center gap-4 pt-3 text-xs text-muted-foreground">
            <span>Version: {prompt.version}</span>
            {prompt.usageCount !== undefined && (
              <span>Uses: {prompt.usageCount}</span>
            )}
            {prompt.successRate !== undefined && (
              <span>Success: {(prompt.successRate * 100).toFixed(0)}%</span>
            )}
          </div>

          {/* Textarea */}
          <textarea
            value={editedText}
            onChange={handleTextChange}
            rows={6}
            className={cn(
              "w-full p-3 text-sm font-mono rounded-lg border resize-none",
              "bg-background focus:outline-none focus:ring-2 focus:ring-primary/50",
              hasChanges ? "border-primary" : "border-border"
            )}
            placeholder="Enter generation prompt..."
          />

          {/* Character count */}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{editedText.length} characters</span>
            {prompt.notes && <span>Note: {prompt.notes}</span>}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 pt-2">
            <Button
              variant="default"
              size="sm"
              onClick={handleSave}
              disabled={!hasChanges || isSaving}
              className="text-xs"
            >
              {isSaving ? (
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
              ) : (
                <Save className="h-3 w-3 mr-1" />
              )}
              Save Changes
            </Button>
            {hasChanges && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
                className="text-xs"
              >
                <X className="h-3 w-3 mr-1" />
                Discard
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Logos Prompts Editor
 *
 * Allows editing of generation prompts for each variant (front, right, left, main).
 * Changes are persisted to the database via API.
 */
export function LogosPromptsEditor() {
  const { data: prompts, isLoading, error, refetch } = usePromptTemplates();
  const updateMutation = useUpdatePrompt();
  const [savingId, setSavingId] = useState<number | null>(null);

  const handleSave = useCallback(
    async (promptId: number, newText: string) => {
      setSavingId(promptId);
      try {
        await updateMutation.mutateAsync({
          promptId,
          data: { promptTemplate: newText },
        });
        toast.success("Prompt updated successfully");
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Failed to update prompt");
      } finally {
        setSavingId(null);
      }
    },
    [updateMutation]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">
          Loading prompts...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8 space-y-2">
        <p className="text-sm text-[var(--status-error-text)]">
          Failed to load prompts
        </p>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-3 w-3 mr-1" />
          Retry
        </Button>
      </div>
    );
  }

  // Group prompts by version
  const versions = [...new Set(prompts?.map((p) => p.version) || [])];
  const activeVersion = versions[0] || "v1"; // Most recent version
  const activePrompts = prompts?.filter((p) => p.version === activeVersion) || [];

  // Sort by variant order
  const variantOrder = ["front", "right", "left", "main"];
  activePrompts.sort(
    (a, b) => variantOrder.indexOf(a.variant) - variantOrder.indexOf(b.variant)
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-medium">Generation Prompts</h4>
          <p className="text-xs text-muted-foreground">
            Edit the prompts used for 3D logo generation
          </p>
        </div>
        <Badge variant="outline" className="text-xs">
          {activeVersion}
        </Badge>
      </div>

      {/* Prompt Cards */}
      <div className="space-y-2">
        {activePrompts.map((prompt) => (
          <PromptCard
            key={prompt.id}
            prompt={prompt}
            onSave={handleSave}
            isSaving={savingId === prompt.id}
          />
        ))}
      </div>

      {/* Help Text */}
      <div className="p-3 bg-muted/30 rounded-lg">
        <p className="text-xs text-muted-foreground">
          <strong>Tips:</strong> Use descriptive prompts that specify style (metallic,
          glossy), perspective (45 degrees), lighting direction, and background
          (transparent). Changes take effect on new generations.
        </p>
      </div>
    </div>
  );
}
