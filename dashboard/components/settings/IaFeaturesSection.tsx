"use client";

import { useState, useEffect } from "react";
import { useIaFeatures } from "@/lib/hooks";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { PromptTemplateViewer } from "./PromptTemplateViewer";
import { PayloadPreview } from "./PayloadPreview";
import { CallHistoryList } from "./CallHistoryList";
import { LLMPlayground } from "./LLMPlayground";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Brain,
  Loader2,
  AlertTriangle,
  MessageSquare,
  Save,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { IaFeaturesUpdatePayload } from "@/lib/api/settings";

type NarrativesState = "on" | "off" | "inherit";

function narrativesStateFromApi(value: boolean | null): NarrativesState {
  if (value === null) return "inherit";
  return value ? "on" : "off";
}

function narrativesStateToApi(state: NarrativesState): boolean | null {
  if (state === "inherit") return null;
  return state === "on";
}

export function IaFeaturesSection() {
  const { data: config, isLoading, error, refetch, update, isUpdating } = useIaFeatures();

  // Local state for form
  const [narrativesState, setNarrativesState] = useState<NarrativesState>("inherit");
  const [primaryModel, setPrimaryModel] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(4096);

  // Track if form has unsaved changes
  const [hasChanges, setHasChanges] = useState(false);

  // Sync local state with fetched config
  useEffect(() => {
    if (config) {
      setNarrativesState(narrativesStateFromApi(config.narrativesEnabled));
      setPrimaryModel(config.primaryModel);
      setTemperature(config.temperature);
      setMaxTokens(config.maxTokens);
      setHasChanges(false);
    }
  }, [config]);

  // Check for changes
  useEffect(() => {
    if (!config) return;

    const changed =
      narrativesState !== narrativesStateFromApi(config.narrativesEnabled) ||
      primaryModel !== config.primaryModel ||
      temperature !== config.temperature ||
      maxTokens !== config.maxTokens;

    setHasChanges(changed);
  }, [config, narrativesState, primaryModel, temperature, maxTokens]);

  const handleSave = async () => {
    const payload: IaFeaturesUpdatePayload = {
      narratives_enabled: narrativesStateToApi(narrativesState),
      primary_model: primaryModel,
      temperature,
      max_tokens: maxTokens,
    };

    const result = await update(payload);

    if (result.success) {
      toast.success("IA Features configuration saved");
    } else {
      toast.error(result.error || "Failed to save configuration");
    }
  };

  const handleReset = () => {
    if (config) {
      setNarrativesState(narrativesStateFromApi(config.narrativesEnabled));
      setPrimaryModel(config.primaryModel);
      setTemperature(config.temperature);
      setMaxTokens(config.maxTokens);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
        <SettingsSectionHeader
          title="IA Features"
          description="Configure narrative generation and LLM settings"
        />
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (error || !config) {
    return (
      <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
        <SettingsSectionHeader
          title="IA Features"
          description="Configure narrative generation and LLM settings"
        />
        <div className="flex flex-col items-center justify-center py-12 gap-4">
          <AlertTriangle className="h-8 w-8 text-[var(--status-warning-text)]" />
          <p className="text-sm text-muted-foreground">
            Failed to load IA Features configuration
          </p>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  const selectedModel = config.availableModels.find((m) => m.id === primaryModel);

  return (
    <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
      <SettingsSectionHeader
        title="IA Features"
        description="Configure narrative generation and LLM settings"
      />

      <div className="space-y-6">
        {/* Narratives Toggle (3-state) */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Narrativas Habilitadas</span>
            </div>
            <div className="flex items-center gap-2">
              <Select value={narrativesState} onValueChange={(v: string) => setNarrativesState(v as NarrativesState)}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="on">On</SelectItem>
                  <SelectItem value="off">Off</SelectItem>
                  <SelectItem value="inherit">Inherit from env</SelectItem>
                </SelectContent>
              </Select>
              <Badge
                variant="outline"
                className={cn(
                  config.effectiveEnabled
                    ? "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]"
                    : "bg-surface text-muted-foreground border-border"
                )}
              >
                {config.effectiveEnabled ? "Active" : "Inactive"}
              </Badge>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            {narrativesState === "inherit"
              ? `Inheriting from FASTPATH_ENABLED env var (currently ${config.envFastpathEnabled ? "enabled" : "disabled"})`
              : narrativesState === "on"
                ? "Narratives are explicitly enabled (override)"
                : "Narratives are explicitly disabled (override)"}
          </p>
        </div>

        {/* Feedback Toggle (disabled placeholder) */}
        <div className="bg-surface rounded-lg p-4 space-y-4 opacity-60">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Feedback de Narrativas</span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary">Soon</Badge>
              <Badge
                variant="outline"
                className="bg-surface text-muted-foreground border-border"
              >
                Disabled
              </Badge>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Recolecta feedback de usuarios sobre la calidad de las narrativas (Phase 2)
          </p>
        </div>

        {/* Divider */}
        <div className="border-t border-border" />

        {/* Primary Model Selection */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Modelo Primario</Label>
          <Select value={primaryModel} onValueChange={setPrimaryModel}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {config.availableModels.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <div className="flex items-center justify-between w-full gap-4">
                    <span>{model.displayName}</span>
                    <span className="text-xs text-muted-foreground">
                      ${model.inputPrice.toFixed(2)}/${model.outputPrice.toFixed(2)} per 1M
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedModel && (
            <p className="text-xs text-muted-foreground">
              Provider: {selectedModel.provider} | Max tokens: {selectedModel.maxTokens.toLocaleString()}
            </p>
          )}
        </div>

        {/* Advanced Settings (Accordion) */}
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="advanced" className="border-none">
            <AccordionTrigger className="py-2 hover:no-underline">
              <span className="text-sm font-medium">Configuracion Avanzada</span>
            </AccordionTrigger>
            <AccordionContent className="pt-4 space-y-6">
              {/* Temperature */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Temperature</Label>
                  <span className="text-sm font-mono text-muted-foreground">
                    {temperature.toFixed(2)}
                  </span>
                </div>
                <Input
                  type="range"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  min={0}
                  max={1}
                  step={0.05}
                  className="w-full h-2 cursor-pointer"
                />
                <p className="text-xs text-muted-foreground">
                  Lower values produce more focused output, higher values are more creative
                </p>
              </div>

              {/* Max Tokens */}
              <div className="space-y-3">
                <Label className="text-sm">Max Tokens</Label>
                <Input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value) || 4096)}
                  min={100}
                  max={131072}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Maximum tokens for narrative generation (100 - 131,072)
                </p>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Save Button */}
        {hasChanges && (
          <div className="flex items-center justify-end gap-2 pt-4 border-t border-border">
            <Button variant="outline" size="sm" onClick={handleReset} disabled={isUpdating}>
              Reset
            </Button>
            <Button size="sm" onClick={handleSave} disabled={isUpdating}>
              {isUpdating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Guardar Cambios
                </>
              )}
            </Button>
          </div>
        )}

        {/* Divider before visibility section */}
        <div className="border-t border-border pt-6 mt-6" />

        {/* LLM Visibility Section (Fase 2) */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-muted-foreground">LLM Visibility</h3>

          {/* Prompt Template Viewer */}
          <PromptTemplateViewer />

          {/* Payload Preview */}
          <PayloadPreview />

          {/* Call History */}
          <CallHistoryList />

          {/* LLM Playground (Fase 3) */}
          <LLMPlayground />
        </div>
      </div>
    </div>
  );
}
