"use client";

import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  FlaskConical,
  Loader2,
  AlertTriangle,
  Clock,
  Coins,
  Hash,
} from "lucide-react";
import { parseCallHistory, parsePlaygroundResponse, PlaygroundResponse } from "@/lib/api/settings";
import { toast } from "sonner";
import { useIaFeatures } from "@/lib/hooks";

interface MatchOption {
  id: number;
  label: string;
}

function formatDuration(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

export function LLMPlayground() {
  const { data: config } = useIaFeatures();

  const [matches, setMatches] = useState<MatchOption[]>([]);
  const [selectedMatchId, setSelectedMatchId] = useState<string>("");
  const [temperature, setTemperature] = useState(0.7);
  const [model, setModel] = useState("");
  const [isLoadingMatches, setIsLoadingMatches] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<PlaygroundResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rateLimit, setRateLimit] = useState<{ remaining: number; resetAt: string } | null>(null);

  // Initialize defaults from config
  useEffect(() => {
    if (config) {
      setTemperature(config.temperature);
      setModel(config.primaryModel);
    }
  }, [config]);

  // Fetch matches from call history
  useEffect(() => {
    async function fetchMatches() {
      setIsLoadingMatches(true);

      try {
        const response = await fetch("/api/settings/ia-features/call-history?limit=20");
        if (!response.ok) {
          throw new Error(`Failed to fetch: ${response.status}`);
        }

        const json = await response.json();
        const parsed = parseCallHistory(json);

        if (parsed) {
          const uniqueMatches = new Map<number, MatchOption>();
          for (const item of parsed.items) {
            if (!uniqueMatches.has(item.matchId)) {
              uniqueMatches.set(item.matchId, {
                id: item.matchId,
                label: item.matchLabel,
              });
            }
          }
          setMatches(Array.from(uniqueMatches.values()));
        }
      } catch (err) {
        console.error("Failed to fetch matches:", err);
      } finally {
        setIsLoadingMatches(false);
      }
    }

    fetchMatches();
  }, []);

  const handleGenerate = async () => {
    if (!selectedMatchId) {
      toast.error("Select a match first");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/settings/ia-features/playground", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          match_id: parseInt(selectedMatchId, 10),
          temperature,
          model: model || undefined,
        }),
      });

      const json = await response.json();

      if (response.status === 429) {
        setRateLimit({
          remaining: 0,
          resetAt: json.rate_limit?.reset_at || "",
        });
        toast.error("Rate limit exceeded (10 calls/hour)");
        return;
      }

      if (!response.ok) {
        throw new Error(json.error || json.detail || `Error ${response.status}`);
      }

      const parsed = parsePlaygroundResponse(json);

      if (parsed) {
        setResult(parsed);
        setRateLimit({
          remaining: parsed.rateLimit.remaining,
          resetAt: parsed.rateLimit.resetAt,
        });
        toast.success("Narrative generated");
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      toast.error("Failed to generate narrative");
    } finally {
      setIsGenerating(false);
    }
  };

  const resetMinutes = rateLimit?.resetAt
    ? Math.max(0, Math.ceil((new Date(rateLimit.resetAt).getTime() - Date.now()) / 60000))
    : null;

  return (
    <div className="bg-surface rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2">
        <FlaskConical className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">LLM Playground</span>
        <Badge variant="secondary" className="text-xs">
          Gemini Only
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Match Selector */}
        <div className="space-y-2">
          <Label className="text-xs">Match</Label>
          <Select
            value={selectedMatchId}
            onValueChange={setSelectedMatchId}
            disabled={isLoadingMatches || isGenerating}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder={isLoadingMatches ? "Loading..." : "Select match"} />
            </SelectTrigger>
            <SelectContent>
              {matches.map((m) => (
                <SelectItem key={m.id} value={String(m.id)}>
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Temperature */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Temperature</Label>
            <span className="text-xs font-mono text-muted-foreground">
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
            disabled={isGenerating}
            className="w-full h-2 cursor-pointer"
          />
        </div>

        {/* Model Selector */}
        <div className="space-y-2">
          <Label className="text-xs">Model</Label>
          <Select
            value={model}
            onValueChange={setModel}
            disabled={isGenerating}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {config?.availableModels.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.displayName}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Warning and Generate Button */}
      <div className="flex items-center justify-between pt-2">
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <AlertTriangle className="h-3.5 w-3.5" />
            <span>~$0.001/call</span>
          </div>
          {rateLimit && (
            <div className="flex items-center gap-1">
              <span>{rateLimit.remaining}/10 restantes</span>
              {resetMinutes !== null && resetMinutes > 0 && (
                <span className="text-muted-foreground/70">
                  (reset en {resetMinutes} min)
                </span>
              )}
            </div>
          )}
        </div>
        <Button
          onClick={handleGenerate}
          disabled={!selectedMatchId || isGenerating || (rateLimit?.remaining === 0)}
          size="sm"
        >
          {isGenerating ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Generating...
            </>
          ) : (
            "Generate Narrative"
          )}
        </Button>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-[var(--status-warning-text)] bg-[var(--status-warning-bg)] rounded-md p-3">
          <AlertTriangle className="h-4 w-4 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-3 bg-muted/30 rounded-lg p-4">
          {/* Title */}
          <h4 className="font-medium">{result.narrative.title}</h4>

          {/* Body */}
          <div className="text-sm whitespace-pre-wrap leading-relaxed">
            {result.narrative.body}
          </div>

          {/* Metrics */}
          <div className="flex items-center gap-4 pt-2 border-t border-border text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" />
              <span>{formatDuration(result.metrics.latencyMs)}</span>
            </div>
            <div className="flex items-center gap-1">
              <Hash className="h-3.5 w-3.5" />
              <span>
                {formatTokens(result.metrics.tokensIn)}/{formatTokens(result.metrics.tokensOut)} tokens
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Coins className="h-3.5 w-3.5" />
              <span>${result.metrics.costUsd.toFixed(4)}</span>
            </div>
            <Badge variant="outline" className="text-xs">
              {result.modelUsed}
            </Badge>
          </div>
        </div>
      )}
    </div>
  );
}
