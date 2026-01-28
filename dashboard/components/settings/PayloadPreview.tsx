"use client";

import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Eye, Copy, Check, Loader2, AlertTriangle, FileJson } from "lucide-react";
import { parsePayloadPreview, PayloadPreviewResponse, CallHistoryItem, parseCallHistory } from "@/lib/api/settings";
import { toast } from "sonner";

interface MatchOption {
  id: number;
  label: string;
  status: string;
}

export function PayloadPreview() {
  const [matches, setMatches] = useState<MatchOption[]>([]);
  const [selectedMatchId, setSelectedMatchId] = useState<string>("");
  const [preview, setPreview] = useState<PayloadPreviewResponse | null>(null);
  const [isLoadingMatches, setIsLoadingMatches] = useState(true);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  // Fetch recent matches from call history
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
          // Extract unique matches
          const uniqueMatches = new Map<number, MatchOption>();
          for (const item of parsed.items) {
            if (!uniqueMatches.has(item.matchId)) {
              uniqueMatches.set(item.matchId, {
                id: item.matchId,
                label: item.matchLabel,
                status: "FT",
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

  // Fetch preview when match changes
  useEffect(() => {
    if (!selectedMatchId) {
      setPreview(null);
      return;
    }

    async function fetchPreview() {
      setIsLoadingPreview(true);
      setError(null);

      try {
        const response = await fetch(`/api/settings/ia-features/preview/${selectedMatchId}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch: ${response.status}`);
        }

        const json = await response.json();
        const parsed = parsePayloadPreview(json);

        if (parsed) {
          setPreview(parsed);
        } else {
          throw new Error("Invalid response format");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
        setPreview(null);
      } finally {
        setIsLoadingPreview(false);
      }
    }

    fetchPreview();
  }, [selectedMatchId]);

  const handleCopy = async (content: string, label: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      toast.success(`${label} copied to clipboard`);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Failed to copy");
    }
  };

  return (
    <div className="bg-surface rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2">
        <Eye className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Live Preview</span>
      </div>

      {/* Match Selector */}
      <div className="space-y-2">
        <Select
          value={selectedMatchId}
          onValueChange={setSelectedMatchId}
          disabled={isLoadingMatches}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={isLoadingMatches ? "Loading matches..." : "Select a match"} />
          </SelectTrigger>
          <SelectContent>
            {matches.map((match) => (
              <SelectItem key={match.id} value={String(match.id)}>
                {match.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Select a completed match to preview the LLM payload
        </p>
      </div>

      {/* Loading State */}
      {isLoadingPreview && (
        <div className="flex items-center gap-2 py-4">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Loading preview...</span>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="flex items-center gap-2 text-[var(--status-warning-text)]">
          <AlertTriangle className="h-4 w-4" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Preview Content */}
      {preview && !isLoadingPreview && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Badge variant="outline">{preview.status}</Badge>
            <div className="flex items-center gap-2">
              <Button
                variant={showRaw ? "default" : "outline"}
                size="sm"
                onClick={() => setShowRaw(!showRaw)}
                className="h-7 text-xs"
              >
                <FileJson className="h-3.5 w-3.5 mr-1" />
                {showRaw ? "Hide Raw" : "View Raw"}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleCopy(preview.promptPreview, "Prompt")}
                className="h-7 px-2"
              >
                {copied ? (
                  <Check className="h-3.5 w-3.5 text-[var(--status-success-text)]" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </Button>
            </div>
          </div>

          {showRaw ? (
            <pre className="text-xs bg-muted/50 rounded-md p-3 overflow-x-auto max-h-[400px] overflow-y-auto whitespace-pre-wrap font-mono">
              {preview.promptPreview}
            </pre>
          ) : (
            <Accordion type="multiple" className="w-full">
              <AccordionItem value="match" className="border-border">
                <AccordionTrigger className="py-2 text-sm hover:no-underline">
                  Match Data
                </AccordionTrigger>
                <AccordionContent>
                  <div className="text-xs space-y-1 font-mono bg-muted/30 rounded p-2">
                    <div><span className="text-muted-foreground">home_team:</span> {preview.matchData.homeTeam}</div>
                    <div><span className="text-muted-foreground">away_team:</span> {preview.matchData.awayTeam}</div>
                    <div><span className="text-muted-foreground">score:</span> {preview.matchData.homeGoals} - {preview.matchData.awayGoals}</div>
                    <div><span className="text-muted-foreground">league:</span> {preview.matchData.leagueName}</div>
                    <div><span className="text-muted-foreground">date:</span> {preview.matchData.date}</div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="stats" className="border-border">
                <AccordionTrigger className="py-2 text-sm hover:no-underline">
                  Stats
                </AccordionTrigger>
                <AccordionContent>
                  <pre className="text-xs bg-muted/30 rounded p-2 overflow-x-auto font-mono">
                    {JSON.stringify(preview.matchData.stats, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="prediction" className="border-border">
                <AccordionTrigger className="py-2 text-sm hover:no-underline">
                  Prediction
                </AccordionTrigger>
                <AccordionContent>
                  <pre className="text-xs bg-muted/30 rounded p-2 overflow-x-auto font-mono">
                    {JSON.stringify(preview.matchData.prediction, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="events" className="border-border">
                <AccordionTrigger className="py-2 text-sm hover:no-underline">
                  Events ({preview.matchData.events.length})
                </AccordionTrigger>
                <AccordionContent>
                  <pre className="text-xs bg-muted/30 rounded p-2 overflow-x-auto font-mono max-h-[200px] overflow-y-auto">
                    {JSON.stringify(preview.matchData.events, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="odds" className="border-none">
                <AccordionTrigger className="py-2 text-sm hover:no-underline">
                  Market Odds
                </AccordionTrigger>
                <AccordionContent>
                  <pre className="text-xs bg-muted/30 rounded p-2 overflow-x-auto font-mono">
                    {JSON.stringify(preview.matchData.marketOdds, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          )}
        </div>
      )}
    </div>
  );
}
