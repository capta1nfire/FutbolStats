"use client";

import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { History, Loader2, AlertTriangle, ExternalLink, RefreshCw } from "lucide-react";
import { parseCallHistory, CallHistoryResponse } from "@/lib/api/settings";

function formatDate(isoString: string | null): string {
  if (!isoString) return "-";
  const date = new Date(isoString);
  return date.toLocaleDateString("es-ES", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatTokens(tokensIn: number, tokensOut: number): string {
  const formatK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
  return `${formatK(tokensIn)}\u2192${formatK(tokensOut)}`;
}

function formatCost(usd: number): string {
  return `$${usd.toFixed(4)}`;
}

export function CallHistoryList() {
  const [data, setData] = useState<CallHistoryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/settings/ia-features/call-history?limit=10");
      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      const json = await response.json();
      const parsed = parseCallHistory(json);

      if (parsed) {
        setData(parsed);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  if (isLoading) {
    return (
      <div className="bg-surface rounded-lg p-4">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Loading call history...</span>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-surface rounded-lg p-4">
        <div className="flex items-center gap-2 text-[var(--status-warning-text)]">
          <AlertTriangle className="h-4 w-4" />
          <span className="text-sm">Failed to load call history</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Call History</span>
          <Badge variant="secondary" className="text-xs">
            {data.total} total
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={fetchHistory}
          className="h-7 px-2"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
      </div>

      {data.items.length === 0 ? (
        <p className="text-sm text-muted-foreground py-4 text-center">
          No narrative generations yet
        </p>
      ) : (
        <div className="space-y-2">
          {data.items.map((item) => (
            <div
              key={`${item.matchId}-${item.generatedAt}`}
              className="flex items-center justify-between py-2 border-b border-border last:border-0"
            >
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate">{item.matchLabel}</div>
                <div className="text-xs text-muted-foreground flex items-center gap-2">
                  <span>{formatDate(item.generatedAt)}</span>
                  {item.model && (
                    <>
                      <span className="text-muted-foreground/50">|</span>
                      <span>{item.model}</span>
                    </>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="font-mono text-muted-foreground">
                  {formatTokens(item.tokensIn, item.tokensOut)}
                </span>
                <span className="font-mono text-muted-foreground">
                  {formatCost(item.costUsd)}
                </span>
                <Badge
                  variant="outline"
                  className={
                    item.status === "success"
                      ? "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]"
                      : "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]"
                  }
                >
                  {item.status}
                </Badge>
                <a
                  href={item.auditUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors"
                  title="View audit"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
