"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { Sparkles } from "lucide-react";
import { useOpsOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";

interface OverviewDrawerSotaProps {
  tab: OverviewTab;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars -- tab prop reserved for future use
export function OverviewDrawerSota({ tab }: OverviewDrawerSotaProps) {
  // Only summary tab for now
  return <SotaSummaryTab />;
}

const statusColors = {
  ok: "bg-[var(--status-success-text)]",
  warn: "bg-[var(--status-warning-text)]",
  red: "bg-[var(--status-error-text)]",
  pending: "bg-[var(--status-info-text)]",
  unavailable: "bg-muted",
};

const keyLabels: Record<string, string> = {
  understat: "Understat xG",
  weather: "Weather",
  venue_geo: "Venue Geo",
  team_profiles: "Team Profiles",
  sofascore_xi: "Sofascore XI",
};

function SotaSummaryTab() {
  const { sotaEnrichment, isSotaEnrichmentDegraded, isLoading } = useOpsOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!sotaEnrichment || isSotaEnrichmentDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        SOTA enrichment data unavailable
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Sparkles className="h-4 w-4 text-primary" />
        <span>SOTA Enrichment Status: {sotaEnrichment.status}</span>
      </div>

      {/* Items */}
      {sotaEnrichment.items.map((item) => (
        <div key={item.key} className="bg-muted/30 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-foreground">
              {keyLabels[item.key] ?? item.key}
            </span>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-foreground tabular-nums">
                {item.with_data}/{item.total}
              </span>
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  statusColors[item.status] ?? statusColors.unavailable
                )}
                title={item.status}
              />
            </div>
          </div>
          {/* Progress bar */}
          <div className="h-1.5 bg-muted rounded-full overflow-hidden mt-2">
            <div
              className={cn(
                "h-full transition-all",
                statusColors[item.status] ?? statusColors.unavailable
              )}
              style={{
                width: `${item.coverage_pct}%`
              }}
            />
          </div>
          {/* Optional note or error */}
          {(item.note || item.error) && (
            <p className="text-xs text-muted-foreground mt-1">
              {item.error ?? item.note}
            </p>
          )}
        </div>
      ))}

      {/* Generated at */}
      {sotaEnrichment.generated_at && (
        <p className="text-xs text-muted-foreground pt-2 border-t border-border">
          Generated: {new Date(sotaEnrichment.generated_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}
