"use client";

import { OverviewTab } from "@/lib/overview-drawer";
import { AlertTriangle, AlertCircle, Clock } from "lucide-react";
import { useActiveIncidentsApi } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";

interface OverviewDrawerIncidentsProps {
  tab: OverviewTab;
}

export function OverviewDrawerIncidents({ tab }: OverviewDrawerIncidentsProps) {
  if (tab === "timeline") {
    return <IncidentsTimelineTab />;
  }
  return <IncidentsSummaryTab />;
}

function IncidentsSummaryTab() {
  const { incidents, isDegraded, isLoading } = useActiveIncidentsApi();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader size="sm" />
      </div>
    );
  }

  if (!incidents || isDegraded) {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        Incidents data unavailable
      </div>
    );
  }

  const severityColors = {
    critical: "bg-[var(--status-error-bg)] border-[var(--status-error-border)] text-[var(--status-error-text)]",
    high: "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]",
    medium: "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]",
    low: "bg-[var(--status-info-bg)] border-[var(--status-info-border)] text-[var(--status-info-text)]",
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <AlertTriangle className="h-4 w-4 text-primary" />
        <span>Active Incidents ({incidents.length})</span>
      </div>

      {incidents.length === 0 ? (
        <div className="bg-[var(--status-success-bg)] rounded-lg p-4 text-center">
          <p className="text-sm text-[var(--status-success-text)]">No active incidents</p>
        </div>
      ) : (
        <div className="space-y-2">
          {incidents.slice(0, 10).map((incident) => (
            <div
              key={incident.id}
              className={cn(
                "rounded-lg p-3 border",
                severityColors[incident.severity as keyof typeof severityColors] ?? severityColors.medium
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {incident.title}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {incident.type}
                  </p>
                </div>
                <span className="text-xs shrink-0 flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {new Date(incident.createdAt).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {incidents.length > 10 && (
        <p className="text-xs text-muted-foreground text-center">
          +{incidents.length - 10} more incidents
        </p>
      )}
    </div>
  );
}

function IncidentsTimelineTab() {
  return (
    <div className="p-4">
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <AlertCircle className="h-8 w-8 text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          Incidents timeline coming soon
        </p>
      </div>
    </div>
  );
}
