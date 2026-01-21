"use client";

import { ActiveIncident } from "@/lib/types";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "@/lib/utils";
import { AlertTriangle, AlertCircle, Info } from "lucide-react";
import Link from "next/link";

interface ActiveIncidentsListProps {
  incidents: ActiveIncident[];
  className?: string;
}

const severityIcons = {
  critical: AlertTriangle,
  warning: AlertCircle,
  info: Info,
};

const severityColors = {
  critical: "text-red-500 bg-red-500/10 border-red-500/30",
  warning: "text-yellow-500 bg-yellow-500/10 border-yellow-500/30",
  info: "text-blue-500 bg-blue-500/10 border-blue-500/30",
};

export function ActiveIncidentsList({
  incidents,
  className,
}: ActiveIncidentsListProps) {
  if (incidents.length === 0) {
    return (
      <div className={cn("text-center py-8", className)}>
        <AlertCircle className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No active incidents</p>
        <p className="text-xs text-muted-foreground mt-1">All systems normal</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {incidents.map((incident) => {
        const Icon = severityIcons[incident.severity];

        return (
          <Link
            key={incident.id}
            href={`/incidents?id=${incident.id}`}
            className={cn(
              "block p-3 rounded-lg border transition-colors hover:opacity-80",
              severityColors[incident.severity]
            )}
          >
            <div className="flex items-start gap-3">
              <Icon className="h-4 w-4 shrink-0 mt-0.5" />

              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">
                  {incident.title}
                </div>
                <div className="text-xs opacity-70 mt-1">
                  {formatDistanceToNow(incident.createdAt)}
                </div>
              </div>
            </div>
          </Link>
        );
      })}
    </div>
  );
}
