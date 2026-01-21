"use client";

import { Badge } from "@/components/ui/badge";
import { AlertTriangle, AlertCircle, Info } from "lucide-react";
import { IncidentSeverity } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SeverityBadgeProps {
  severity: IncidentSeverity;
  className?: string;
}

const severityConfig: Record<
  IncidentSeverity,
  { label: string; icon: typeof AlertTriangle; className: string }
> = {
  critical: {
    label: "Critical",
    icon: AlertTriangle,
    className: "bg-red-500/20 text-red-400 border-red-500/30",
  },
  warning: {
    label: "Warning",
    icon: AlertCircle,
    className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  },
  info: {
    label: "Info",
    icon: Info,
    className: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  },
};

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  const config = severityConfig[severity];
  const Icon = config.icon;

  return (
    <Badge
      variant="outline"
      className={cn("gap-1 font-medium", config.className, className)}
    >
      <Icon className="h-3 w-3" />
      {config.label}
    </Badge>
  );
}
