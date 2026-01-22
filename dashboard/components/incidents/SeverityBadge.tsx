"use client";

import { IncidentSeverity } from "@/lib/types";
import { SeverityBars, SeverityLevel } from "@/components/ui/severity-bars";
import { cn } from "@/lib/utils";

interface SeverityBadgeProps {
  severity: IncidentSeverity;
  /** Show text label next to bars */
  showLabel?: boolean;
  className?: string;
}

/** Map incident severity to SeverityBars level */
const severityToLevel: Record<IncidentSeverity, SeverityLevel> = {
  info: 1,      // Low - green
  warning: 2,   // Medium - yellow
  critical: 4,  // Very High - red
};

/** Map incident severity to display label */
const severityLabels: Record<IncidentSeverity, string> = {
  info: "Info",
  warning: "Warning",
  critical: "Critical",
};

/**
 * Incident severity indicator using UniFi-style bars
 */
export function SeverityBadge({
  severity,
  showLabel = true,
  className,
}: SeverityBadgeProps) {
  const level = severityToLevel[severity];
  const label = severityLabels[severity];

  return (
    <SeverityBars
      level={level}
      label={label}
      showLabel={showLabel}
      className={cn(className)}
    />
  );
}
