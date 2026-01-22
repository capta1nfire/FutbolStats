"use client";

import { AuditSeverity } from "@/lib/types";
import { SeverityBars, SeverityLevel } from "@/components/ui/severity-bars";
import { cn } from "@/lib/utils";

interface AuditSeverityBadgeProps {
  severity: AuditSeverity;
  /** Show text label next to bars */
  showLabel?: boolean;
  className?: string;
}

/** Map audit severity to SeverityBars level */
const severityToLevel: Record<AuditSeverity, SeverityLevel> = {
  info: 1,    // Low - green
  warning: 2, // Medium - yellow
  error: 4,   // Very High - red
};

/** Map audit severity to display label */
const severityLabels: Record<AuditSeverity, string> = {
  info: "Info",
  warning: "Warning",
  error: "Error",
};

/**
 * Audit severity indicator using UniFi-style bars
 */
export function AuditSeverityBadge({
  severity,
  showLabel = true,
  className,
}: AuditSeverityBadgeProps) {
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
