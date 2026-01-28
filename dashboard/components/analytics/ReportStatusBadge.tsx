"use client";

import { Badge } from "@/components/ui/badge";
import { AnalyticsReportStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { CheckCircle2, AlertTriangle, Clock } from "lucide-react";

interface ReportStatusBadgeProps {
  status: AnalyticsReportStatus;
  showIcon?: boolean;
  className?: string;
}

const statusConfig: Record<
  AnalyticsReportStatus,
  { label: string; icon: typeof CheckCircle2; className: string }
> = {
  ok: {
    label: "OK",
    icon: CheckCircle2,
    className: "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]",
  },
  warning: {
    label: "Warning",
    icon: AlertTriangle,
    className: "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]",
  },
  stale: {
    label: "Stale",
    icon: Clock,
    className: "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]",
  },
};

export function ReportStatusBadge({
  status,
  showIcon = true,
  className,
}: ReportStatusBadgeProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1 font-medium",
        config.className,
        className
      )}
    >
      {showIcon && <Icon className="h-3 w-3" />}
      {config.label}
    </Badge>
  );
}
