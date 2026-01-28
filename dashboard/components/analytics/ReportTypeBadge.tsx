"use client";

import { Badge } from "@/components/ui/badge";
import { AnalyticsReportType, ANALYTICS_REPORT_TYPE_LABELS } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Brain, Target, Activity, Zap } from "lucide-react";

interface ReportTypeBadgeProps {
  type: AnalyticsReportType;
  showIcon?: boolean;
  className?: string;
}

const typeConfig: Record<
  AnalyticsReportType,
  { icon: typeof Brain; className: string }
> = {
  model_performance: {
    icon: Brain,
    className: "bg-[var(--tag-purple-bg)] text-[var(--tag-purple-text)] border-[var(--tag-purple-border)]",
  },
  prediction_accuracy: {
    icon: Target,
    className: "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)] border-[var(--tag-blue-border)]",
  },
  system_metrics: {
    icon: Activity,
    className: "bg-[var(--tag-cyan-bg)] text-[var(--tag-cyan-text)] border-[var(--tag-cyan-border)]",
  },
  api_usage: {
    icon: Zap,
    className: "bg-[var(--tag-orange-bg)] text-[var(--tag-orange-text)] border-[var(--tag-orange-border)]",
  },
};

export function ReportTypeBadge({
  type,
  showIcon = true,
  className,
}: ReportTypeBadgeProps) {
  const config = typeConfig[type];
  const Icon = config.icon;
  const label = ANALYTICS_REPORT_TYPE_LABELS[type];

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
      {label}
    </Badge>
  );
}
