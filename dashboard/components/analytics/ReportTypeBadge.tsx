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
    className: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  },
  prediction_accuracy: {
    icon: Target,
    className: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  },
  system_metrics: {
    icon: Activity,
    className: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  },
  api_usage: {
    icon: Zap,
    className: "bg-orange-500/20 text-orange-400 border-orange-500/30",
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
