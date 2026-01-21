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
    className: "bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30",
  },
  warning: {
    label: "Warning",
    icon: AlertTriangle,
    className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/30",
  },
  stale: {
    label: "Stale",
    icon: Clock,
    className: "bg-gray-500/20 text-gray-400 border-gray-500/30 hover:bg-gray-500/30",
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
