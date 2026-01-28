"use client";

import { Badge } from "@/components/ui/badge";
import { DataQualityStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";

interface DataQualityStatusBadgeProps {
  status: DataQualityStatus;
  showIcon?: boolean;
  className?: string;
}

const statusConfig: Record<
  DataQualityStatus,
  { label: string; variant: "default" | "secondary" | "destructive" | "outline"; icon: typeof CheckCircle2; className: string }
> = {
  passing: {
    label: "Passing",
    variant: "default",
    icon: CheckCircle2,
    className: "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]",
  },
  warning: {
    label: "Warning",
    variant: "default",
    icon: AlertTriangle,
    className: "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]",
  },
  failing: {
    label: "Failing",
    variant: "destructive",
    icon: XCircle,
    className: "bg-[var(--status-error-bg)] text-[var(--status-error-text)] border-[var(--status-error-border)]",
  },
};

export function DataQualityStatusBadge({
  status,
  showIcon = true,
  className,
}: DataQualityStatusBadgeProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <Badge
      variant={config.variant}
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
