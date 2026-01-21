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
    className: "bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30",
  },
  warning: {
    label: "Warning",
    variant: "default",
    icon: AlertTriangle,
    className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/30",
  },
  failing: {
    label: "Failing",
    variant: "destructive",
    icon: XCircle,
    className: "bg-red-500/20 text-red-400 border-red-500/30 hover:bg-red-500/30",
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
