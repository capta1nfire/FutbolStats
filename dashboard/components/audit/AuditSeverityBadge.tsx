"use client";

import { Badge } from "@/components/ui/badge";
import { AuditSeverity } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Info, AlertTriangle, XCircle } from "lucide-react";

interface AuditSeverityBadgeProps {
  severity: AuditSeverity;
  showIcon?: boolean;
  className?: string;
}

const severityConfig: Record<
  AuditSeverity,
  { label: string; icon: typeof Info; className: string }
> = {
  info: {
    label: "Info",
    icon: Info,
    className: "bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30",
  },
  warning: {
    label: "Warning",
    icon: AlertTriangle,
    className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/30",
  },
  error: {
    label: "Error",
    icon: XCircle,
    className: "bg-red-500/20 text-red-400 border-red-500/30 hover:bg-red-500/30",
  },
};

export function AuditSeverityBadge({
  severity,
  showIcon = true,
  className,
}: AuditSeverityBadgeProps) {
  const config = severityConfig[severity];
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
