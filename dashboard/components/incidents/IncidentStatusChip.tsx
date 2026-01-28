"use client";

import { Badge } from "@/components/ui/badge";
import { Circle, CheckCircle2, Clock } from "lucide-react";
import { IncidentStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

interface IncidentStatusChipProps {
  status: IncidentStatus;
  className?: string;
}

const statusConfig: Record<
  IncidentStatus,
  { label: string; icon: typeof Circle; className: string }
> = {
  active: {
    label: "Active",
    icon: Circle,
    className: "bg-[var(--status-error-bg)] text-[var(--status-error-text)] border-[var(--status-error-border)]",
  },
  acknowledged: {
    label: "Acknowledged",
    icon: Clock,
    className: "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]",
  },
  resolved: {
    label: "Resolved",
    icon: CheckCircle2,
    className: "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]",
  },
};

export function IncidentStatusChip({ status, className }: IncidentStatusChipProps) {
  const config = statusConfig[status];
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
