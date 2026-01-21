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
    className: "bg-red-500/20 text-red-400 border-red-500/30",
  },
  acknowledged: {
    label: "Acknowledged",
    icon: Clock,
    className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  },
  resolved: {
    label: "Resolved",
    icon: CheckCircle2,
    className: "bg-green-500/20 text-green-400 border-green-500/30",
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
