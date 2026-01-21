"use client";

import { cn } from "@/lib/utils";
import { MatchStatus } from "@/lib/types";

interface StatusDotProps {
  status: MatchStatus;
  className?: string;
}

const statusConfig: Record<MatchStatus, { color: string; label: string }> = {
  scheduled: { color: "bg-muted-foreground", label: "Scheduled" },
  live: { color: "bg-success animate-pulse", label: "Live" },
  ht: { color: "bg-warning", label: "HT" },
  ft: { color: "bg-info", label: "FT" },
  postponed: { color: "bg-warning", label: "Postponed" },
  cancelled: { color: "bg-error", label: "Cancelled" },
};

export function StatusDot({ status, className }: StatusDotProps) {
  const config = statusConfig[status];

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span className={cn("w-2 h-2 rounded-full", config.color)} />
      <span className="text-xs text-muted-foreground uppercase">{config.label}</span>
    </div>
  );
}
