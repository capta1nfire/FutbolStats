"use client";

import { cn } from "@/lib/utils";
import { MatchStatus } from "@/lib/types";
import {
  Calendar,
  Play,
  Coffee,
  XCircle,
  AlertTriangle,
  LucideIcon,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface StatusDotProps {
  status: MatchStatus;
  className?: string;
  /** Show label text next to icon */
  showLabel?: boolean;
}

interface StatusConfig {
  icon: LucideIcon | null;
  color: string;
  label: string;
}

const statusConfig: Record<MatchStatus, StatusConfig> = {
  scheduled: { icon: Calendar, color: "text-primary", label: "Scheduled" },
  live: { icon: Play, color: "text-success animate-pulse", label: "Live" },
  ht: { icon: Coffee, color: "text-warning", label: "Half Time" },
  ft: { icon: null, color: "text-muted-foreground", label: "Final" },
  postponed: { icon: AlertTriangle, color: "text-warning", label: "Postponed" },
  cancelled: { icon: XCircle, color: "text-error", label: "Cancelled" },
};

export function StatusDot({ status, className, showLabel = false }: StatusDotProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  const content = (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      {Icon && <Icon className={cn("h-4 w-4", config.color)} />}
      {(showLabel || !Icon) && (
        <span className={cn("text-xs", config.color)}>{config.label}</span>
      )}
    </span>
  );

  // Only wrap in tooltip when label is not shown AND icon exists
  if (showLabel || !Icon) {
    return content;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          {content}
        </TooltipTrigger>
        <TooltipContent side="top" sideOffset={8}>
          <p>{config.label}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
