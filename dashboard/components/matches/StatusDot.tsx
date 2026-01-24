"use client";

import { cn } from "@/lib/utils";
import { MatchStatus } from "@/lib/types";
import {
  Calendar,
  Play,
  Coffee,
  CheckCircle,
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
}

interface StatusConfig {
  icon: LucideIcon;
  color: string;
  label: string;
}

const statusConfig: Record<MatchStatus, StatusConfig> = {
  scheduled: { icon: Calendar, color: "text-primary", label: "Scheduled" },
  live: { icon: Play, color: "text-success animate-pulse", label: "Live" },
  ht: { icon: Coffee, color: "text-warning", label: "HT" },
  ft: { icon: CheckCircle, color: "text-info", label: "FT" },
  postponed: { icon: AlertTriangle, color: "text-warning", label: "Postponed" },
  cancelled: { icon: XCircle, color: "text-error", label: "Cancelled" },
};

export function StatusDot({ status, className }: StatusDotProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={cn("flex items-center gap-2", className)}>
            <Icon className={cn("h-4 w-4", config.color)} />
          </div>
        </TooltipTrigger>
        <TooltipContent side="right">
          <p>{config.label}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
