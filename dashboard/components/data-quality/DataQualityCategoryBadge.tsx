"use client";

import { Badge } from "@/components/ui/badge";
import { DataQualityCategory } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  BarChart3,
  RefreshCcw,
  CheckSquare,
  Clock,
  TrendingUp
} from "lucide-react";

interface DataQualityCategoryBadgeProps {
  category: DataQualityCategory;
  showIcon?: boolean;
  className?: string;
}

const categoryConfig: Record<
  DataQualityCategory,
  { label: string; icon: typeof BarChart3; className: string }
> = {
  coverage: {
    label: "Coverage",
    icon: BarChart3,
    className: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  },
  consistency: {
    label: "Consistency",
    icon: RefreshCcw,
    className: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  },
  completeness: {
    label: "Completeness",
    icon: CheckSquare,
    className: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  },
  freshness: {
    label: "Freshness",
    icon: Clock,
    className: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  },
  odds: {
    label: "Odds",
    icon: TrendingUp,
    className: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  },
};

export function DataQualityCategoryBadge({
  category,
  showIcon = true,
  className,
}: DataQualityCategoryBadgeProps) {
  const config = categoryConfig[category];
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
