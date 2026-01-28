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
    className: "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)] border-[var(--tag-blue-border)]",
  },
  consistency: {
    label: "Consistency",
    icon: RefreshCcw,
    className: "bg-[var(--tag-purple-bg)] text-[var(--tag-purple-text)] border-[var(--tag-purple-border)]",
  },
  completeness: {
    label: "Completeness",
    icon: CheckSquare,
    className: "bg-[var(--tag-cyan-bg)] text-[var(--tag-cyan-text)] border-[var(--tag-cyan-border)]",
  },
  freshness: {
    label: "Freshness",
    icon: Clock,
    className: "bg-[var(--tag-orange-bg)] text-[var(--tag-orange-text)] border-[var(--tag-orange-border)]",
  },
  odds: {
    label: "Odds",
    icon: TrendingUp,
    className: "bg-[var(--tag-cyan-bg)] text-[var(--tag-cyan-text)] border-[var(--tag-cyan-border)]",
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
