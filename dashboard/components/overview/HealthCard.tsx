"use client";

import { HealthCard as HealthCardType, HealthStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface HealthCardProps {
  card: HealthCardType;
  className?: string;
}

const statusColors: Record<HealthStatus, string> = {
  healthy: "border-green-500/30 bg-green-500/5",
  warning: "border-yellow-500/30 bg-yellow-500/5",
  critical: "border-red-500/30 bg-red-500/5",
};

const statusDotColors: Record<HealthStatus, string> = {
  healthy: "bg-green-500",
  warning: "bg-yellow-500",
  critical: "bg-red-500",
};

const trendIcons = {
  up: TrendingUp,
  down: TrendingDown,
  stable: Minus,
};

const trendColors = {
  up: "text-green-500",
  down: "text-red-500",
  stable: "text-muted-foreground",
};

export function HealthCard({ card, className }: HealthCardProps) {
  const TrendIcon = card.trend ? trendIcons[card.trend] : null;

  return (
    <div
      className={cn(
        "rounded-lg border p-4 transition-colors",
        statusColors[card.status],
        className
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "h-2 w-2 rounded-full",
              statusDotColors[card.status]
            )}
          />
          <span className="text-sm font-medium text-foreground">
            {card.title}
          </span>
        </div>
        {TrendIcon && (
          <TrendIcon
            className={cn("h-4 w-4", trendColors[card.trend!])}
            aria-label={`Trend: ${card.trend}`}
          />
        )}
      </div>

      <div className="text-2xl font-bold text-foreground mb-1">
        {card.value}
      </div>

      {card.subtitle && (
        <div className="text-xs text-muted-foreground">{card.subtitle}</div>
      )}
    </div>
  );
}
