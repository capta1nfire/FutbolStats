"use client";

import { cn } from "@/lib/utils";

interface CoverageBarProps {
  percentage: number;
  label?: string;
  className?: string;
}

export function CoverageBar({
  percentage,
  label = "Prediction Coverage",
  className,
}: CoverageBarProps) {
  // Clamp percentage between 0 and 100
  const clampedPct = Math.max(0, Math.min(100, percentage));

  // Determine color based on coverage
  const getBarColor = () => {
    if (clampedPct >= 90) return "bg-[var(--status-success-text)]";
    if (clampedPct >= 70) return "bg-[var(--status-warning-text)]";
    return "bg-[var(--status-error-text)]";
  };

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="text-sm font-bold text-foreground">{clampedPct}%</span>
      </div>

      <div
        className="h-2 w-full rounded-full bg-muted overflow-hidden"
        role="progressbar"
        aria-valuenow={clampedPct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${clampedPct}%`}
      >
        <div
          className={cn("h-full rounded-full transition-all duration-500", getBarColor())}
          style={{ width: `${clampedPct}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-muted-foreground">
        <span>0%</span>
        <span>Target: 95%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
