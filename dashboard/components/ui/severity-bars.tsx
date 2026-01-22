"use client";

import * as React from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/** Severity level 1-4 */
export type SeverityLevel = 1 | 2 | 3 | 4;

/** Map level to color and label */
const SEVERITY_CONFIG: Record<SeverityLevel, { color: string; label: string }> = {
  1: { color: "#37be5f", label: "Low" },
  2: { color: "#dfc116", label: "Medium" },
  3: { color: "#e79613", label: "High" },
  4: { color: "#ee6368", label: "Very High" },
};

const INACTIVE_COLOR = "#282b2f";

interface SeverityBarsProps {
  /** Severity level 1-4 (number of active bars) */
  level: SeverityLevel;
  /** Optional label override for tooltip/aria */
  label?: string;
  /** Show text label next to bars */
  showLabel?: boolean;
  /** Additional className */
  className?: string;
}

/**
 * UniFi-style severity indicator with 4 horizontal bars.
 * - 1 bar = Low (green)
 * - 2 bars = Medium (yellow)
 * - 3 bars = High (orange)
 * - 4 bars = Very High (red)
 */
export function SeverityBars({
  level,
  label,
  showLabel = false,
  className,
}: SeverityBarsProps) {
  const config = SEVERITY_CONFIG[level];
  const displayLabel = label || config.label;
  const ariaLabel = `Severity: ${displayLabel} (${level}/4)`;

  return (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn("flex items-center gap-2", className)}
            role="img"
            aria-label={ariaLabel}
          >
            {/* 4 bars container */}
            <div className="flex items-center gap-[2px]">
              {[1, 2, 3, 4].map((barIndex) => (
                <span
                  key={barIndex}
                  className="w-[8px] h-[4px] rounded-[2px]"
                  style={{
                    backgroundColor:
                      barIndex <= level ? config.color : INACTIVE_COLOR,
                  }}
                />
              ))}
            </div>
            {/* Optional text label */}
            {showLabel && (
              <span className="text-xs text-muted-foreground">{displayLabel}</span>
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p>{displayLabel}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
