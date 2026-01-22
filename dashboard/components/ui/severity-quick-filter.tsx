"use client";

import * as React from "react";
import { SeverityLevel } from "./severity-bars";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/** Configuration for each severity level */
const SEVERITY_CONFIG: Record<SeverityLevel, { color: string; label: string }> = {
  1: { color: "#37be5f", label: "Low" },
  2: { color: "#dfc116", label: "Medium" },
  3: { color: "#e79613", label: "High" },
  4: { color: "#ee6368", label: "Very High" },
};

const INACTIVE_COLOR = "#282b2f";

interface SeverityToggleProps {
  level: SeverityLevel;
  active: boolean;
  count?: number;
  onClick: () => void;
  label?: string;
}

/**
 * Individual severity toggle button with bars
 */
function SeverityToggle({
  level,
  active,
  count,
  onClick,
  label,
}: SeverityToggleProps) {
  const config = SEVERITY_CONFIG[level];
  const displayLabel = label || config.label;

  return (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={onClick}
            aria-pressed={active}
            aria-label={`Filter by ${displayLabel}`}
            className={cn(
              "flex items-center gap-1.5 px-2 py-1.5 rounded-md transition-all",
              "border",
              active
                ? "border-border bg-surface"
                : "border-transparent hover:bg-surface/50"
            )}
          >
            {/* 4 bars */}
            <div className="flex items-center gap-[2px]">
              {[1, 2, 3, 4].map((barIndex) => (
                <span
                  key={barIndex}
                  className="w-[8px] h-[4px] rounded-[2px] transition-colors"
                  style={{
                    backgroundColor:
                      barIndex <= level
                        ? active
                          ? config.color
                          : `${config.color}60` // Dimmed when not active
                        : INACTIVE_COLOR,
                  }}
                />
              ))}
            </div>
            {/* Count badge */}
            {count !== undefined && (
              <span
                className={cn(
                  "text-xs font-medium tabular-nums",
                  active ? "text-foreground" : "text-muted-foreground"
                )}
              >
                {count}
              </span>
            )}
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          <p>{displayLabel}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export interface SeverityFilterItem {
  level: SeverityLevel;
  id: string;
  label?: string;
  count?: number;
}

interface SeverityQuickFilterProps {
  /** Available severity levels to show */
  items: SeverityFilterItem[];
  /** Currently selected severity IDs */
  selectedIds: string[];
  /** Callback when a severity is toggled */
  onToggle: (id: string) => void;
  /** Additional className */
  className?: string;
}

/**
 * UniFi-style severity quick filter with bar toggles.
 * Supports multiselect (1..n levels can be active).
 */
export function SeverityQuickFilter({
  items,
  selectedIds,
  onToggle,
  className,
}: SeverityQuickFilterProps) {
  return (
    <div className={cn("flex items-center gap-1", className)}>
      {items.map((item) => (
        <SeverityToggle
          key={item.id}
          level={item.level}
          active={selectedIds.includes(item.id)}
          count={item.count}
          onClick={() => onToggle(item.id)}
          label={item.label}
        />
      ))}
    </div>
  );
}
