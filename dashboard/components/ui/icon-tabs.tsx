"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export interface IconTab {
  /** Unique identifier for the tab */
  id: string;
  /** Icon to display (should be 24x24) */
  icon: ReactNode;
  /** Tooltip text */
  label: string;
  /** Whether the tab is disabled */
  disabled?: boolean;
}

interface IconTabsProps {
  /** Array of tab definitions */
  tabs: IconTab[];
  /** Currently selected tab id */
  value: string;
  /** Called when a tab is selected */
  onValueChange: (value: string) => void;
  /** Additional class names for the container */
  className?: string;
}

/**
 * Icon Tabs (UniFi style)
 *
 * A horizontal tab bar with icon-only tabs and tooltips.
 * Based on UniFi Network drawer tabs design.
 *
 * Features:
 * - Icon-only tabs with tooltips
 * - Equal width distribution (flex: 1 1 0%)
 * - Smooth transitions (150ms cubic-bezier)
 * - Focus-visible outline for accessibility
 *
 * Usage:
 * ```tsx
 * <IconTabs
 *   tabs={[
 *     { id: "overview", icon: <Info />, label: "Overview" },
 *     { id: "predictions", icon: <TrendingUp />, label: "Predictions" },
 *   ]}
 *   value={activeTab}
 *   onValueChange={setActiveTab}
 * />
 * ```
 */
export function IconTabs({
  tabs,
  value,
  onValueChange,
  className,
}: IconTabsProps) {
  return (
    <nav
      className={cn(
        "inline-flex items-center bg-surface rounded-lg p-1 min-h-9 h-10",
        className
      )}
      role="tablist"
    >
      {tabs.map((tab) => (
        <IconTabButton
          key={tab.id}
          tab={tab}
          isSelected={value === tab.id}
          onSelect={() => !tab.disabled && onValueChange(tab.id)}
        />
      ))}
    </nav>
  );
}

interface IconTabButtonProps {
  tab: IconTab;
  isSelected: boolean;
  onSelect: () => void;
}

function IconTabButton({ tab, isSelected, onSelect }: IconTabButtonProps) {
  return (
    <div className="flex-1">
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            role="tab"
            aria-selected={isSelected}
            aria-label={tab.label}
            disabled={tab.disabled}
            onClick={onSelect}
            className={cn(
              // Layout
              "inline-flex flex-col items-center justify-center",
              "w-full h-8 min-h-8 p-1 rounded",
              // Transitions
              "transition-[background-color,color] duration-150",
              // Focus state
              "focus-visible:outline focus-visible:outline-1 focus-visible:outline-primary",
              // Default state
              "bg-transparent text-muted-foreground",
              // Hover state (not selected, not disabled)
              !isSelected && !tab.disabled && "hover:text-primary",
              // Selected state
              isSelected && "bg-accent text-primary",
              // Disabled state
              tab.disabled && "text-muted-foreground/50 cursor-not-allowed"
            )}
            style={{
              // Custom cubic-bezier for UniFi feel
              transitionTimingFunction: "cubic-bezier(0.7, 0, 0.3, 1)",
            }}
          >
            <span className="w-6 h-6 flex items-center justify-center [&>svg]:w-5 [&>svg]:h-5">
              {tab.icon}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="top">
          {tab.label}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
