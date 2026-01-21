"use client";

import { ReactNode, useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

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
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Handle tooltip show/hide with animation
  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => {
      setShowTooltip(true);
      // Small delay to trigger CSS transition
      requestAnimationFrame(() => {
        setTooltipVisible(true);
      });
    }, 200); // Delay before showing tooltip
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setTooltipVisible(false);
    // Wait for fade out animation
    setTimeout(() => {
      setShowTooltip(false);
    }, 100);
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="relative flex-1">
      <button
        ref={buttonRef}
        role="tab"
        aria-selected={isSelected}
        aria-label={tab.label}
        disabled={tab.disabled}
        onClick={onSelect}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onFocus={handleMouseEnter}
        onBlur={handleMouseLeave}
        className={cn(
          // Layout
          "inline-flex flex-col items-center justify-center",
          "flex-1 w-full h-8 min-h-8 p-1 rounded",
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

      {/* Tooltip */}
      {showTooltip && (
        <div
          className={cn(
            "absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50",
            "px-4 py-2 rounded text-[11px] leading-4 font-normal",
            "bg-[#2b2b2b] text-[#fafafa]",
            "shadow-[0_8px_24px_rgb(0,0,0),0_0_1px_rgba(249,250,250,0.08)]",
            "whitespace-nowrap pointer-events-none",
            "transition-opacity duration-100",
            tooltipVisible ? "opacity-100" : "opacity-0"
          )}
          style={{
            transitionTimingFunction: "cubic-bezier(1, 0, 0.6, 0.8)",
          }}
          role="tooltip"
        >
          {tab.label}
          {/* Tooltip arrow */}
          <span
            className={cn(
              "absolute left-1/2 -translate-x-1/2 -bottom-1.5",
              "w-3 h-3 bg-[#2b2b2b] rounded-tl",
              "rotate-[-135deg]"
            )}
          />
        </div>
      )}
    </div>
  );
}
