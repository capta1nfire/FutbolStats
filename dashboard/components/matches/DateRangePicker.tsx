"use client";

import { useState, useMemo } from "react";
import { DateRange } from "react-day-picker";
import { Calendar } from "@/components/ui/calendar";
import { cn } from "@/lib/utils";

/** Quick preset options for time ranges */
const PRESETS = [
  { label: "30m", minutes: 30 },
  { label: "1h", minutes: 60 },
  { label: "1D", minutes: 60 * 24 },
  { label: "1W", minutes: 60 * 24 * 7 },
  { label: "1M", minutes: 60 * 24 * 30 },
] as const;

export interface DateRangeValue {
  from: Date;
  to: Date;
}

interface DateRangePickerProps {
  /** Current date range */
  value?: DateRangeValue;
  /** Callback when range changes */
  onChange: (range: DateRangeValue) => void;
  /** Whether to show future dates (upcoming) or past dates (finished) */
  mode?: "future" | "past";
  /** Additional class names */
  className?: string;
}

/**
 * Date Range Picker
 *
 * Features:
 * - Quick presets (30m, 1h, 1D, 1W, 1M)
 * - Calendar with range selection
 * - Styled to match UniFi dark theme
 */
export function DateRangePicker({
  value,
  onChange,
  mode = "future",
  className,
}: DateRangePickerProps) {
  // Track which preset is active (if any)
  const [activePreset, setActivePreset] = useState<string | null>(null);

  // Convert value to DateRange for calendar
  const dateRange: DateRange | undefined = useMemo(() => {
    if (!value) return undefined;
    return { from: value.from, to: value.to };
  }, [value]);

  // Handle preset click
  const handlePresetClick = (preset: typeof PRESETS[number]) => {
    const now = new Date();
    let from: Date;
    let to: Date;

    if (mode === "future") {
      // Future: from now to +preset
      from = now;
      to = new Date(now.getTime() + preset.minutes * 60 * 1000);
    } else {
      // Past: from -preset to now
      from = new Date(now.getTime() - preset.minutes * 60 * 1000);
      to = now;
    }

    setActivePreset(preset.label);
    onChange({ from, to });
  };

  // Handle calendar selection
  const handleCalendarSelect = (range: DateRange | undefined) => {
    if (range?.from && range?.to) {
      setActivePreset(null);
      onChange({ from: range.from, to: range.to });
    } else if (range?.from) {
      // Single date selected - wait for second click
      setActivePreset(null);
    }
  };

  return (
    <div className={cn("space-y-3", className)}>
      {/* Presets row */}
      <div className="flex items-center gap-1 px-1">
        {PRESETS.map((preset) => (
          <button
            key={preset.label}
            onClick={() => handlePresetClick(preset)}
            className={cn(
              "px-2.5 py-1 text-xs font-medium rounded transition-colors",
              activePreset === preset.label
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-surface"
            )}
          >
            {preset.label}
          </button>
        ))}
        {/* Calendar icon indicator (last button) */}
        <div className="ml-auto">
          <div className="w-6 h-6 flex items-center justify-center text-primary">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect width="18" height="18" x="3" y="4" rx="2" ry="2" />
              <line x1="16" x2="16" y1="2" y2="6" />
              <line x1="8" x2="8" y1="2" y2="6" />
              <line x1="3" x2="21" y1="10" y2="10" />
            </svg>
          </div>
        </div>
      </div>

      {/* Calendar */}
      <Calendar
        mode="range"
        selected={dateRange}
        onSelect={handleCalendarSelect}
        numberOfMonths={1}
        className="rounded-lg"
        classNames={{
          months: "flex flex-col",
          month: "space-y-2",
          caption_label: "text-sm font-medium text-foreground",
          nav: "flex items-center justify-between",
          table: "w-full border-collapse",
          head_row: "flex",
          head_cell: "text-muted-foreground w-8 font-normal text-xs",
          row: "flex w-full mt-1",
          cell: cn(
            "relative p-0 text-center text-sm focus-within:relative focus-within:z-20",
            "first:[&:has([aria-selected])]:rounded-l-md last:[&:has([aria-selected])]:rounded-r-md"
          ),
          day: cn(
            "h-8 w-8 p-0 font-normal",
            "aria-selected:opacity-100"
          ),
          day_selected: "bg-primary text-primary-foreground rounded-full",
          day_today: "bg-accent text-accent-foreground rounded-full",
          day_outside: "text-muted-foreground/50",
          day_disabled: "text-muted-foreground/30",
          day_range_middle: "aria-selected:bg-accent aria-selected:text-accent-foreground rounded-none",
          day_range_start: "rounded-l-full",
          day_range_end: "rounded-r-full",
        }}
      />
    </div>
  );
}
