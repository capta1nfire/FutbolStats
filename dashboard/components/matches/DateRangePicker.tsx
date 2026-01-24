"use client";

import { useMemo } from "react";
import { Calendar } from "@/components/ui/calendar";
import { cn } from "@/lib/utils";
import { useRegion, type LocalDate } from "@/components/providers/RegionProvider";
import { localDateToDate } from "@/lib/region";

/**
 * Date value as LocalDate string (YYYY-MM-DD)
 * No timezone ambiguity - represents a calendar day
 */
export interface DatePickerValue {
  /** Selected date as YYYY-MM-DD string */
  date: LocalDate;
}

interface DateRangePickerProps {
  /** Current selected date (LocalDate string) */
  value?: LocalDate;
  /** Callback when date changes */
  onChange: (date: LocalDate) => void;
  /** Additional class names */
  className?: string;
}

/**
 * Date Picker - Single day selection
 *
 * Uses LocalDate strings (YYYY-MM-DD) to avoid timezone issues.
 * The parent component handles conversion to UTC when querying backend.
 */
export function DateRangePicker({
  value,
  onChange,
  className,
}: DateRangePickerProps) {
  const { dateToLocalDate } = useRegion();

  // Convert LocalDate string to Date object for calendar display
  const selectedDate = useMemo(() => {
    if (!value) return undefined;
    return localDateToDate(value);
  }, [value]);

  // Handle calendar selection - convert Date back to LocalDate string
  const handleCalendarSelect = (date: Date | undefined) => {
    if (date) {
      const localDate = dateToLocalDate(date);
      onChange(localDate);
    }
  };

  return (
    <div className={cn("", className)}>
      <Calendar
        mode="single"
        selected={selectedDate}
        onSelect={handleCalendarSelect}
        className="rounded-lg"
      />
    </div>
  );
}

// Re-export LocalDate type for convenience
export type { LocalDate };
