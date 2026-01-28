"use client";

import { cn } from "@/lib/utils";
import { MLHealthStatus } from "@/lib/types";

interface StatusBadgeProps {
  status: MLHealthStatus | null;
  size?: "sm" | "md";
  className?: string;
}

/**
 * Get status color classes based on backend status
 * IMPORTANT: Colors represent backend-determined status, no frontend thresholds
 */
function getStatusColors(status: MLHealthStatus | null): string {
  switch (status) {
    case "ok":
      return "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]";
    case "warn":
      return "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]";
    case "error":
      return "bg-[var(--status-error-bg)] text-[var(--status-error-text)] border-[var(--status-error-border)]";
    case "partial":
      return "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]";
    case "not_ready":
      return "bg-muted/50 text-muted-foreground border-border";
    default:
      return "bg-muted/50 text-muted-foreground border-border";
  }
}

/**
 * Get display label for status
 */
function getStatusLabel(status: MLHealthStatus | null): string {
  switch (status) {
    case "ok":
      return "OK";
    case "warn":
      return "Warning";
    case "error":
      return "Error";
    case "partial":
      return "Partial";
    case "not_ready":
      return "Not Ready";
    default:
      return "Unknown";
  }
}

/**
 * Status badge component
 * Displays backend-determined status with appropriate colors
 */
export function StatusBadge({ status, size = "sm", className }: StatusBadgeProps) {
  const sizeClasses = size === "sm"
    ? "px-1.5 py-0.5 text-[10px]"
    : "px-2 py-1 text-xs";

  return (
    <span
      className={cn(
        "font-medium rounded border uppercase tracking-wide",
        sizeClasses,
        getStatusColors(status),
        className
      )}
    >
      {getStatusLabel(status)}
    </span>
  );
}
