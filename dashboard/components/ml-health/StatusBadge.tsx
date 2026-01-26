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
      return "bg-green-500/20 text-green-400 border-green-500/30";
    case "warn":
      return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    case "error":
      return "bg-red-500/20 text-red-400 border-red-500/30";
    case "partial":
      return "bg-orange-500/20 text-orange-400 border-orange-500/30";
    case "not_ready":
      return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    default:
      return "bg-gray-500/20 text-gray-400 border-gray-500/30";
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
