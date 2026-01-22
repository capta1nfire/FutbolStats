"use client";

import { cn } from "@/lib/utils";

interface LoaderProps {
  className?: string;
  /** Size variant */
  size?: "sm" | "md" | "lg";
}

/**
 * Three-dot bounce loader (UniFi style)
 * Replaces traditional spinner for page loading states
 */
export function Loader({ className, size = "md" }: LoaderProps) {
  const sizeClasses = {
    sm: "w-2.5",
    md: "w-[15px]",
    lg: "w-5",
  };

  return (
    <div
      className={cn("loader", sizeClasses[size], className)}
      role="status"
      aria-label="Loading"
    />
  );
}
