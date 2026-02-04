import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface SurfaceCardProps {
  children: ReactNode;
  className?: string;
}

/**
 * SurfaceCard
 *
 * UniFi-style "section" container for drawer/settings panels.
 * Uses semantic tokens (`bg-surface`) to stay aligned with the global theme.
 */
export function SurfaceCard({ children, className }: SurfaceCardProps) {
  return (
    <div className={cn("bg-surface rounded-lg p-4", className)}>
      {children}
    </div>
  );
}

