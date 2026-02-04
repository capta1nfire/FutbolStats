import { cn } from "@/lib/utils";

interface QualificationBadgeProps {
  value: number | string;
  active?: boolean;
  className?: string;
}

/**
 * QualificationBadge
 *
 * UniFi-style numeric badge to mark qualifiers (e.g. top 8).
 * Uses semantic tokens (primary) and avoids hardcoded colors.
 */
export function QualificationBadge({
  value,
  active = false,
  className,
}: QualificationBadgeProps) {
  if (!active) {
    return (
      <span className={cn("text-xs text-muted-foreground/[0.92] tabular-nums", className)}>
        {value}
      </span>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center justify-center",
        "min-w-5 h-5 px-1",
        "rounded-md tabular-nums text-[11px] font-semibold",
        "bg-primary/15 text-primary border border-primary/25",
        className
      )}
    >
      {value}
    </span>
  );
}

