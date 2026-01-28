import { ReactNode } from "react";
import { cn } from "@/lib/utils";

export type StatusTone = "success" | "warning" | "error" | "info";

/**
 * Static mapping of tone → Tailwind classes (literal strings for JIT safety).
 * Consumes CSS variables from globals.css (--status-{tone}-{bg/text/border}).
 */
const TONE_CLASSES: Record<StatusTone, string> = {
  success:
    "bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]",
  warning:
    "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]",
  error:
    "bg-[var(--status-error-bg)] text-[var(--status-error-text)] border-[var(--status-error-border)]",
  info:
    "bg-[var(--status-info-bg)] text-[var(--status-info-text)] border-[var(--status-info-border)]",
};

interface StatusChipProps {
  tone: StatusTone;
  children: ReactNode;
  icon?: ReactNode;
  className?: string;
}

/**
 * StatusChip — ADS-compliant status badge
 *
 * Semantic badge consuming 3-channel status tokens (text/bg/border).
 * Replaces ad-hoc bg-green-500/15, bg-red-500/20, etc.
 *
 * @example
 * <StatusChip tone="success" icon={<CheckCircle2 className="h-3 w-3" />}>Passing</StatusChip>
 */
export function StatusChip({ tone, children, icon, className }: StatusChipProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full text-xs font-medium px-2 py-0.5 border whitespace-nowrap",
        TONE_CLASSES[tone],
        className
      )}
    >
      {icon}
      {children}
    </span>
  );
}
