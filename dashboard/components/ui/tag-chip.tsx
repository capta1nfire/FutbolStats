import { ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * TagTone — identifier/category colors (NOT status).
 *
 * ADS rule: green/emerald are reserved for status-success.
 * Tags use non-status tones only: purple, cyan, pink, indigo, blue, orange, gray.
 */
export type TagTone = "purple" | "cyan" | "pink" | "indigo" | "blue" | "orange" | "gray";

/**
 * Static tone → class mapping (no dynamic class construction).
 * Each tone uses 3-channel CSS variables: tag-{tone}-{bg/text/border}.
 */
const TONE_CLASSES: Record<TagTone, string> = {
  purple:
    "bg-[var(--tag-purple-bg)] text-[var(--tag-purple-text)] border-[var(--tag-purple-border)]",
  cyan:
    "bg-[var(--tag-cyan-bg)] text-[var(--tag-cyan-text)] border-[var(--tag-cyan-border)]",
  pink:
    "bg-[var(--tag-pink-bg)] text-[var(--tag-pink-text)] border-[var(--tag-pink-border)]",
  indigo:
    "bg-[var(--tag-indigo-bg)] text-[var(--tag-indigo-text)] border-[var(--tag-indigo-border)]",
  blue:
    "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)] border-[var(--tag-blue-border)]",
  orange:
    "bg-[var(--tag-orange-bg)] text-[var(--tag-orange-text)] border-[var(--tag-orange-border)]",
  gray:
    "bg-[var(--tag-gray-bg)] text-[var(--tag-gray-text)] border-[var(--tag-gray-border)]",
};

interface TagChipProps {
  tone: TagTone;
  children: ReactNode;
  icon?: ReactNode;
  className?: string;
}

/**
 * TagChip — identifier/category badge.
 *
 * Use for type/category labels (audit types, report types, data quality categories).
 * For status indicators (ok/warn/error/info), use StatusChip instead.
 */
export function TagChip({ tone, children, icon, className }: TagChipProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full text-xs font-medium px-2 py-0.5 border whitespace-nowrap",
        TONE_CLASSES[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}
