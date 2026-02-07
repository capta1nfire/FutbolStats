"use client";

import type { ManagerInfo } from "@/lib/types/squad";

interface ManagerTimelineProps {
  history: ManagerInfo[];
  maxVisible?: number;
}

export function ManagerTimeline({ history, maxVisible = 5 }: ManagerTimelineProps) {
  const visible = history.slice(0, maxVisible);

  if (visible.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">No manager history available</p>
    );
  }

  return (
    <div className="relative space-y-0">
      {visible.map((mgr, i) => {
        const isCurrent = i === 0 && !mgr.end_date;
        return (
          <div key={`${mgr.external_id ?? mgr.name}-${i}`} className="relative flex gap-3 pb-4 last:pb-0">
            {/* Vertical line */}
            {i < visible.length - 1 && (
              <div className="absolute left-[7px] top-4 h-full w-px bg-border" />
            )}
            {/* Dot */}
            <div
              className={`relative z-10 mt-1 h-[15px] w-[15px] flex-shrink-0 rounded-full border-2 ${
                isCurrent
                  ? "border-[var(--status-success-text)] bg-[var(--status-success-bg)]"
                  : "border-muted-foreground/30 bg-background"
              }`}
            />
            {/* Content */}
            <div className="min-w-0">
              <p className={`truncate text-sm ${isCurrent ? "font-medium" : ""}`}>
                {mgr.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {mgr.start_date
                  ? new Date(mgr.start_date).toLocaleDateString("en-US", { month: "short", year: "numeric" })
                  : "Unknown"}
                {mgr.end_date
                  ? ` — ${new Date(mgr.end_date).toLocaleDateString("en-US", { month: "short", year: "numeric" })}`
                  : isCurrent
                    ? " — Present"
                    : ""}
              </p>
            </div>
          </div>
        );
      })}
      {history.length > maxVisible && (
        <p className="pl-7 text-xs text-muted-foreground">
          +{history.length - maxVisible} more
        </p>
      )}
    </div>
  );
}
