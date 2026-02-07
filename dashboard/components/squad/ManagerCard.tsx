"use client";

import { User } from "lucide-react";
import type { ManagerInfo } from "@/lib/types/squad";

interface ManagerCardProps {
  manager: ManagerInfo;
  compact?: boolean;
}

export function ManagerCard({ manager, compact = false }: ManagerCardProps) {
  const tenureLabel = manager.tenure_days != null
    ? manager.tenure_days < 30
      ? `${manager.tenure_days}d`
      : `${Math.floor(manager.tenure_days / 30)}mo`
    : null;

  return (
    <div className="flex items-center gap-3 rounded-lg bg-muted/50 p-3">
      {manager.photo_url ? (
        <img
          src={manager.photo_url}
          alt={manager.name}
          className="h-8 w-8 rounded-full object-cover"
        />
      ) : (
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted">
          <User className="h-4 w-4 text-muted-foreground" />
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">{manager.name}</span>
          {manager.tenure_days != null && manager.tenure_days < 60 && (
            <span className="inline-flex items-center rounded-full bg-[var(--status-warning-bg)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--status-warning-text)] border border-[var(--status-warning-border)]">
              NEW
            </span>
          )}
        </div>
        {!compact && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {manager.nationality && <span>{manager.nationality}</span>}
            {manager.start_date && (
              <>
                {manager.nationality && <span>&middot;</span>}
                <span>Since {new Date(manager.start_date).toLocaleDateString("en-US", { month: "short", year: "numeric" })}</span>
              </>
            )}
            {tenureLabel && (
              <>
                <span>&middot;</span>
                <span>{tenureLabel}</span>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
