"use client";

import { AlertTriangle } from "lucide-react";

interface DegradedAlertProps {
  error?: string;
}

/**
 * Alert shown when a section is degraded
 * Shows truncated error message without breaking layout
 */
export function DegradedAlert({ error }: DegradedAlertProps) {
  // Truncate error to prevent layout issues
  const truncatedError = error && error.length > 100
    ? `${error.slice(0, 100)}...`
    : error;

  return (
    <div className="p-3 bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)] rounded flex items-start gap-2">
      <AlertTriangle className="h-4 w-4 text-[var(--status-warning-text)] shrink-0 mt-0.5" />
      <div className="min-w-0">
        <p className="text-sm font-medium text-[var(--status-warning-text)]">Section Degraded</p>
        {truncatedError && (
          <p className="text-xs text-[var(--status-warning-text)] opacity-80 truncate">{truncatedError}</p>
        )}
      </div>
    </div>
  );
}
