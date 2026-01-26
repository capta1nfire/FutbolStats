"use client";

import { PitCompliance } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";
import { DegradedAlert } from "./DegradedAlert";
import { ShieldCheck, ShieldAlert } from "lucide-react";

interface PitComplianceCardProps {
  data: PitCompliance | null;
}

/**
 * PIT Compliance Card
 *
 * Shows:
 * - total_rows
 * - violations (highlighted if > 0)
 * - violation_pct
 * - status
 */
export function PitComplianceCard({ data }: PitComplianceCardProps) {
  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <ShieldCheck className="h-4 w-4" />
          <span className="text-sm">PIT Compliance unavailable</span>
        </div>
      </div>
    );
  }

  const hasViolations = (data.violations ?? 0) > 0;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {hasViolations ? (
            <ShieldAlert className="h-4 w-4 text-red-400" />
          ) : (
            <ShieldCheck className="h-4 w-4 text-green-400" />
          )}
          <h3 className="text-sm font-semibold text-foreground">PIT Compliance</h3>
        </div>
        <StatusBadge status={data.status} />
      </div>

      {/* Degraded alert */}
      {data._degraded && (
        <div className="mb-4">
          <DegradedAlert error={data._error} />
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-4">
        {/* Total Rows */}
        <div className="text-center">
          <p className="text-2xl font-semibold tabular-nums text-foreground">
            {(data.total_rows ?? 0).toLocaleString()}
          </p>
          <p className="text-xs text-muted-foreground">Total Rows</p>
        </div>

        {/* Violations */}
        <div className="text-center">
          <p
            className={`text-2xl font-semibold tabular-nums ${
              hasViolations ? "text-red-400" : "text-green-400"
            }`}
          >
            {(data.violations ?? 0).toLocaleString()}
          </p>
          <p className="text-xs text-muted-foreground">Violations</p>
        </div>

        {/* Violation % */}
        <div className="text-center">
          <p
            className={`text-2xl font-semibold tabular-nums ${
              hasViolations ? "text-red-400" : "text-green-400"
            }`}
          >
            {(data.violation_pct ?? 0).toFixed(1)}%
          </p>
          <p className="text-xs text-muted-foreground">Violation Rate</p>
        </div>
      </div>

      {/* Warning message if violations */}
      {hasViolations && (
        <div className="mt-4 p-2 bg-red-500/10 border border-red-500/20 rounded text-sm text-red-400">
          {data.violations} PIT violation{data.violations !== 1 ? "s" : ""} detected
        </div>
      )}
    </div>
  );
}
