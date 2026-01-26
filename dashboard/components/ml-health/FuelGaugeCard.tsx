"use client";

import { FuelGauge, MLHealthStatus } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";
import { cn } from "@/lib/utils";
import { Fuel, AlertTriangle, CheckCircle, XCircle } from "lucide-react";

interface FuelGaugeCardProps {
  fuelGauge: FuelGauge | null;
  rootHealth: MLHealthStatus | null;
  generatedAt: string | null;
  cached: boolean;
  cacheAgeSeconds: number | null;
}

/**
 * Get icon based on fuel gauge status
 */
function StatusIcon({ status }: { status: MLHealthStatus | null }) {
  const baseClass = "h-5 w-5";

  switch (status) {
    case "ok":
      return <CheckCircle className={cn(baseClass, "text-green-400")} />;
    case "warn":
      return <AlertTriangle className={cn(baseClass, "text-yellow-400")} />;
    case "error":
      return <XCircle className={cn(baseClass, "text-red-400")} />;
    default:
      return <Fuel className={cn(baseClass, "text-muted-foreground")} />;
  }
}

/**
 * Format datetime for display
 */
function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  try {
    const date = new Date(iso);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return iso;
  }
}

/**
 * Fuel Gauge Card - Primary ML Pipeline Health Indicator
 *
 * Shows:
 * - fuel_gauge.status (ok/warn/error)
 * - fuel_gauge.reasons (list of issues)
 * - root health status
 * - cache info
 */
export function FuelGaugeCard({
  fuelGauge,
  rootHealth,
  generatedAt,
  cached,
  cacheAgeSeconds,
}: FuelGaugeCardProps) {
  const status = fuelGauge?.status ?? null;
  const reasons = fuelGauge?.reasons ?? [];
  const asOfUtc = fuelGauge?.as_of_utc ?? null;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <StatusIcon status={status} />
          <div>
            <h3 className="text-sm font-semibold text-foreground">ML Health / Fuel Gauge</h3>
            <p className="text-xs text-muted-foreground">Pipeline health indicator</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={status} size="md" />
        </div>
      </div>

      {/* Root Health */}
      <div className="flex items-center gap-2 mb-4 pb-4 border-b border-border">
        <span className="text-xs text-muted-foreground">Root Health:</span>
        <StatusBadge status={rootHealth} size="sm" />
      </div>

      {/* Reasons */}
      {reasons.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-medium text-muted-foreground mb-2">Issues:</p>
          <ul className="space-y-1">
            {reasons.map((reason, idx) => (
              <li
                key={idx}
                className="text-sm text-yellow-400 flex items-start gap-2"
              >
                <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* No issues message */}
      {reasons.length === 0 && status === "ok" && (
        <div className="mb-4 flex items-center gap-2 text-green-400">
          <CheckCircle className="h-4 w-4" />
          <span className="text-sm">All systems operational</span>
        </div>
      )}

      {/* Metadata footer */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground pt-2 border-t border-border">
        <span>Generated: {formatDateTime(generatedAt)}</span>
        {asOfUtc && asOfUtc !== generatedAt && (
          <span>As of: {formatDateTime(asOfUtc)}</span>
        )}
        {cached && (
          <span className="px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded text-[10px]">
            CACHED {cacheAgeSeconds !== null && `(${cacheAgeSeconds}s)`}
          </span>
        )}
      </div>
    </div>
  );
}
