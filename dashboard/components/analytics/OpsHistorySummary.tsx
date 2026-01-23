/**
 * Ops History Summary Card
 *
 * Displays a summary of operational metrics from the last N days.
 */

import { OpsHistoryRollup, summarizeHistory } from "@/lib/api/analytics";
import { Loader } from "@/components/ui/loader";

interface OpsHistorySummaryProps {
  data: OpsHistoryRollup[];
  isLoading: boolean;
  isDegraded?: boolean;
}

export function OpsHistorySummary({
  data,
  isLoading,
  isDegraded,
}: OpsHistorySummaryProps) {
  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-center h-20">
          <Loader size="sm" />
        </div>
      </div>
    );
  }

  const summary = summarizeHistory(data);

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-sm font-medium text-muted-foreground">
          Operations Summary ({summary.daysWithData} days)
        </h3>
        {isDegraded && (
          <span className="text-xs text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded">
            degraded
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricItem
          label="PIT Snapshots"
          value={formatNumber(summary.totalPitSnapshots)}
        />
        <MetricItem
          label="Evaluable Bets"
          value={formatNumber(summary.totalEvaluableBets)}
        />
        <MetricItem
          label="Baseline Coverage"
          value={`${summary.avgBaselineCoverage.toFixed(1)}%`}
          variant={summary.avgBaselineCoverage >= 80 ? "success" : summary.avgBaselineCoverage >= 60 ? "warning" : "error"}
        />
        <MetricItem
          label="Total Errors"
          value={formatNumber(summary.totalErrors)}
          variant={summary.totalErrors === 0 ? "success" : summary.totalErrors < 10 ? "warning" : "error"}
        />
      </div>

      {data.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border">
          <h4 className="text-xs font-medium text-muted-foreground mb-2">Recent Days</h4>
          <div className="flex gap-1 overflow-x-auto">
            {data.slice(0, 7).map((day) => (
              <DayIndicator key={day.day} day={day} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface MetricItemProps {
  label: string;
  value: string;
  variant?: "default" | "success" | "warning" | "error";
}

function MetricItem({ label, value, variant = "default" }: MetricItemProps) {
  const variantClasses = {
    default: "text-foreground",
    success: "text-green-400",
    warning: "text-yellow-400",
    error: "text-red-400",
  };

  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`text-lg font-semibold ${variantClasses[variant]}`}>
        {value}
      </div>
    </div>
  );
}

interface DayIndicatorProps {
  day: OpsHistoryRollup;
}

function DayIndicator({ day }: DayIndicatorProps) {
  const hasErrors = day.errors429 > 0 || day.timeouts > 0;
  const hasData = day.pitSnapshots > 0;

  let bgColor = "bg-muted";
  if (hasData && !hasErrors) {
    bgColor = "bg-green-500/30";
  } else if (hasData && hasErrors) {
    bgColor = "bg-yellow-500/30";
  } else if (!hasData) {
    bgColor = "bg-red-500/30";
  }

  const dateLabel = day.day.slice(5); // MM-DD

  return (
    <div
      className={`flex flex-col items-center px-2 py-1 rounded ${bgColor}`}
      title={`${day.day}: ${day.pitSnapshots} snapshots, ${day.errors429 + day.timeouts} errors`}
    >
      <span className="text-[10px] text-muted-foreground">{dateLabel}</span>
      <span className="text-xs font-medium">{day.pitSnapshots}</span>
    </div>
  );
}

function formatNumber(n: number): string {
  if (n >= 1000000) {
    return `${(n / 1000000).toFixed(1)}M`;
  }
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}K`;
  }
  return n.toString();
}
