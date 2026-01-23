/**
 * Predictions Performance Card
 *
 * Displays model performance metrics including accuracy, Brier score, and calibration.
 */

import { PredictionsPerformance } from "@/lib/api/analytics";
import { Loader } from "@/components/ui/loader";

interface PredictionsPerformanceCardProps {
  data: PredictionsPerformance | null;
  isLoading: boolean;
  isDegraded?: boolean;
}

export function PredictionsPerformanceCard({
  data,
  isLoading,
  isDegraded,
}: PredictionsPerformanceCardProps) {
  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-center h-20">
          <Loader size="sm" />
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-center h-20 text-muted-foreground text-sm">
          No performance data available
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-sm font-medium text-muted-foreground">
          Predictions Performance ({data.totalPredictions} predictions)
        </h3>
        {isDegraded && (
          <span className="text-xs text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded">
            degraded
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricItem
          label="Accuracy"
          value={`${data.accuracy.toFixed(1)}%`}
          variant={data.accuracy >= 40 ? "success" : data.accuracy >= 30 ? "warning" : "error"}
        />
        <MetricItem
          label="Brier Score"
          value={data.brierScore.toFixed(3)}
          variant={data.brierScore <= 0.2 ? "success" : data.brierScore <= 0.25 ? "warning" : "error"}
        />
        <MetricItem
          label="Log Loss"
          value={data.logLoss.toFixed(3)}
          variant={data.logLoss <= 1.0 ? "success" : data.logLoss <= 1.1 ? "warning" : "error"}
        />
        {data.coveragePct !== undefined && (
          <MetricItem
            label="Coverage"
            value={`${data.coveragePct.toFixed(1)}%`}
            variant={data.coveragePct >= 80 ? "success" : data.coveragePct >= 60 ? "warning" : "error"}
          />
        )}
      </div>

      {data.marketBrier !== undefined && data.skillVsMarket !== undefined && (
        <div className="mt-4 pt-3 border-t border-border">
          <h4 className="text-xs font-medium text-muted-foreground mb-2">vs Market</h4>
          <div className="grid grid-cols-2 gap-4">
            <MetricItem
              label="Market Brier"
              value={data.marketBrier.toFixed(3)}
            />
            <MetricItem
              label="Skill vs Market"
              value={`${(data.skillVsMarket * 100).toFixed(1)}%`}
              variant={data.skillVsMarket > 0 ? "success" : data.skillVsMarket < 0 ? "error" : "default"}
            />
          </div>
        </div>
      )}

      {data.byLeague.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border">
          <h4 className="text-xs font-medium text-muted-foreground mb-2">
            By League (top 5)
          </h4>
          <div className="space-y-1">
            {data.byLeague.slice(0, 5).map((league) => (
              <LeagueRow key={league.leagueId} league={league} />
            ))}
          </div>
        </div>
      )}

      {data.calibration.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border">
          <h4 className="text-xs font-medium text-muted-foreground mb-2">Calibration</h4>
          <div className="flex gap-1">
            {data.calibration.map((bin) => (
              <CalibrationBin key={bin.bin} bin={bin} />
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

interface LeagueRowProps {
  league: {
    leagueId: string;
    n: number;
    accuracy: number;
    brierScore: number;
  };
}

function LeagueRow({ league }: LeagueRowProps) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-muted-foreground truncate max-w-[120px]">
        {league.leagueId}
      </span>
      <div className="flex gap-3">
        <span className="text-muted-foreground">n={league.n}</span>
        <span className={league.accuracy >= 40 ? "text-green-400" : "text-foreground"}>
          {league.accuracy.toFixed(1)}%
        </span>
        <span className="text-foreground">
          {league.brierScore.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

interface CalibrationBinProps {
  bin: {
    bin: string;
    count: number;
    avgConfidence: number;
    empiricalAccuracy: number;
    calibrationError: number | null;
  };
}

function CalibrationBin({ bin }: CalibrationBinProps) {
  const error = bin.calibrationError;
  let bgColor = "bg-muted";

  if (error !== null) {
    if (Math.abs(error) <= 0.05) {
      bgColor = "bg-green-500/30";
    } else if (Math.abs(error) <= 0.1) {
      bgColor = "bg-yellow-500/30";
    } else {
      bgColor = "bg-red-500/30";
    }
  }

  return (
    <div
      className={`flex flex-col items-center px-2 py-1 rounded ${bgColor} flex-1`}
      title={`${bin.bin}: ${bin.count} predictions, ${(bin.empiricalAccuracy * 100).toFixed(1)}% actual vs ${(bin.avgConfidence * 100).toFixed(1)}% predicted`}
    >
      <span className="text-[10px] text-muted-foreground">{bin.bin}</span>
      <span className="text-xs font-medium">{bin.count}</span>
    </div>
  );
}
