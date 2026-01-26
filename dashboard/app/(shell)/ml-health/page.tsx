"use client";

import { useMLHealth } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { RefreshCw, AlertTriangle, Copy, Check } from "lucide-react";
import { useState, useCallback } from "react";
import {
  FuelGaugeCard,
  SotaStatsCoverageCard,
  TitanCoverageCard,
  PitComplianceCard,
  FreshnessCard,
  PredictionConfidenceCard,
  TopRegressionsCard,
} from "@/components/ml-health";

/**
 * ML Health Dashboard Page
 *
 * One-glance view for ML pipeline health:
 * - Fuel Gauge (primary indicator)
 * - SOTA Stats Coverage (root cause)
 * - TITAN Coverage (feature_matrix)
 * - PIT Compliance
 * - Freshness/Staleness
 * - Prediction Confidence
 * - Top Regressions
 */
export default function MLHealthPage() {
  const {
    data,
    health,
    isDegraded,
    generatedAt,
    cached,
    cacheAgeSeconds,
    isLoading,
    error,
    refetch,
  } = useMLHealth();

  const [copied, setCopied] = useState(false);

  const handleCopyJson = useCallback(async () => {
    if (!data) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textArea = document.createElement("textarea");
      textArea.value = JSON.stringify(data, null, 2);
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [data]);

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <Loader size="md" />
      </div>
    );
  }

  // Error/degraded state
  if (isDegraded || !data) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 text-center max-w-md">
          <AlertTriangle className="h-12 w-12 text-yellow-400" />
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-2">
              ML Health Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || "Unable to fetch ML health data from backend"}
            </p>
          </div>
          <Button onClick={() => refetch()} variant="secondary">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="shrink-0 px-6 py-4 border-b border-border flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-foreground">ML Health Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Pipeline health monitoring and diagnostics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopyJson}
            className="gap-2"
          >
            {copied ? (
              <>
                <Check className="h-4 w-4 text-green-400" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-4 w-4" />
                Copy JSON
              </>
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => refetch()}
            className="gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Content - Scrollable (min-h-0 is the flexbox trick to allow shrinking) */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="p-6 space-y-6">
          {/* Row 1: Fuel Gauge (Primary) */}
          <FuelGaugeCard
            fuelGauge={data.fuel_gauge}
            rootHealth={health}
            generatedAt={generatedAt}
            cached={cached}
            cacheAgeSeconds={cacheAgeSeconds}
          />

          {/* Row 2: Coverage Cards (2 columns on desktop) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SotaStatsCoverageCard data={data.sota_stats_coverage} />
            <TitanCoverageCard data={data.titan_coverage} />
          </div>

          {/* Row 3: PIT + Freshness (2 columns on desktop) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PitComplianceCard data={data.pit_compliance} />
            <FreshnessCard data={data.freshness} />
          </div>

          {/* Row 4: Prediction Confidence */}
          <PredictionConfidenceCard data={data.prediction_confidence} />

          {/* Row 5: Top Regressions */}
          <TopRegressionsCard data={data.top_regressions} />
        </div>
      </ScrollArea>
    </div>
  );
}
