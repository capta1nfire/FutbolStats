"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";
import { TrendingUp, Trophy, Loader2, AlertCircle, AlertTriangle } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { useModelBenchmark, useModelSelection } from "@/lib/hooks";
import { DailyModelStats } from "@/lib/types/model-benchmark";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Dynamic import to avoid SSR issues with ApexCharts
const ReactApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

// All available models
const ALL_MODELS = ["Market", "Model A", "Shadow", "Sensor B"] as const;

// Model config with colors
const MODEL_CONFIG: Record<string, { color: string; strokeColor: string }> = {
  Market: { color: "rgba(148, 163, 184, 0.2)", strokeColor: "#94a3b8" },
  "Model A": { color: "rgba(34, 197, 94, 0.2)", strokeColor: "#22c55e" },
  Shadow: { color: "rgba(168, 85, 247, 0.2)", strokeColor: "#a855f7" },
  "Sensor B": { color: "rgba(249, 115, 22, 0.2)", strokeColor: "#f97316" },
};

interface ModelBenchmarkTileProps {
  className?: string;
}

/**
 * Calculate cumulative accuracy up to each day
 */
function calculateCumulativeData(dailyData: DailyModelStats[], selectedModels: string[]) {
  let marketCum = 0;
  let modelACum = 0;
  let shadowCum = 0;
  let sensorBCum = 0;
  let totalMatches = 0;

  return dailyData.map((day) => {
    marketCum += day.market_correct;
    modelACum += day.model_a_correct;
    shadowCum += day.shadow_correct;
    sensorBCum += day.sensor_b_correct;
    totalMatches += day.matches;

    // Parse date as local time (not UTC) to avoid timezone shift
    // "2026-01-29" + "T12:00:00" prevents the date from shifting to previous day
    return {
      date: new Date(day.date + "T12:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      market: selectedModels.includes("Market") && totalMatches > 0 ? (marketCum / totalMatches) * 100 : 0,
      modelA: selectedModels.includes("Model A") && totalMatches > 0 ? (modelACum / totalMatches) * 100 : 0,
      shadow: selectedModels.includes("Shadow") && totalMatches > 0 ? (shadowCum / totalMatches) * 100 : 0,
      sensorB: selectedModels.includes("Sensor B") && totalMatches > 0 ? (sensorBCum / totalMatches) * 100 : 0,
    };
  });
}

/**
 * Format date for display (e.g., "10 ene")
 * Uses T12:00:00 to avoid timezone shift when parsing ISO date strings
 */
function formatStartDate(dateStr: string): string {
  const date = new Date(dateStr + "T12:00:00");
  return date.toLocaleDateString("es-ES", { day: "numeric", month: "short" });
}

// =============================================================================
// Trend Arrow Calculation (WMA 3d vs 3d prev)
// =============================================================================

type ModelKey = "market" | "model_a" | "shadow" | "sensor_b";

interface TrendInfo {
  arrow: string;
  delta: number;
}

const TREND_THRESHOLDS = {
  strongUp: 2.0,
  moderateUp: 0.5,
  moderateDown: -0.5,
  strongDown: -2.0,
};

const MIN_MATCHES_FOR_TREND = 25;

/**
 * Map series name to model key for accessing DailyModelStats fields
 */
function seriesNameToModelKey(name: string): ModelKey | null {
  const map: Record<string, ModelKey> = {
    "Market": "market",
    "Model A": "model_a",
    "Shadow": "shadow",
    "Sensor B": "sensor_b",
  };
  return map[name] ?? null;
}

/**
 * Get arrow symbol based on delta (percentage points)
 */
function getArrowForDelta(delta: number): string {
  if (delta > TREND_THRESHOLDS.strongUp) return "↑";
  if (delta > TREND_THRESHOLDS.moderateUp) return "↗";
  if (delta > TREND_THRESHOLDS.moderateDown) return "→";
  if (delta > TREND_THRESHOLDS.strongDown) return "↘";
  return "↓";
}

/**
 * Calculate trend using weighted moving average (3 days vs previous 3 days)
 * Returns null if insufficient data for meaningful trend
 */
function calculateTrend(
  dailyData: DailyModelStats[],
  dataIndex: number,
  modelKey: ModelKey
): TrendInfo | null {
  // Need at least 6 days of data for 3+3 comparison
  if (dataIndex < 5 || dailyData.length < 6) return null;

  const correctKey = `${modelKey}_correct` as keyof DailyModelStats;

  // Recent 3 days (including current point)
  let recentCorrect = 0;
  let recentMatches = 0;
  for (let i = dataIndex; i > dataIndex - 3 && i >= 0; i--) {
    recentCorrect += dailyData[i][correctKey] as number;
    recentMatches += dailyData[i].matches;
  }

  // Previous 3 days
  let prevCorrect = 0;
  let prevMatches = 0;
  for (let i = dataIndex - 3; i > dataIndex - 6 && i >= 0; i--) {
    prevCorrect += dailyData[i][correctKey] as number;
    prevMatches += dailyData[i].matches;
  }

  // Check minimum sample size - fallback to neutral if insufficient
  if (recentMatches < MIN_MATCHES_FOR_TREND || prevMatches < MIN_MATCHES_FOR_TREND) {
    return null;
  }

  const recentAcc = (recentCorrect / recentMatches) * 100;
  const prevAcc = (prevCorrect / prevMatches) * 100;
  const delta = recentAcc - prevAcc;

  return {
    arrow: getArrowForDelta(delta),
    delta: delta,
  };
}

/**
 * Model Benchmark Tile
 *
 * Full-width card showing model performance comparison over time.
 * Features:
 * - Dynamic date range based on selected models
 * - Area chart with cumulative accuracy
 * - Model selector toggles (min 2 required)
 * - Summary stats (accuracy, days won, correct predictions)
 * - Highlights best performing model
 */
export function ModelBenchmarkTile({ className }: ModelBenchmarkTileProps) {
  // Track selected models with localStorage persistence
  const { selectedModels, toggleModel, canDeselect } = useModelSelection();

  // Fetch data with selected models
  const { data, isLoading, isDegraded, validationError } = useModelBenchmark(selectedModels);

  // Transform API data for chart
  const chartData = useMemo(() => {
    if (!data?.daily_data || data.daily_data.length === 0) return [];
    return calculateCumulativeData(data.daily_data, selectedModels);
  }, [data, selectedModels]);

  // Find leader (highest accuracy)
  const leader = useMemo(() => {
    if (!data?.models || data.models.length === 0) return null;
    return data.models.reduce((best, model) =>
      model.accuracy > best.accuracy ? model : best
    );
  }, [data]);

  // Build chart series from selected models
  const chartSeries = useMemo(() => {
    const series = [];

    if (selectedModels.includes("Market") && chartData.length > 0) {
      series.push({
        name: "Market",
        data: chartData.map((d) => d.market),
      });
    }
    if (selectedModels.includes("Model A") && chartData.length > 0) {
      series.push({
        name: "Model A",
        data: chartData.map((d) => d.modelA),
      });
    }
    if (selectedModels.includes("Shadow") && chartData.length > 0) {
      series.push({
        name: "Shadow",
        data: chartData.map((d) => d.shadow),
      });
    }
    if (selectedModels.includes("Sensor B") && chartData.length > 0) {
      series.push({
        name: "Sensor B",
        data: chartData.map((d) => d.sensorB),
      });
    }
    return series;
  }, [chartData, selectedModels]);

  // Calculate Y-axis range from data
  const yAxisRange = useMemo(() => {
    if (chartData.length === 0) return { min: 30, max: 60 };

    const allValues: number[] = [];
    chartData.forEach((d) => {
      if (selectedModels.includes("Market")) allValues.push(d.market);
      if (selectedModels.includes("Model A")) allValues.push(d.modelA);
      if (selectedModels.includes("Shadow")) allValues.push(d.shadow);
      if (selectedModels.includes("Sensor B")) allValues.push(d.sensorB);
    });

    if (allValues.length === 0) return { min: 30, max: 60 };

    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    // Add some padding
    const padding = (maxVal - minVal) * 0.1 || 5;
    return {
      min: Math.max(0, Math.floor(minVal - padding)),
      max: Math.min(100, Math.ceil(maxVal + padding)),
    };
  }, [chartData, selectedModels]);

  // Build chart colors based on selected models
  const chartColors = useMemo(() => {
    const colors: string[] = [];
    if (selectedModels.includes("Market")) colors.push("#94a3b8");
    if (selectedModels.includes("Model A")) colors.push("#22c55e");
    if (selectedModels.includes("Shadow")) colors.push("#a855f7");
    if (selectedModels.includes("Sensor B")) colors.push("#f97316");
    return colors;
  }, [selectedModels]);

  const chartOptions: ApexCharts.ApexOptions = useMemo(
    () => ({
      chart: {
        type: "area",
        height: 280,
        background: "transparent",
        toolbar: { show: false },
        fontFamily: "inherit",
      },
      theme: {
        mode: "dark",
      },
      colors: chartColors,
      stroke: {
        curve: "smooth",
        width: 2,
      },
      fill: {
        type: "gradient",
        gradient: {
          shadeIntensity: 1,
          opacityFrom: 0.4,
          opacityTo: 0.05,
          stops: [0, 90, 100],
        },
      },
      dataLabels: { enabled: false },
      legend: { show: false },
      grid: {
        borderColor: "rgba(255,255,255,0.05)",
        strokeDashArray: 3,
        xaxis: { lines: { show: true } },
        yaxis: { lines: { show: true } },
        padding: { right: 20 },
      },
      xaxis: {
        categories: chartData.map((d) => d.date),
        labels: {
          style: { colors: "#6b7280", fontSize: "11px" },
          rotate: -45,
          rotateAlways: false,
          hideOverlappingLabels: true,
          trim: true,
          showDuplicates: false,
        },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tooltip: { enabled: false },
        tickAmount: Math.min(chartData.length, 10),
        tickPlacement: "on",
      },
      yaxis: {
        min: yAxisRange.min,
        max: yAxisRange.max,
        tickAmount: 5,
        labels: {
          style: { colors: "#6b7280", fontSize: "11px" },
          formatter: (value: number) => `${value.toFixed(0)}%`,
        },
      },
      tooltip: {
        theme: "dark",
        shared: true,
        intersect: false,
        marker: { show: false }, // Hide default circles, we use custom arrows
        custom: function({ series, dataPointIndex, w }) {
          // Closure captures dailyData from component scope
          const dailyData = data?.daily_data ?? [];
          const dayData = dailyData[dataPointIndex];

          let html = '<div style="padding: 8px 12px; background: #1f2937; border: 1px solid #374151; border-radius: 6px; font-family: inherit;">';

          // Header with date and total matches
          const dateLabel = w.globals.categoryLabels?.[dataPointIndex] ?? "";
          const matchesLabel = dayData ? ` (${dayData.matches} partidos)` : "";
          html += `<div style="color: #9ca3af; font-size: 10px; margin-bottom: 6px; border-bottom: 1px solid #374151; padding-bottom: 4px;">${dateLabel}${matchesLabel}</div>`;

          // Map series name to correct field in dayData
          const correctFieldMap: Record<string, string> = {
            "Market": "market_correct",
            "Model A": "model_a_correct",
            "Shadow": "shadow_correct",
            "Sensor B": "sensor_b_correct",
          };

          for (let i = 0; i < series.length; i++) {
            const seriesName = w.config.series[i].name as string;
            const color = w.config.colors[i] as string;

            // Get model key for trend calculation
            const modelKey = seriesNameToModelKey(seriesName);

            // Calculate trend (returns null if insufficient data)
            const trend = modelKey ? calculateTrend(dailyData, dataPointIndex, modelKey) : null;

            // Arrow defaults to "→" if no trend data
            const arrow = trend?.arrow ?? "→";
            const deltaStr = trend
              ? `${trend.delta >= 0 ? "+" : ""}${trend.delta.toFixed(1)}pp`
              : "";

            // Get daily correct count for this model
            const correctField = correctFieldMap[seriesName];
            const dailyCorrect = dayData && correctField ? (dayData as unknown as Record<string, number>)[correctField] : null;
            const dailyStr = dailyCorrect !== null ? `${dailyCorrect}/${dayData?.matches ?? 0}` : "";

            // Calculate daily percentage (not cumulative)
            const dailyPercent = dailyCorrect !== null && dayData?.matches
              ? ((dailyCorrect / dayData.matches) * 100).toFixed(1)
              : "0.0";

            // For Model A, append version suffix from daily data
            let displayName = seriesName;
            if (seriesName === "Model A" && dayData?.model_a_version) {
              displayName = `Model A (${dayData.model_a_version})`;
            }

            html += `
              <div style="display: flex; align-items: center; gap: 8px; padding: 2px 0;">
                <span style="color: ${color}; font-size: 14px; width: 16px; text-align: center;">${arrow}</span>
                <span style="color: #6b7280; font-size: 10px; width: 50px; font-family: monospace;">${deltaStr}</span>
                <span style="color: #e5e7eb; font-size: 11px; flex: 1;">${displayName}</span>
                <span style="color: #9ca3af; font-size: 10px; font-family: monospace; margin-right: 4px;">${dailyStr}</span>
                <span style="color: ${color}; font-weight: 600; font-size: 11px; font-family: monospace;">${dailyPercent}%</span>
              </div>
            `;
          }

          html += "</div>";
          return html;
        },
      },
      markers: {
        size: 0,
        strokeWidth: 0,
        hover: { size: 4 },
      },
    }),
    [chartData, chartColors, yAxisRange, data?.daily_data]
  );

  // Loading state
  if (isLoading) {
    return (
      <div
        className={cn(
          "bg-tile border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
          className
        )}
      >
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span className="text-sm">Loading benchmark data...</span>
        </div>
      </div>
    );
  }

  // Validation error state (< 2 models)
  if (validationError) {
    return (
      <div
        className={cn(
          "bg-tile border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
          className
        )}
      >
        <div className="flex flex-col items-center gap-2 text-yellow-500">
          <AlertTriangle className="h-6 w-6" />
          <span className="text-sm">{validationError}</span>
        </div>
      </div>
    );
  }

  // Error/degraded state
  if (isDegraded || !data) {
    return (
      <div
        className={cn(
          "bg-tile border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
          className
        )}
      >
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <AlertCircle className="h-6 w-6" />
          <span className="text-sm">Failed to load benchmark data</span>
        </div>
      </div>
    );
  }

  // Empty data state
  if (data.total_matches === 0) {
    return (
      <div
        className={cn(
          "bg-tile border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
          className
        )}
      >
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <TrendingUp className="h-6 w-6" />
          <span className="text-sm">No benchmark data available yet</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn("bg-tile border border-border rounded-lg p-4", className)}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Model Benchmark</h3>
          <span className="text-xs text-muted-foreground ml-2">
            desde {formatStartDate(data.start_date)}, n={data.total_matches}
          </span>
        </div>

        {/* Summary badges */}
        {leader && (
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5">
              <Trophy className="h-3 w-3 text-yellow-500" />
              <span className="font-medium text-foreground">{leader.name}:</span>
              <span className="text-muted-foreground">{leader.accuracy}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Model toggles */}
      <div className="flex flex-wrap gap-2 mb-4">
        {ALL_MODELS.map((modelName) => {
          const isSelected = selectedModels.includes(modelName);
          const modelData = data.models.find((m) => m.name === modelName);
          const config = MODEL_CONFIG[modelName];
          const canToggle = canDeselect(modelName);

          const isDisabled = isSelected && !canToggle;

          return (
            <label
              key={modelName}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all border cursor-pointer select-none",
                isSelected
                  ? "bg-tile border-border hover:border-primary/50"
                  : "bg-transparent border-transparent opacity-50 hover:opacity-75",
                isDisabled && "cursor-not-allowed opacity-75"
              )}
            >
              <Checkbox
                checked={isSelected}
                onCheckedChange={() => !isDisabled && toggleModel(modelName)}
                disabled={isDisabled}
              />
              <span className="text-foreground">{modelName}</span>

              {/* Stats row (only if model has data) */}
              {modelData && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2 ml-1 cursor-pointer">
                        <span
                          className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                          style={{
                            backgroundColor: config.color,
                            color: config.strokeColor,
                          }}
                        >
                          {modelData.accuracy}%
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-gray-500/20 text-gray-400 font-medium">
                          {Number.isInteger(modelData.days_won) ? modelData.days_won : modelData.days_won.toFixed(1)}D | {modelData.correct}
                        </span>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-[220px]">
                      <div className="text-xs space-y-1">
                        <div><strong>{modelData.accuracy}%</strong> accuracy acumulada</div>
                        <div><strong>{Number.isInteger(modelData.days_won) ? modelData.days_won : modelData.days_won.toFixed(1)}</strong> días ganados</div>
                        <div><strong>{modelData.correct}</strong> predicciones correctas</div>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </label>
          );
        })}
      </div>

      {/* Chart */}
      <div className="h-[280px] -mx-2">
        {typeof window !== "undefined" && chartData.length > 0 && (
          <ReactApexChart
            options={chartOptions}
            series={chartSeries}
            type="area"
            height={280}
          />
        )}
      </div>

    </div>
  );
}
