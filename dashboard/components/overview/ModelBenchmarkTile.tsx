"use client";

import { useState, useMemo } from "react";
import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";
import { TrendingUp, Trophy, CheckSquare, Loader2, AlertCircle, AlertTriangle } from "lucide-react";
import { useModelBenchmark } from "@/lib/hooks/use-model-benchmark";
import { DailyModelStats } from "@/lib/types/model-benchmark";

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
  // Track selected models as array
  const [selectedModels, setSelectedModels] = useState<string[]>([...ALL_MODELS]);

  // Fetch data with selected models
  const { data, isLoading, isDegraded, validationError } = useModelBenchmark(selectedModels);

  // Toggle model selection
  const toggleModel = (name: string) => {
    setSelectedModels((prev) => {
      const isSelected = prev.includes(name);
      if (isSelected) {
        // Don't allow deselecting if it would leave < 2 models
        if (prev.length <= 2) return prev;
        return prev.filter((m) => m !== name);
      } else {
        return [...prev, name];
      }
    });
  };

  // Check if a model can be toggled off
  const canDeselect = (name: string): boolean => {
    const isSelected = selectedModels.includes(name);
    return !isSelected || selectedModels.length > 2;
  };

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
        tickAmount: Math.min(chartData.length, 10), // Show ~10 ticks including last
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
        y: {
          formatter: (value: number) => `${value.toFixed(1)}%`,
        },
        style: {
          fontSize: "12px",
        },
        marker: { show: true },
      },
      markers: {
        size: 0,
        strokeWidth: 0,
        hover: { size: 4 },
      },
    }),
    [chartData, chartColors, yAxisRange]
  );

  // Loading state
  if (isLoading) {
    return (
      <div
        className={cn(
          "bg-surface border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
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
          "bg-surface border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
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
          "bg-surface border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
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
          "bg-surface border border-border rounded-lg p-4 h-[420px] flex items-center justify-center",
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
      className={cn("bg-surface border border-border rounded-lg p-4", className)}
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
              <span className="text-muted-foreground">Líder:</span>
              <span className="font-medium text-foreground">{leader.name}</span>
              <span className="text-muted-foreground">({leader.accuracy}%)</span>
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

          return (
            <button
              key={modelName}
              onClick={() => toggleModel(modelName)}
              disabled={isSelected && !canToggle}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all border",
                isSelected
                  ? "bg-surface border-border hover:border-primary/50"
                  : "bg-transparent border-transparent opacity-50 hover:opacity-75",
                isSelected && !canToggle && "cursor-not-allowed opacity-75"
              )}
              title={!canToggle ? "Mínimo 2 modelos requeridos" : undefined}
            >
              <CheckSquare
                className={cn(
                  "h-3.5 w-3.5 transition-colors",
                  isSelected ? "text-primary" : "text-muted-foreground"
                )}
              />
              <span className="text-foreground">{modelName}</span>

              {/* Stats row (only if model has data) */}
              {modelData && (
                <div className="flex items-center gap-2 ml-1">
                  <span
                    className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                    style={{
                      backgroundColor: config.color,
                      color: config.strokeColor,
                    }}
                  >
                    {modelData.accuracy}%
                  </span>
                  <span
                    className="px-1 py-0.5 rounded text-[10px] bg-yellow-500/20 text-yellow-500 font-medium"
                    title="Días ganados (empates se dividen: 2 empatados = 0.5 cada uno)"
                  >
                    {Number.isInteger(modelData.days_won) ? modelData.days_won : modelData.days_won.toFixed(1)}
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-gray-500/20 text-gray-400">
                    {modelData.correct}
                  </span>
                </div>
              )}
            </button>
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

      {/* Footer note */}
      <div className="mt-3 pt-3 border-t border-border">
        <p className="text-[10px] text-muted-foreground/70">
          Comparando accuracy acumulada desde {formatStartDate(data.start_date)}.
          Badge colors: <span className="text-yellow-500">amarillo</span> = días ganados (empates se dividen),
          <span className="text-gray-400"> gris</span> = predicciones correctas totales.
          Empates ±0.5pp → co-picks.
        </p>
      </div>
    </div>
  );
}
