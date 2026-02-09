"use client";

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useBenchmarkMatrix } from "@/lib/hooks/use-benchmark-matrix";
import { Loader } from "@/components/ui/loader";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  BenchmarkCell,
  BenchmarkMetric,
  BenchmarkPeriod,
  BenchmarkSource,
  BenchmarkLeague,
} from "@/lib/types/benchmark-matrix";
import {
  BENCHMARK_PERIODS,
  BENCHMARK_METRICS,
} from "@/lib/types/benchmark-matrix";

// ─── Helpers ────────────────────────────────────────────────────────────────

function getCellValue(cell: BenchmarkCell | undefined, metric: BenchmarkMetric): number | null {
  if (!cell) return null;
  return cell[metric];
}

function formatCellValue(value: number | null, metric: BenchmarkMetric): string {
  if (value === null) return "--";
  if (metric === "skill_pct") return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
  if (metric === "delta_brier") return `${value > 0 ? "+" : ""}${value.toFixed(4)}`;
  return `${value > 0 ? "+" : ""}${value.toFixed(4)}`;
}

/**
 * Color for cell text based on metric value.
 * For skill_pct: positive = good (green), negative = bad (red)
 * For delta_brier/delta_logloss: negative = good (green, means lower than Pinnacle), positive = bad
 */
function getCellColor(value: number | null, metric: BenchmarkMetric): string {
  if (value === null) return "text-muted-foreground";
  const isGood = metric === "skill_pct" ? value > 0 : value < 0;
  const isBad = metric === "skill_pct" ? value < 0 : value > 0;
  const magnitude = Math.abs(value);
  const threshold = metric === "skill_pct" ? 5 : 0.01;

  if (isGood && magnitude > threshold) return "text-emerald-400";
  if (isGood) return "text-emerald-300/80";
  if (isBad && magnitude > threshold) return "text-red-400";
  if (isBad) return "text-red-300/80";
  return "text-muted-foreground";
}

function getConfidenceOpacity(tier: string): string {
  if (tier === "insufficient") return "opacity-30";
  if (tier === "low") return "opacity-60";
  return "";
}

function formatCI(cell: BenchmarkCell): string {
  if (cell.ci_lo === null || cell.ci_hi === null) return "";
  return `CI95: [${cell.ci_lo > 0 ? "+" : ""}${cell.ci_lo.toFixed(4)}, ${cell.ci_hi > 0 ? "+" : ""}${cell.ci_hi.toFixed(4)}]`;
}

// ─── Cell Component ─────────────────────────────────────────────────────────

function MatrixCell({
  cell,
  metric,
  sourceName,
  leagueName,
}: {
  cell: BenchmarkCell | undefined;
  metric: BenchmarkMetric;
  sourceName: string;
  leagueName: string;
}) {
  if (!cell || cell.n === 0) {
    return (
      <td className="px-2 py-1.5 text-center text-[11px] text-muted-foreground/40 whitespace-nowrap">
        --
      </td>
    );
  }

  // Guardrail: don't render numeric values for insufficient N (<20)
  if (cell.confidence_tier === "insufficient") {
    const tooltipLines = [
      `${sourceName} vs Pinnacle`,
      `League: ${leagueName}`,
      `Insufficient data (N=${cell.n})`,
    ];
    return (
      <td className="px-2 py-1.5 text-center text-[11px] text-muted-foreground/40 whitespace-nowrap">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="cursor-default">--</span>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-xs">
            <div className="text-xs space-y-0.5">
              {tooltipLines.map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
          </TooltipContent>
        </Tooltip>
      </td>
    );
  }

  const value = getCellValue(cell, metric);
  const displayValue = formatCellValue(value, metric);
  const color = getCellColor(value, metric);
  const opacity = getConfidenceOpacity(cell.confidence_tier);
  const ci = formatCI(cell);

  const tooltipLines = [
    `${sourceName} vs Pinnacle`,
    `League: ${leagueName}`,
    `N: ${cell.n} (${cell.confidence_tier})`,
    `Brier: ${cell.brier_abs?.toFixed(4) ?? "--"} (Pin: ${cell.pinnacle_brier?.toFixed(4) ?? "--"})`,
    `LogLoss: ${cell.logloss_abs?.toFixed(4) ?? "--"} (Pin: ${cell.pinnacle_logloss?.toFixed(4) ?? "--"})`,
    `Skill%: ${cell.skill_pct !== null ? `${cell.skill_pct > 0 ? "+" : ""}${cell.skill_pct.toFixed(1)}%` : "--"}`,
  ];
  if (ci) tooltipLines.push(ci);

  return (
    <td className={cn("px-2 py-1.5 text-center whitespace-nowrap", opacity)}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              "text-[11px] font-mono cursor-default",
              color,
              cell.confidence_tier === "low" && "after:content-['*'] after:text-muted-foreground after:text-[9px]"
            )}
          >
            {displayValue}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <div className="text-xs space-y-0.5">
            {tooltipLines.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </div>
        </TooltipContent>
      </Tooltip>
    </td>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

export function BenchmarkMatrix() {
  const [period, setPeriod] = useState<BenchmarkPeriod>("30d");
  const [metric, setMetric] = useState<BenchmarkMetric>("skill_pct");

  const { data, isLoading, error } = useBenchmarkMatrix(period);

  // Separate sources into models and bookmakers
  const { modelSources, bookieSources } = useMemo(() => {
    if (!data) return { modelSources: [] as BenchmarkSource[], bookieSources: [] as BenchmarkSource[] };
    return {
      modelSources: data.sources.filter((s) => s.kind === "model"),
      bookieSources: data.sources.filter((s) => s.kind === "bookmaker"),
    };
  }, [data]);

  const allSources = useMemo(
    () => [...modelSources, ...bookieSources],
    [modelSources, bookieSources]
  );

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader size="md" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-2">
          <p className="text-sm text-muted-foreground">
            Failed to load benchmark matrix
          </p>
          <p className="text-xs text-muted-foreground/70">
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Controls bar */}
        <div className="h-12 flex items-center gap-4 px-6 border-b border-border shrink-0">
          <h2 className="text-sm font-semibold text-foreground mr-auto">
            Benchmark Matrix
            <span className="ml-2 text-xs text-muted-foreground font-normal">
              vs {data.anchor}
            </span>
          </h2>

          {/* Metric selector */}
          <div className="flex items-center gap-1.5">
            {BENCHMARK_METRICS.map((m) => (
              <button
                key={m.value}
                onClick={() => setMetric(m.value)}
                className={cn(
                  "px-2 py-1 text-[11px] rounded transition-colors",
                  metric === m.value
                    ? "bg-accent text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Period selector */}
          <Select value={period} onValueChange={(v) => setPeriod(v as BenchmarkPeriod)}>
            <SelectTrigger className="w-[100px] h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {BENCHMARK_PERIODS.map((p) => (
                <SelectItem key={p.value} value={p.value} className="text-xs">
                  {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Matrix table */}
        <div className="flex-1 overflow-auto">
          <table className="w-full border-collapse text-xs">
            <thead className="sticky top-0 z-20 bg-surface">
              {/* Row 1: Group headers */}
              <tr className="border-b border-border">
                <th
                  className="sticky left-0 z-30 bg-surface px-3 py-1.5 text-left text-[10px] text-muted-foreground font-normal uppercase tracking-wider"
                  rowSpan={2}
                >
                  League
                </th>
                {modelSources.length > 0 && (
                  <th
                    colSpan={modelSources.length}
                    className="px-2 py-1 text-center text-[10px] text-muted-foreground font-normal uppercase tracking-wider border-l border-border"
                  >
                    Models
                  </th>
                )}
                {bookieSources.length > 0 && (
                  <th
                    colSpan={bookieSources.length}
                    className="px-2 py-1 text-center text-[10px] text-muted-foreground font-normal uppercase tracking-wider border-l border-border"
                  >
                    Bookmakers
                  </th>
                )}
              </tr>

              {/* Row 2: Individual source names */}
              <tr className="border-b border-border">
                {allSources.map((source, i) => {
                  const isFirstBookie = i === modelSources.length && modelSources.length > 0;
                  return (
                    <th
                      key={source.key}
                      className={cn(
                        "px-2 py-1.5 text-center text-[10px] font-normal text-muted-foreground whitespace-nowrap",
                        (i === 0 || isFirstBookie) && "border-l border-border"
                      )}
                    >
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="cursor-default">
                            {source.kind === "model"
                              ? source.label.replace("Model A ", "").replace("Shadow ", "S:")
                              : source.label}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" className="text-xs">
                          <div>{source.label}</div>
                          <div className="text-muted-foreground">
                            {source.total_matches} matches
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    </th>
                  );
                })}
              </tr>
            </thead>

            <tbody>
              {/* League rows */}
              {data.leagues.map((league: BenchmarkLeague) => (
                <LeagueRow
                  key={league.league_id}
                  league={league}
                  sources={allSources}
                  modelSourcesCount={modelSources.length}
                  cells={data.cells}
                  metric={metric}
                />
              ))}

              {/* Global row */}
              <tr className="border-t-2 border-border bg-surface/50">
                <td className="sticky left-0 z-10 bg-surface/50 px-3 py-2 font-semibold text-foreground whitespace-nowrap">
                  GLOBAL
                  <span className="ml-2 text-[10px] text-muted-foreground font-normal">
                    ({data.leagues.reduce((s: number, l: BenchmarkLeague) => s + l.total_resolved, 0)} matches)
                  </span>
                </td>
                {allSources.map((source, i) => {
                  const isFirstBookie = i === modelSources.length && modelSources.length > 0;
                  return (
                    <td
                      key={source.key}
                      className={cn(
                        "px-2 py-2 text-center whitespace-nowrap",
                        (i === 0 || isFirstBookie) && "border-l border-border"
                      )}
                    >
                      <GlobalCell
                        cell={data.global_row[source.key]}
                        metric={metric}
                        sourceName={source.label}
                      />
                    </td>
                  );
                })}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </TooltipProvider>
  );
}

// ─── League Row ─────────────────────────────────────────────────────────────

function LeagueRow({
  league,
  sources,
  modelSourcesCount,
  cells,
  metric,
}: {
  league: BenchmarkLeague;
  sources: BenchmarkSource[];
  modelSourcesCount: number;
  cells: Record<string, BenchmarkCell>;
  metric: BenchmarkMetric;
}) {
  return (
    <tr className="border-b border-border/50 hover:bg-accent/30 transition-colors">
      <td className="sticky left-0 z-10 bg-background px-3 py-1.5 whitespace-nowrap">
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-foreground font-medium truncate max-w-[200px]">
            {league.name}
          </span>
          <span className="text-[9px] text-muted-foreground shrink-0">
            {league.country}
          </span>
          <span className="text-[9px] text-muted-foreground/60 shrink-0">
            N={league.total_resolved}
          </span>
        </div>
      </td>
      {sources.map((source, i) => {
        const cellKey = `${league.league_id}:${source.key}`;
        const cell = cells[cellKey];
        const isFirstBookie = i === modelSourcesCount && modelSourcesCount > 0;
        return (
          <td
            key={source.key}
            className={cn(
              (i === 0 || isFirstBookie) && "border-l border-border"
            )}
          >
            <MatrixCell
              cell={cell}
              metric={metric}
              sourceName={source.label}
              leagueName={league.name}
            />
          </td>
        );
      })}
    </tr>
  );
}

// ─── Global Cell ────────────────────────────────────────────────────────────

function GlobalCell({
  cell,
  metric,
  sourceName,
}: {
  cell: BenchmarkCell | undefined;
  metric: BenchmarkMetric;
  sourceName: string;
}) {
  if (!cell || cell.n === 0 || cell.confidence_tier === "insufficient") {
    return <span className="text-[11px] text-muted-foreground/40">--</span>;
  }

  const value = getCellValue(cell, metric);
  const displayValue = formatCellValue(value, metric);
  const color = getCellColor(value, metric);
  const ci = formatCI(cell);

  const tooltipLines = [
    `${sourceName} vs Pinnacle (Global)`,
    `N: ${cell.n} (${cell.confidence_tier})`,
    `Brier: ${cell.brier_abs?.toFixed(4) ?? "--"} (Pin: ${cell.pinnacle_brier?.toFixed(4) ?? "--"})`,
    `LogLoss: ${cell.logloss_abs?.toFixed(4) ?? "--"} (Pin: ${cell.pinnacle_logloss?.toFixed(4) ?? "--"})`,
    `Skill%: ${cell.skill_pct !== null ? `${cell.skill_pct > 0 ? "+" : ""}${cell.skill_pct.toFixed(1)}%` : "--"}`,
  ];
  if (ci) tooltipLines.push(ci);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className={cn("text-[11px] font-mono font-semibold cursor-default", color)}>
          {displayValue}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs">
        <div className="text-xs space-y-0.5">
          {tooltipLines.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
