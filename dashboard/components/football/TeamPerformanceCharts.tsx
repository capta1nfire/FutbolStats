"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import { useTeamPerformance } from "@/lib/hooks";
import type { TeamPerformanceMatch } from "@/lib/types/performance";

const ReactApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

// Chart color palette (from globals.css --chart-N)
const COLORS = {
  blue: "#4797FF",
  green: "#37be5f",
  yellow: "#eab308",
  red: "#ef4444",
  cyan: "#06b6d4",
};

const GRID_STYLE = {
  borderColor: "rgba(255,255,255,0.05)",
  strokeDashArray: 3,
  xaxis: { lines: { show: false } },
  yaxis: { lines: { show: true } },
} as const;

const AXIS_LABEL_STYLE = { colors: "#6b7280", fontSize: "11px" };

function roundLabel(r: string | null, idx: number): string {
  if (!r) return `J${idx + 1}`;
  // Extract numeric part like "Regular Season - 5" → "J5"
  const num = r.match(/(\d+)\s*$/);
  return num ? `J${num[1]}` : `J${idx + 1}`;
}

interface Props {
  teamId: number;
  leagueId: number;
  season: number | null;
}

export function TeamPerformanceCharts({ teamId, leagueId, season }: Props) {
  const { data, isLoading, isError } = useTeamPerformance(teamId, leagueId, season);

  const matches = data?.matches ?? [];
  const hasXG = data?.xg_source !== null && matches.some((m) => m.cumulative_xg_for !== null);
  const hasRating = matches.some((m) => m.avg_team_rating !== null);

  const labels = useMemo(() => matches.map((m, i) => roundLabel(m.round, i)), [matches]);

  // ── Chart 1: Cumulative Points (area) ──
  const pointsOptions: ApexCharts.ApexOptions = useMemo(
    () => ({
      chart: { type: "area", height: 220, background: "transparent", toolbar: { show: false }, fontFamily: "inherit" },
      theme: { mode: "dark" },
      colors: [COLORS.blue],
      stroke: { curve: "smooth", width: 2 },
      fill: { type: "gradient", gradient: { shadeIntensity: 1, opacityFrom: 0.35, opacityTo: 0.05, stops: [0, 90, 100] } },
      dataLabels: { enabled: false },
      legend: { show: false },
      grid: GRID_STYLE,
      xaxis: {
        categories: labels,
        labels: { style: AXIS_LABEL_STYLE, hideOverlappingLabels: true, rotateAlways: false },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tooltip: { enabled: false },
      },
      yaxis: {
        labels: { style: AXIS_LABEL_STYLE, formatter: (v: number) => String(Math.round(v)) },
      },
      tooltip: {
        theme: "dark",
        custom: ({ dataPointIndex }: { dataPointIndex: number }) => {
          const m = matches[dataPointIndex];
          if (!m) return "";
          const resultColor = m.result === "W" ? COLORS.green : m.result === "L" ? COLORS.red : COLORS.yellow;
          return `<div style="padding:6px 10px;background:#1f2937;border:1px solid #374151;border-radius:6px;font-size:12px">
            <div style="color:#9ca3af;font-size:10px;margin-bottom:4px">${m.match_date} · ${m.is_home ? "Local" : "Visitante"}</div>
            <div style="color:#fff;font-weight:600">${m.opponent_name}</div>
            <div style="margin-top:2px"><span style="color:${resultColor};font-weight:700">${m.result}</span> ${m.goals_for}-${m.goals_against} · <span style="color:${COLORS.blue}">${m.cumulative_points} pts</span></div>
          </div>`;
        },
      },
    }),
    [labels, matches]
  );

  const pointsSeries = useMemo(
    () => [{ name: "Puntos", data: matches.map((m) => m.cumulative_points) }],
    [matches]
  );

  // ── Chart 2: Goals For vs Against (bar) ──
  const goalsOptions: ApexCharts.ApexOptions = useMemo(
    () => ({
      chart: { type: "bar", height: 220, background: "transparent", toolbar: { show: false }, fontFamily: "inherit", stacked: false },
      theme: { mode: "dark" },
      colors: [COLORS.green, COLORS.red],
      plotOptions: { bar: { columnWidth: "60%", borderRadius: 2 } },
      dataLabels: { enabled: false },
      legend: { show: true, position: "top", labels: { colors: "#9ca3af" }, fontSize: "11px", markers: { size: 6, shape: "circle" as const } },
      grid: GRID_STYLE,
      xaxis: {
        categories: labels,
        labels: { style: AXIS_LABEL_STYLE, hideOverlappingLabels: true, rotateAlways: false },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tooltip: { enabled: false },
      },
      yaxis: {
        labels: { style: AXIS_LABEL_STYLE, formatter: (v: number) => String(Math.round(v)) },
      },
      tooltip: {
        theme: "dark",
        shared: true,
        intersect: false,
        custom: ({ dataPointIndex }: { dataPointIndex: number }) => {
          const m = matches[dataPointIndex];
          if (!m) return "";
          return `<div style="padding:6px 10px;background:#1f2937;border:1px solid #374151;border-radius:6px;font-size:12px">
            <div style="color:#9ca3af;font-size:10px;margin-bottom:4px">${m.match_date} vs ${m.opponent_name}</div>
            <div><span style="color:${COLORS.green}">GF: ${m.goals_for}</span> · <span style="color:${COLORS.red}">GC: ${m.goals_against}</span></div>
          </div>`;
        },
      },
    }),
    [labels, matches]
  );

  const goalsSeries = useMemo(
    () => [
      { name: "Goles a favor", data: matches.map((m) => m.goals_for) },
      { name: "Goles en contra", data: matches.map((m) => m.goals_against) },
    ],
    [matches]
  );

  // ── Chart 3: xG vs Goals cumulative (line) ──
  const xgOptions: ApexCharts.ApexOptions = useMemo(
    () => ({
      chart: { type: "line", height: 220, background: "transparent", toolbar: { show: false }, fontFamily: "inherit" },
      theme: { mode: "dark" },
      colors: [COLORS.green, COLORS.cyan, COLORS.red, "#f97316"],
      stroke: { curve: "smooth", width: [2, 2, 2, 2], dashArray: [0, 4, 0, 4] },
      dataLabels: { enabled: false },
      legend: { show: true, position: "top", labels: { colors: "#9ca3af" }, fontSize: "11px", markers: { size: 6, shape: "circle" as const } },
      grid: GRID_STYLE,
      xaxis: {
        categories: labels,
        labels: { style: AXIS_LABEL_STYLE, hideOverlappingLabels: true, rotateAlways: false },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tooltip: { enabled: false },
      },
      yaxis: {
        labels: { style: AXIS_LABEL_STYLE, formatter: (v: number) => v.toFixed(1) },
      },
      tooltip: {
        theme: "dark",
        shared: true,
        intersect: false,
        custom: ({ dataPointIndex }: { dataPointIndex: number }) => {
          const m = matches[dataPointIndex];
          if (!m) return "";
          const xgf = m.cumulative_xg_for !== null ? m.cumulative_xg_for.toFixed(2) : "—";
          const xga = m.cumulative_xg_against !== null ? m.cumulative_xg_against.toFixed(2) : "—";
          const gf = matches.slice(0, dataPointIndex + 1).reduce((s, x) => s + x.goals_for, 0);
          const gc = matches.slice(0, dataPointIndex + 1).reduce((s, x) => s + x.goals_against, 0);
          return `<div style="padding:6px 10px;background:#1f2937;border:1px solid #374151;border-radius:6px;font-size:12px">
            <div style="color:#9ca3af;font-size:10px;margin-bottom:4px">${m.match_date} vs ${m.opponent_name}</div>
            <div><span style="color:${COLORS.green}">GF: ${gf}</span> <span style="color:${COLORS.cyan}">xGF: ${xgf}</span></div>
            <div><span style="color:${COLORS.red}">GC: ${gc}</span> <span style="color:#f97316">xGA: ${xga}</span></div>
          </div>`;
        },
      },
    }),
    [labels, matches]
  );

  const xgSeries = useMemo(() => {
    let cumGF = 0;
    let cumGC = 0;
    const goalsForCum = matches.map((m) => { cumGF += m.goals_for; return cumGF; });
    const goalsCum = matches.map((m) => { cumGC += m.goals_against; return cumGC; });
    return [
      { name: "Goles F (acum)", data: goalsForCum },
      { name: "xG F (acum)", data: matches.map((m) => m.cumulative_xg_for) },
      { name: "Goles C (acum)", data: goalsCum },
      { name: "xG C (acum)", data: matches.map((m) => m.cumulative_xg_against) },
    ];
  }, [matches]);

  // ── Chart 4: Avg Team Rating (line) ──
  const ratingOptions: ApexCharts.ApexOptions = useMemo(
    () => ({
      chart: { type: "line", height: 220, background: "transparent", toolbar: { show: false }, fontFamily: "inherit" },
      theme: { mode: "dark" },
      colors: [COLORS.yellow],
      stroke: { curve: "smooth", width: 2 },
      dataLabels: { enabled: false },
      legend: { show: false },
      grid: GRID_STYLE,
      annotations: {
        yaxis: [{ y: 7.0, borderColor: "rgba(255,255,255,0.15)", strokeDashArray: 4, label: { text: "7.0", style: { color: "#6b7280", background: "transparent", fontSize: "10px" }, position: "left" } }],
      },
      xaxis: {
        categories: labels,
        labels: { style: AXIS_LABEL_STYLE, hideOverlappingLabels: true, rotateAlways: false },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tooltip: { enabled: false },
      },
      yaxis: {
        min: 5.5,
        max: 8.5,
        tickAmount: 6,
        labels: { style: AXIS_LABEL_STYLE, formatter: (v: number) => v.toFixed(1) },
      },
      tooltip: {
        theme: "dark",
        custom: ({ dataPointIndex }: { dataPointIndex: number }) => {
          const m = matches[dataPointIndex];
          if (!m) return "";
          const rating = m.avg_team_rating !== null ? m.avg_team_rating.toFixed(2) : "—";
          return `<div style="padding:6px 10px;background:#1f2937;border:1px solid #374151;border-radius:6px;font-size:12px">
            <div style="color:#9ca3af;font-size:10px;margin-bottom:4px">${m.match_date} vs ${m.opponent_name}</div>
            <div style="color:${COLORS.yellow};font-weight:600">Rating: ${rating}</div>
          </div>`;
        },
      },
    }),
    [labels, matches]
  );

  const ratingSeries = useMemo(
    () => [{ name: "Rating", data: matches.map((m) => m.avg_team_rating ?? null) }],
    [matches]
  );

  // ── Render ──

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16 text-muted-foreground gap-2">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-sm">Cargando rendimiento...</span>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex items-center justify-center py-16 text-muted-foreground text-sm">
        Error al cargar datos de rendimiento
      </div>
    );
  }

  if (matches.length === 0) {
    return (
      <div className="flex items-center justify-center py-16 text-muted-foreground text-sm">
        Sin partidos completados esta temporada
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-4">
      {/* Chart 1: Cumulative Points */}
      <section>
        <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 px-1">
          Puntos acumulados
        </h4>
        <ReactApexChart type="area" height={220} options={pointsOptions} series={pointsSeries} />
      </section>

      {/* Chart 2: Goals For vs Against */}
      <section>
        <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 px-1">
          Goles por jornada
        </h4>
        <ReactApexChart type="bar" height={220} options={goalsOptions} series={goalsSeries} />
      </section>

      {/* Chart 3: xG vs Goals (conditional) */}
      {hasXG && (
        <section>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 px-1">
            xG vs Goles (acumulado)
            <span className="ml-2 text-[10px] font-normal normal-case text-muted-foreground/60">
              fuente: {data.xg_source}
            </span>
          </h4>
          <ReactApexChart type="line" height={220} options={xgOptions} series={xgSeries} />
        </section>
      )}

      {/* Chart 4: Avg Team Rating (conditional) */}
      {hasRating && (
        <section>
          <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 px-1">
            Rating promedio del XI
          </h4>
          <ReactApexChart type="line" height={220} options={ratingOptions} series={ratingSeries} />
        </section>
      )}
    </div>
  );
}
