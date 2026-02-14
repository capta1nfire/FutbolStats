"use client";

import { useRef, useEffect, useCallback } from "react";
import { geoNaturalEarth1 } from "d3-geo";
import type { CoverageCountry } from "@/lib/types/coverage-map";

// --- Country name mapping ---

const ISO3_TO_ECHARTS: Record<string, string> = {
  ARG: "Argentina",
  BEL: "Belgium",
  BOL: "Bolivia",
  BRA: "Brazil",
  CHL: "Chile",
  COL: "Colombia",
  ECU: "Ecuador",
  GBR: "United Kingdom",
  FRA: "France",
  DEU: "Germany",
  ITA: "Italy",
  MEX: "Mexico",
  NLD: "Netherlands",
  PRY: "Paraguay",
  PER: "Peru",
  PRT: "Portugal",
  SAU: "Saudi Arabia",
  ESP: "Spain",
  TUR: "Turkey",
  URY: "Uruguay",
  USA: "United States of America",
  VEN: "Venezuela",
};

const ECHARTS_TO_ISO3: Record<string, string> = {
  ...Object.fromEntries(
    Object.entries(ISO3_TO_ECHARTS).map(([k, v]) => [v, k])
  ),
  "United States": "USA",
};

const TIER_LABELS: Record<string, string> = {
  xi_odds_xg: "XI+Odds+xG",
  odds_xg: "Odds+xG",
  xg: "xG",
  odds: "Odds",
  base: "Base",
  insufficient_data: "Insufficient",
};

// --- Component ---

interface CoverageWorldMapProps {
  countries: CoverageCountry[];
  onCountryClick: (iso3: string | null) => void;
  selectedCountry: string | null;
}

// Module-level lazy state
let echartsModule: typeof import("echarts") | null = null;
let geoRegistered = false;

export function CoverageWorldMap({
  countries,
  onCountryClick,
  selectedCountry,
}: CoverageWorldMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof import("echarts")["init"]> | null>(
    null
  );

  const handleClick = useCallback(
    (params: { name?: string }) => {
      if (!params.name) return;
      const iso = ECHARTS_TO_ISO3[params.name];
      if (!iso) return;
      onCountryClick(selectedCountry === iso ? null : iso);
    },
    [onCountryClick, selectedCountry]
  );

  useEffect(() => {
    if (!containerRef.current) return;
    let disposed = false;

    async function init() {
      // Lazy-load echarts
      if (!echartsModule) {
        echartsModule = await import("echarts");
      }
      const echarts = echartsModule;

      // Register world map once
      if (!geoRegistered) {
        const res = await fetch("/geo/world.json");
        if (!res.ok) throw new Error("Failed to load world GeoJSON");
        const geoJson = await res.json();
        echarts.registerMap("world", geoJson);
        geoRegistered = true;
      }

      if (disposed || !containerRef.current) return;

      // Init chart
      if (!chartRef.current) {
        chartRef.current = echarts.init(containerRef.current, undefined, {
          renderer: "canvas",
        });
      }

      // Natural Earth projection — proper globe-like appearance
      const projection = geoNaturalEarth1();

      // Color bands: [min, max, r, g, b]
      const BANDS: [number, number, number, number, number][] = [
        [85, 100, 34, 197, 94],
        [70, 85, 21, 128, 61],
        [50, 70, 3, 105, 161],
        [25, 50, 180, 83, 9],
        [0, 25, 127, 29, 29],
      ];

      function bandColor(pct: number, alpha: number): string {
        const b = BANDS.find(([min]) => pct >= min) || BANDS[BANDS.length - 1];
        return `rgba(${b[2]}, ${b[3]}, ${b[4]}, ${alpha})`;
      }

      // Build series data with per-item colors
      const seriesData = countries.map((c) => ({
        name: ISO3_TO_ECHARTS[c.country_iso3] || c.country_name,
        value: c.coverage_total_pct,
        _raw: c,
        itemStyle: { areaColor: bandColor(c.coverage_total_pct, 0.55) },
        emphasis: { itemStyle: { areaColor: bandColor(c.coverage_total_pct, 0.80) } },
      }));

      chartRef.current.setOption({
        backgroundColor: "transparent",
        tooltip: {
          trigger: "item",
          backgroundColor: "#232326",
          borderColor: "rgba(249,250,250,0.10)",
          textStyle: { color: "#dee0e3", fontSize: 12 },
          formatter: (p: Record<string, unknown>) => {
            const d = p.data as { _raw?: CoverageCountry; value?: number };
            if (!d?._raw) return `${p.name}<br/>No data`;
            const c = d._raw;
            return [
              `<b>${c.country_name}</b>`,
              `Coverage: <b>${c.coverage_total_pct}%</b>`,
              `Tier: ${TIER_LABELS[c.universe_tier] || c.universe_tier}`,
              `Leagues: ${c.league_count}`,
              `Matches: ${c.eligible_matches.toLocaleString()}`,
              `<span style="color:#b7bcc2">P0: ${c.p0_pct}% · P1: ${c.p1_pct}% · P2: ${c.p2_pct}%</span>`,
            ].join("<br/>");
          },
        },
        visualMap: {
          show: false,
          type: "piecewise",
          pieces: [
            { min: 85, max: 100, color: "rgba(34, 197, 94, 0.55)" },
            { min: 70, max: 84.9, color: "rgba(21, 128, 61, 0.55)" },
            { min: 50, max: 69.9, color: "rgba(3, 105, 161, 0.55)" },
            { min: 25, max: 49.9, color: "rgba(180, 83, 9, 0.55)" },
            { min: 0, max: 24.9, color: "rgba(127, 29, 29, 0.55)" },
          ],
        },
        series: [
          {
            type: "map",
            map: "world",
            projection: {
              project: (point: number[]) => projection(point as [number, number]) as number[],
              unproject: (point: number[]) => projection.invert!(point as [number, number]) as number[],
            },
            roam: true,
            scaleLimit: { min: 1, max: 8 },
            label: { show: false },
            itemStyle: {
              areaColor: "rgba(28, 30, 33, 0.4)",
              borderColor: "rgba(249,250,250,0.07)",
              borderWidth: 0.5,
            },
            emphasis: {
              label: { show: true, color: "#dee0e3", fontSize: 11 },
              itemStyle: {
                borderColor: "rgba(249,250,250,0.25)",
                borderWidth: 1.5,
              },
            },
            select: {
              label: { show: true, color: "#fff" },
              itemStyle: { borderColor: "#4797FF", borderWidth: 2 },
            },
            data: seriesData,
          },
        ],
      });

      // Click handler
      chartRef.current.off("click");
      chartRef.current.on("click", handleClick);

      // Force resize after layout settles
      requestAnimationFrame(() => chartRef.current?.resize());
    }

    init().catch(console.error);

    // ResizeObserver for responsive
    const observer = new ResizeObserver(() => {
      chartRef.current?.resize();
    });
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      disposed = true;
      observer.disconnect();
      if (chartRef.current) {
        chartRef.current.dispose();
        chartRef.current = null;
      }
    };
  }, [countries, handleClick]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
    />
  );
}
