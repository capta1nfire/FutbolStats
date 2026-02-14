"use client";

import { useRef, useEffect, useCallback } from "react";
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

      // Build series data
      const seriesData = countries.map((c) => ({
        name: ISO3_TO_ECHARTS[c.country_iso3] || c.country_name,
        value: c.coverage_total_pct,
        _raw: c,
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
          type: "piecewise",
          pieces: [
            { min: 85, max: 100, label: "85-100%", color: "#22c55e" },
            { min: 70, max: 84.9, label: "70-84%", color: "#15803d" },
            { min: 50, max: 69.9, label: "50-69%", color: "#0369a1" },
            { min: 25, max: 49.9, label: "25-49%", color: "#b45309" },
            { min: 0, max: 24.9, label: "0-24%", color: "#7f1d1d" },
          ],
          orient: "vertical",
          left: 16,
          bottom: 16,
          textStyle: { color: "#b7bcc2", fontSize: 11 },
          itemWidth: 14,
          itemHeight: 14,
        },
        series: [
          {
            type: "map",
            map: "world",
            roam: true,
            scaleLimit: { min: 1, max: 8 },
            label: { show: false },
            itemStyle: {
              areaColor: "#1c1e21",
              borderColor: "rgba(249,250,250,0.07)",
              borderWidth: 0.5,
            },
            emphasis: {
              label: { show: true, color: "#dee0e3", fontSize: 11 },
              itemStyle: {
                areaColor: "#282b2f",
                borderColor: "#4797FF",
                borderWidth: 2,
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
      className="w-full"
      style={{ height: 520, minHeight: 400 }}
    />
  );
}
