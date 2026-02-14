"use client";

import { useRef, useEffect, useCallback } from "react";
import { geoMercator } from "d3-geo";
import type { CoverageCountry, CoverageLeague } from "@/lib/types/coverage-map";

// --- ISO3 → ISO2 for flag SVGs ---

const ISO3_TO_ISO2: Record<string, string> = {
  ARG: "ar", BEL: "be", BOL: "bo", BRA: "br", CHL: "cl",
  COL: "co", ECU: "ec", GBR: "gb", FRA: "fr", DEU: "de",
  ITA: "it", MEX: "mx", NLD: "nl", PRY: "py", PER: "pe",
  PRT: "pt", SAU: "sa", ESP: "es", TUR: "tr", URY: "uy",
  USA: "us", VEN: "ve",
};

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
  leagues: CoverageLeague[];
  onCountryClick: (iso3: string | null) => void;
  selectedCountry: string | null;
}

// Module-level lazy state
let echartsModule: typeof import("echarts") | null = null;
let geoRegistered = false;

export function CoverageWorldMap({
  countries,
  leagues,
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

      // Mercator projection — same as Google Maps / MapTiler / UniFi
      const projection = geoMercator();

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
        itemStyle: { areaColor: bandColor(c.coverage_total_pct, 1.0) },
        emphasis: { itemStyle: { areaColor: bandColor(c.coverage_total_pct, 1.0) } },
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
            const iso2 = ISO3_TO_ISO2[c.country_iso3] || "";
            const flag = iso2
              ? `<img src="/flags/${iso2}.svg" width="16" height="16" style="border-radius:50%;vertical-align:middle;margin-right:6px">`
              : "";
            const countryLeagues = leagues.filter((l) => l.country_iso3 === c.country_iso3);
            const leagueLines = countryLeagues.map((l) => {
              const logo = l.logo_url
                ? `<img src="${l.logo_url}" width="16" height="16" style="vertical-align:middle;margin-right:4px">`
                : "";
              return `${logo}${l.league_name}`;
            });
            const leagueHtml = leagueLines.length
              ? leagueLines.join("<br/>")
              : c.country_name;
            return [
              `<div style="display:flex;align-items:center;margin-bottom:4px">${flag}<b>${leagueHtml}</b></div>`,
              `Coverage: <b>${c.coverage_total_pct}%</b>`,
              `Tier: ${TIER_LABELS[c.universe_tier] || c.universe_tier}`,
              `Matches: ${c.eligible_matches.toLocaleString()}`,
              `<span style="color:#b7bcc2">P0: ${c.p0_pct}% · P1: ${c.p1_pct}% · P2: ${c.p2_pct}%</span>`,
            ].join("<br/>");
          },
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
            layoutCenter: ["50%", "38%"],
            layoutSize: "189%",
            scaleLimit: { min: 1, max: 8 },
            label: { show: false },
            itemStyle: {
              areaColor: "#2d3039",
              borderColor: "rgba(249,250,250,0.12)",
              borderWidth: 0.5,
            },
            emphasis: {
              label: { show: true, color: "#dee0e3", fontSize: 11 },
              itemStyle: {
                areaColor: "#383d47",
                borderColor: "rgba(249,250,250,0.25)",
                borderWidth: 1,
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
