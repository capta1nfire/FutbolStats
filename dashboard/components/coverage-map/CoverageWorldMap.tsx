"use client";

import { useRef, useEffect, useCallback, useMemo, useState } from "react";
import { geoMercator } from "d3-geo";
import Image from "next/image";
import type { CoverageCountry, CoverageLeague } from "@/lib/types/coverage-map";

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

const ISO3_TO_ISO2: Record<string, string> = {
  ARG: "ar", BEL: "be", BOL: "bo", BRA: "br", CHL: "cl",
  COL: "co", ECU: "ec", GBR: "gb", FRA: "fr", DEU: "de",
  ITA: "it", MEX: "mx", NLD: "nl", PRY: "py", PER: "pe",
  PRT: "pt", SAU: "sa", ESP: "es", TUR: "tr", URY: "uy",
  USA: "us", VEN: "ve",
};

// Approximate geographic centers [lon, lat] for badge positioning
const COUNTRY_CENTERS: Record<string, [number, number]> = {
  ARG: [-64, -34], BEL: [4.5, 50.5], BOL: [-65, -17], BRA: [-51, -14],
  CHL: [-71, -35], COL: [-74, 4], ECU: [-78, -1.8], GBR: [-3, 54],
  FRA: [2, 46], DEU: [10, 51], ITA: [12, 42], MEX: [-102, 23],
  NLD: [5, 52], PRY: [-58, -23], PER: [-75, -10], PRT: [-8, 39.5],
  SAU: [45, 24], ESP: [-4, 40], TUR: [35, 39], URY: [-56, -33],
  USA: [-98, 38], VEN: [-66, 7],
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
  const [badgePos, setBadgePos] = useState<{ x: number; y: number } | null>(null);
  const [hoveredCountry, setHoveredCountry] = useState<string | null>(null);

  // Badge shows for selected country, or hovered country as fallback
  const badgeCountry = selectedCountry || hoveredCountry;

  // Ref-based badge updater — avoids triggering chart effect on hover changes
  const badgeCountryRef = useRef(badgeCountry);
  badgeCountryRef.current = badgeCountry;

  const updateBadgePos = useCallback(() => {
    const chart = chartRef.current;
    const iso = badgeCountryRef.current;
    if (!chart || !iso) {
      setBadgePos(null);
      return;
    }
    const center = COUNTRY_CENTERS[iso];
    if (!center) { setBadgePos(null); return; }
    try {
      const px = chart.convertToPixel({ seriesIndex: 0 }, center);
      if (px) setBadgePos({ x: px[0], y: px[1] });
    } catch {
      setBadgePos(null);
    }
  }, []); // stable — reads from ref

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
    setHoveredCountry(null);
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
      // When a country is selected, dim others to 35% opacity
      const dimmed = selectedCountry !== null;
      const seriesData = countries.map((c) => {
        const isSelected = selectedCountry === c.country_iso3;
        const alpha = dimmed && !isSelected ? 0.35 : 1.0;
        return {
          name: ISO3_TO_ECHARTS[c.country_iso3] || c.country_name,
          value: c.coverage_total_pct,
          _raw: c,
          itemStyle: {
            areaColor: bandColor(c.coverage_total_pct, alpha),
            ...(isSelected && { shadowColor: "rgba(0,0,0,0.4)", shadowBlur: 8 }),
          },
          emphasis: { itemStyle: { areaColor: bandColor(c.coverage_total_pct, 1.0) } },
        };
      });

      chartRef.current.setOption({
        backgroundColor: "transparent",
        tooltip: {
          trigger: "item",
          backgroundColor: "#232326",
          borderColor: "transparent",
          borderWidth: 0,
          padding: [8, 16],
          textStyle: { color: "#dee0e3", fontSize: 11, lineHeight: 16 },
          extraCssText:
            "border-radius:4px;box-shadow:0 8px 24px rgba(0,0,0,1),0 0 1px rgba(249,250,250,0.08);",
          formatter: (p: Record<string, unknown>) => {
            const d = p.data as { _raw?: CoverageCountry; value?: number };
            if (!d?._raw) return `${p.name}<br/>No data`;
            const c = d._raw;
            const countryLeagues = leagues.filter((l) => l.country_iso3 === c.country_iso3);
            const leagueLines = countryLeagues.map((l) => {
              const logo = l.logo_url
                ? `<img src="${l.logo_url}" width="14" height="14" style="vertical-align:middle;margin-right:5px">`
                : "";
              return `<div style="display:flex;align-items:center;gap:2px">${logo}<b>${l.league_name}</b></div>`;
            });
            const header = leagueLines.length
              ? leagueLines.join("")
              : `<b>${c.country_name}</b>`;
            return [
              `<div style="margin-bottom:4px">${header}</div>`,
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
              areaColor: "#282b2f",
              borderColor: "rgba(249,250,250,0.12)",
              borderWidth: 0.5,
            },
            emphasis: {
              label: { show: false },
              itemStyle: {
                areaColor: "#353840",
                borderColor: "rgba(249,250,250,0.25)",
                borderWidth: 1,
              },
            },
            data: seriesData,
          },
        ],
      });

      // Click handler
      chartRef.current.off("click");
      chartRef.current.on("click", handleClick);

      // Hover handler for badge
      chartRef.current.off("mouseover");
      chartRef.current.on("mouseover", (params: { name?: string }) => {
        if (!params.name) return;
        const iso = ECHARTS_TO_ISO3[params.name];
        if (iso) setHoveredCountry(iso);
      });
      chartRef.current.off("mouseout");
      chartRef.current.on("mouseout", () => setHoveredCountry(null));

      // Update badge on zoom/pan
      chartRef.current.off("georoam");
      chartRef.current.on("georoam", updateBadgePos);

      // Force resize after layout settles, then position badge
      requestAnimationFrame(() => {
        chartRef.current?.resize();
        updateBadgePos();
      });
    }

    init().catch(console.error);

    // ResizeObserver for responsive
    const observer = new ResizeObserver(() => {
      chartRef.current?.resize();
      updateBadgePos();
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
  }, [countries, selectedCountry, handleClick, updateBadgePos]);

  // Update badge position when badgeCountry changes (lightweight, no chart rebuild)
  useEffect(() => {
    updateBadgePos();
  }, [badgeCountry, updateBadgePos]);

  const badgeLabel = useMemo(() => {
    if (!badgeCountry) return null;
    const country = countries.find((c) => c.country_iso3 === badgeCountry);
    const name = country?.country_name || ISO3_TO_ECHARTS[badgeCountry] || badgeCountry;
    const iso2 = ISO3_TO_ISO2[badgeCountry];
    return { name, iso2 };
  }, [badgeCountry, countries]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      {badgeLabel && badgePos && (
        <div
          className="absolute z-10 pointer-events-none -translate-x-1/2 -translate-y-full"
          style={{ left: badgePos.x, top: badgePos.y - 8 }}
        >
          <div className="relative inline-flex items-center gap-1.5 bg-surface-elevated/90 backdrop-blur-sm rounded-lg px-3 py-1.5" style={{ boxShadow: "0 2px 8px rgba(0,0,0,0.4)" }}>
            {badgeLabel.iso2 && (
              <Image
                src={`/flags/${badgeLabel.iso2}.svg`}
                alt=""
                width={16}
                height={16}
                className="rounded-full object-cover"
              />
            )}
            <span className="text-xs font-medium text-foreground">
              {badgeLabel.name}
            </span>
            {/* Arrow */}
            <div
              className="absolute left-1/2 -translate-x-1/2 -bottom-1.5 w-0 h-0"
              style={{
                borderLeft: "6px solid transparent",
                borderRight: "6px solid transparent",
                borderTop: "6px solid rgba(35,35,38,0.9)",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
