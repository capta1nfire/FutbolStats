"use client";

import { useState, useCallback, useMemo } from "react";
import { useCoverageMap } from "@/lib/hooks/use-coverage-map";
import { useIsDesktop } from "@/lib/hooks/use-media-query";
import { CoverageWorldMap } from "@/components/coverage-map";
import { DetailDrawer } from "@/components/shell/DetailDrawer";
import { Loader } from "@/components/ui/loader";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type {
  CoverageWindow,
  CoverageMapData,
  CoverageLeague,
} from "@/lib/types/coverage-map";
import { Globe } from "lucide-react";
import Image from "next/image";

// --- Constants ---

const ISO3_TO_ISO2: Record<string, string> = {
  ARG: "ar", BEL: "be", BOL: "bo", BRA: "br", CHL: "cl",
  COL: "co", ECU: "ec", GBR: "gb", FRA: "fr", DEU: "de",
  ITA: "it", MEX: "mx", NLD: "nl", PRY: "py", PER: "pe",
  PRT: "pt", SAU: "sa", ESP: "es", TUR: "tr", URY: "uy",
  USA: "us", VEN: "ve",
};

const WINDOW_OPTIONS: { value: CoverageWindow; label: string }[] = [
  { value: "current_season", label: "Current Season" },
  { value: "prev_season", label: "Previous Season" },
  { value: "prev_season_2", label: "2 Seasons Ago" },
  { value: "since_2023", label: "All (Since 2023)" },
];

const LEGEND_BANDS = [
  { label: "85-100%", color: "rgba(34, 197, 94, 0.55)" },
  { label: "70-84%", color: "rgba(21, 128, 61, 0.55)" },
  { label: "50-69%", color: "rgba(3, 105, 161, 0.55)" },
  { label: "25-49%", color: "rgba(180, 83, 9, 0.55)" },
  { label: "0-24%", color: "rgba(127, 29, 29, 0.55)" },
];

const TIER_LABELS: Record<string, string> = {
  xi_odds_xg: "XI+Odds+xG",
  odds_xg: "Odds+xG",
  xg: "xG",
  odds: "Odds",
  base: "Base",
  insufficient_data: "Insufficient",
};

const TIER_COLORS: Record<string, { bg: string; text: string }> = {
  xi_odds_xg: { bg: "rgba(34,197,94,0.15)", text: "#22c55e" },
  odds_xg: { bg: "rgba(21,128,61,0.15)", text: "#15803d" },
  xg: { bg: "rgba(3,105,161,0.15)", text: "#0369a1" },
  odds: { bg: "rgba(180,83,9,0.15)", text: "#b45309" },
  base: { bg: "rgba(127,29,29,0.15)", text: "#7f1d1d" },
  insufficient_data: { bg: "rgba(107,114,128,0.15)", text: "#6b7280" },
};

function pctColor(v: number): string {
  if (v >= 85) return "#22c55e";
  if (v >= 50) return "#0369a1";
  if (v >= 25) return "#b45309";
  return "#7f1d1d";
}

// --- Sidebar ---

function MapSidebar({
  summary,
  leagues,
  timeWindow,
  onWindowChange,
  selectedCountry,
  onCountryClick,
}: {
  summary: CoverageMapData["summary"] | undefined;
  leagues: CoverageLeague[];
  timeWindow: CoverageWindow;
  onWindowChange: (w: CoverageWindow) => void;
  selectedCountry: string | null;
  onCountryClick: (iso3: string | null) => void;
}) {
  const sorted = useMemo(
    () =>
      [...leagues].sort((a, b) => b.coverage_total_pct - a.coverage_total_pct),
    [leagues]
  );

  return (
    <div className="w-[290px] min-w-[290px] shrink-0 bg-sidebar border-r border-border flex flex-col overflow-hidden">
      <div className="flex flex-col py-4 px-3 gap-5">
        {/* Time Window */}
        <div>
          <h3 className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground mb-2">
            Time Window
          </h3>
          <div className="flex flex-col gap-0.5">
            {WINDOW_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => onWindowChange(opt.value)}
                className={`text-left text-[13px] px-2 py-1 rounded transition-smooth ${
                  timeWindow === opt.value
                    ? "text-foreground bg-[rgba(71,151,255,0.12)]"
                    : "text-muted-foreground hover:text-foreground hover:bg-[rgba(249,250,250,0.04)]"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Summary */}
        {summary && (
          <div>
            <h3 className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground mb-2">
              Summary
            </h3>
            <div className="flex flex-col gap-1 text-[12px]">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Countries</span>
                <span className="text-foreground tabular-nums">
                  {summary.countries}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Leagues</span>
                <span className="text-foreground tabular-nums">
                  {summary.leagues}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Matches</span>
                <span className="text-foreground tabular-nums">
                  {summary.eligible_matches.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        )}

      </div>

      {/* Leagues list */}
      <div className="flex-1 overflow-auto border-t border-border">
        <h3 className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground px-3 pt-3 pb-1.5">
          All Leagues
        </h3>
        <div className="flex flex-col">
          {sorted.map((lg) => (
            <button
              key={lg.league_id}
              onClick={() =>
                onCountryClick(
                  selectedCountry === lg.country_iso3 ? null : lg.country_iso3
                )
              }
              className={`text-left px-3 py-1.5 text-[12px] flex justify-between items-center transition-smooth ${
                selectedCountry === lg.country_iso3
                  ? "bg-[rgba(71,151,255,0.08)] text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-[rgba(249,250,250,0.04)]"
              }`}
            >
              <span className="truncate mr-2">{lg.league_name}</span>
              <span
                className="tabular-nums flex-shrink-0 text-[11px]"
                style={{ color: pctColor(lg.coverage_total_pct) }}
              >
                {lg.coverage_total_pct}%
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- Country Detail Content (shared between desktop & mobile) ---

function CountryDetailContent({ leagues }: { leagues: CoverageLeague[] }) {
  return (
    <div className="flex flex-col gap-4">
      {leagues.map((lg) => {
        const tierStyle = TIER_COLORS[lg.universe_tier] || TIER_COLORS.base;
        return (
          <div
            key={lg.league_id}
            className="border border-border rounded-lg p-3"
          >
            {/* League header */}
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-foreground">
                {lg.league_name}
              </h4>
              <span
                className="px-2 py-0.5 rounded text-[11px] font-medium"
                style={{ background: tierStyle.bg, color: tierStyle.text }}
              >
                {TIER_LABELS[lg.universe_tier] || lg.universe_tier}
              </span>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[12px] mb-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Matches</span>
                <span className="text-foreground tabular-nums">
                  {lg.eligible_matches.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total</span>
                <span
                  className="tabular-nums font-semibold"
                  style={{ color: pctColor(lg.coverage_total_pct) }}
                >
                  {lg.coverage_total_pct}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">P0</span>
                <span
                  className="tabular-nums"
                  style={{ color: pctColor(lg.p0_pct) }}
                >
                  {lg.p0_pct}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">P1</span>
                <span
                  className="tabular-nums"
                  style={{ color: pctColor(lg.p1_pct) }}
                >
                  {lg.p1_pct}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">P2</span>
                <span
                  className="tabular-nums"
                  style={{ color: pctColor(lg.p2_pct) }}
                >
                  {lg.p2_pct}%
                </span>
              </div>
            </div>

            {/* Dimensions */}
            {lg.dimensions && (
              <div className="border-t border-border pt-2">
                <h5 className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground mb-1.5">
                  Dimensions
                </h5>
                <div className="grid grid-cols-1 gap-1 text-[11px]">
                  {Object.entries(lg.dimensions).map(([key, dim]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-muted-foreground truncate mr-2">
                        {key.replace(/_/g, " ")}
                      </span>
                      <span className="flex-shrink-0">
                        <span
                          className="tabular-nums"
                          style={{ color: pctColor(dim.pct) }}
                        >
                          {dim.pct}%
                        </span>
                        <span className="text-muted-foreground ml-1">
                          ({dim.numerator}/{dim.denominator})
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// --- Main Page ---

function CoverageMapContent() {
  const [timeWindow, setTimeWindow] = useState<CoverageWindow>("current_season");
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const { data, isLoading, error } = useCoverageMap(timeWindow);
  const isDesktop = useIsDesktop();

  const handleWindowChange = useCallback((w: CoverageWindow) => {
    setTimeWindow(w);
    setSelectedCountry(null);
  }, []);

  const selectedLeagues = useMemo(() => {
    if (!selectedCountry || !data?.leagues) return [];
    return data.leagues.filter((l) => l.country_iso3 === selectedCountry);
  }, [selectedCountry, data?.leagues]);

  const selectedCountryName = useMemo(() => {
    if (!selectedCountry || !data?.countries) return "";
    return (
      data.countries.find((c) => c.country_iso3 === selectedCountry)
        ?.country_name ?? selectedCountry
    );
  }, [selectedCountry, data?.countries]);

  const selectedFlagIso2 = selectedCountry ? ISO3_TO_ISO2[selectedCountry] : null;

  const drawerTitle = useMemo(() => {
    if (!selectedCountryName) return null;
    return (
      <span className="inline-flex items-center gap-2">
        {selectedFlagIso2 && (
          <Image
            src={`/flags/${selectedFlagIso2}.svg`}
            alt=""
            width={20}
            height={20}
            className="rounded-full object-cover"
          />
        )}
        {selectedCountryName}
      </span>
    );
  }, [selectedCountryName, selectedFlagIso2]);

  const drawerOpen = !!selectedCountry;
  const handleDrawerClose = useCallback(() => setSelectedCountry(null), []);

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="h-12 flex items-center px-6 border-b border-border">
        <div className="flex items-center gap-2">
          <Globe className="h-4 w-4 text-muted-foreground" />
          <h1 className="text-lg font-semibold text-foreground">
            Coverage Map
          </h1>
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader size="md" />
        </div>
      ) : error ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-[var(--error,#ef4444)]">
          <p className="text-sm">Failed to load coverage data</p>
          <p className="text-xs text-muted-foreground">
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
        </div>
      ) : (
        <div className="flex-1 flex min-h-0 relative">
          {/* Left sidebar */}
          <MapSidebar
            summary={data?.summary}
            leagues={data?.leagues ?? []}
            timeWindow={timeWindow}
            onWindowChange={handleWindowChange}
            selectedCountry={selectedCountry}
            onCountryClick={setSelectedCountry}
          />

          {/* Map — 100% remaining space */}
          <div className="flex-1 min-w-0 relative">
            <CoverageWorldMap
              countries={data?.countries ?? []}
              leagues={data?.leagues ?? []}
              onCountryClick={setSelectedCountry}
              selectedCountry={selectedCountry}
            />

            {/* Coverage legend overlay */}
            <div className="absolute bottom-4 left-4 bg-[var(--surface-elevated,#232326)] border border-border rounded-lg px-3 py-2.5 pointer-events-none">
              <h4 className="text-[11px] font-semibold text-foreground mb-2">
                Coverage
              </h4>
              <div className="flex items-center gap-0.5 mb-1.5">
                {[...LEGEND_BANDS].reverse().map((band) => (
                  <span
                    key={band.label}
                    className="h-2 flex-1 first:rounded-l last:rounded-r"
                    style={{ background: band.color, minWidth: 28 }}
                  />
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground">
                <span>0%</span>
                <span>100%</span>
              </div>
            </div>
          </div>

          {/* Desktop: DetailDrawer overlay (same as SOTA, Data Quality, etc.) */}
          {isDesktop && (
            <DetailDrawer
              open={drawerOpen}
              onClose={handleDrawerClose}
              title={drawerTitle}
            >
              <CountryDetailContent leagues={selectedLeagues} />
            </DetailDrawer>
          )}
        </div>
      )}

      {/* Mobile: Sheet overlay */}
      {!isDesktop && (
        <Sheet
          open={drawerOpen}
          onOpenChange={(isOpen) => !isOpen && handleDrawerClose()}
        >
          <SheetContent side="right" className="w-full sm:max-w-md p-0">
            <SheetHeader className="px-4 py-3 border-b border-border">
              <SheetTitle className="text-sm font-semibold truncate flex items-center justify-center gap-2">
                {selectedFlagIso2 && (
                  <Image
                    src={`/flags/${selectedFlagIso2}.svg`}
                    alt=""
                    width={20}
                    height={20}
                    className="rounded-full object-cover"
                  />
                )}
                {selectedCountryName}
              </SheetTitle>
            </SheetHeader>
            <ScrollArea className="h-[calc(100vh-60px)]">
              <div className="p-4">
                <CountryDetailContent leagues={selectedLeagues} />
              </div>
            </ScrollArea>
          </SheetContent>
        </Sheet>
      )}
    </div>
  );
}

export default function CoverageMapPage() {
  return <CoverageMapContent />;
}
