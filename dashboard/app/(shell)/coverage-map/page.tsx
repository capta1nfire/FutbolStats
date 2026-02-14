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
  CoverageLeague,
} from "@/lib/types/coverage-map";
import { Shield, Trophy, Users, ExternalLink } from "lucide-react";
import { IconTabs } from "@/components/ui/icon-tabs";
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

type ListTab = "leagues" | "tournaments" | "national";

const LIST_TABS = [
  { id: "leagues", icon: <Shield />, label: "Leagues" },
  { id: "tournaments", icon: <Trophy />, label: "Tournaments" },
  { id: "national", icon: <Users />, label: "National Teams", disabled: true },
];

function leagueKind(l: CoverageLeague): string {
  if (l.kind) return l.kind;
  // Fallback when backend hasn't deployed kind yet:
  // international tournaments have no country_iso3
  return l.country_iso3 ? "league" : "international";
}

function filterByTab(leagues: CoverageLeague[], tab: ListTab): CoverageLeague[] {
  if (tab === "leagues") return leagues.filter((l) => leagueKind(l) === "league");
  if (tab === "tournaments") {
    const k = new Set(["cup", "international"]);
    return leagues.filter((l) => k.has(leagueKind(l)));
  }
  return [];
}

function MapSidebar({
  leagues,
  timeWindow,
  onWindowChange,
  selectedCountry,
  onCountryClick,
}: {
  leagues: CoverageLeague[];
  timeWindow: CoverageWindow;
  onWindowChange: (w: CoverageWindow) => void;
  selectedCountry: string | null;
  onCountryClick: (iso3: string | null) => void;
}) {
  const [activeTab, setActiveTab] = useState<ListTab>("leagues");

  const sorted = useMemo(
    () =>
      [...filterByTab(leagues, activeTab)].sort((a, b) => b.coverage_total_pct - a.coverage_total_pct),
    [leagues, activeTab]
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

      </div>

      {/* Tab bar + list */}
      <div className="flex-1 overflow-auto border-t border-border">
        <div className="px-3 pt-3 pb-1.5">
          <IconTabs
            tabs={LIST_TABS}
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as ListTab)}
            className="w-full"
          />
        </div>
        {activeTab === "national" ? (
          <div className="px-3 py-6 text-center text-[11px] text-muted-foreground">
            Coming soon
          </div>
        ) : (
          <div className="flex flex-col">
            {sorted.map((lg) => (
              <button
                key={lg.league_id}
                onClick={() => onCountryClick(lg.country_iso3)}
                className={`text-left px-3 py-1.5 text-[12px] flex justify-between items-center transition-smooth ${
                  selectedCountry === lg.country_iso3
                    ? "bg-[rgba(71,151,255,0.08)] text-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-[rgba(249,250,250,0.04)]"
                }`}
              >
                <span className="truncate mr-2 inline-flex items-center gap-1.5">
                  {lg.country_iso3 && ISO3_TO_ISO2[lg.country_iso3] ? (
                    <Image
                      src={`/flags/${ISO3_TO_ISO2[lg.country_iso3]}.svg`}
                      alt=""
                      width={14}
                      height={14}
                      className="rounded-full object-cover shrink-0"
                    />
                  ) : lg.logo_url ? (
                    <img
                      src={lg.logo_url}
                      alt=""
                      width={14}
                      height={14}
                      className="shrink-0"
                    />
                  ) : null}
                  {lg.league_name}
                </span>
                <span
                  className="tabular-nums flex-shrink-0 text-[11px]"
                  style={{ color: pctColor(lg.coverage_total_pct) }}
                >
                  {lg.coverage_total_pct}%
                </span>
              </button>
            ))}
          </div>
        )}
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

  const footballLink = useMemo(() => {
    if (!selectedLeagues.length) return null;
    const lg = selectedLeagues[0];
    return `/football?category=leagues_by_country&country=${encodeURIComponent(lg.country)}&league=${lg.league_id}`;
  }, [selectedLeagues]);

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
        {footballLink && (
          <a
            href={footballLink}
            className="text-muted-foreground hover:text-primary transition-colors"
            title="Open in Football"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </span>
    );
  }, [selectedCountryName, selectedFlagIso2, footballLink]);

  const drawerOpen = !!selectedCountry;
  const handleDrawerClose = useCallback(() => setSelectedCountry(null), []);

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background">
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

            {/* Coverage legend + summary overlay */}
            <div className="absolute bottom-4 left-4 bg-[var(--surface-elevated,#232326)] border border-border rounded-lg px-3 py-2.5 pointer-events-none">
              <div className="text-[11px] font-medium text-foreground mb-1.5">
                {WINDOW_OPTIONS.find((o) => o.value === timeWindow)?.label ?? timeWindow}
              </div>
              {data?.summary && (
                <div className="flex items-center gap-3 text-[11px] mb-2.5">
                  <span className="text-muted-foreground">
                    <span className="text-foreground tabular-nums">{data.summary.countries}</span> countries
                  </span>
                  <span className="text-muted-foreground">
                    <span className="text-foreground tabular-nums">{data.summary.leagues}</span> leagues
                  </span>
                  <span className="text-muted-foreground">
                    <span className="text-foreground tabular-nums">{data.summary.eligible_matches.toLocaleString()}</span> matches
                  </span>
                </div>
              )}
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
