"use client";

import { useState, useMemo, useCallback } from "react";
import type { CoverageLeague } from "@/lib/types/coverage-map";

// --- Tier config ---

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

const TIER_ORDER = [
  "xi_odds_xg",
  "odds_xg",
  "xg",
  "odds",
  "base",
  "insufficient_data",
];

function pctColor(v: number): string {
  if (v >= 85) return "#22c55e";
  if (v >= 50) return "#0369a1";
  if (v >= 25) return "#b45309";
  return "#7f1d1d";
}

// --- Columns ---

type SortCol =
  | "league_name"
  | "country"
  | "eligible_matches"
  | "p0_pct"
  | "p1_pct"
  | "p2_pct"
  | "coverage_total_pct"
  | "universe_tier";

const COLUMNS: { key: SortCol; label: string; align?: "right" }[] = [
  { key: "league_name", label: "League" },
  { key: "country", label: "Country" },
  { key: "eligible_matches", label: "Matches", align: "right" },
  { key: "p0_pct", label: "P0%", align: "right" },
  { key: "p1_pct", label: "P1%", align: "right" },
  { key: "p2_pct", label: "P2%", align: "right" },
  { key: "coverage_total_pct", label: "Total%", align: "right" },
  { key: "universe_tier", label: "Tier" },
];

// --- Component ---

interface CoverageLeagueTableProps {
  leagues: CoverageLeague[];
  filterISO3: string | null;
  onClearFilter: () => void;
}

export function CoverageLeagueTable({
  leagues,
  filterISO3,
  onClearFilter,
}: CoverageLeagueTableProps) {
  const [sortCol, setSortCol] = useState<SortCol>("coverage_total_pct");
  const [sortAsc, setSortAsc] = useState(false);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const handleSort = useCallback(
    (col: SortCol) => {
      if (sortCol === col) setSortAsc((prev) => !prev);
      else {
        setSortCol(col);
        setSortAsc(false);
      }
    },
    [sortCol]
  );

  const filtered = useMemo(() => {
    if (!filterISO3) return leagues;
    return leagues.filter((l) => l.country_iso3 === filterISO3);
  }, [leagues, filterISO3]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let va: string | number = a[sortCol] as string | number;
      let vb: string | number = b[sortCol] as string | number;
      if (sortCol === "universe_tier") {
        va = TIER_ORDER.indexOf(va as string);
        vb = TIER_ORDER.indexOf(vb as string);
      }
      if (typeof va === "string" && typeof vb === "string") {
        va = va.toLowerCase();
        vb = vb.toLowerCase();
      }
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    });
  }, [filtered, sortCol, sortAsc]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Section header */}
      <div className="flex items-center justify-between px-6 py-2 border-b border-border bg-[var(--surface,#1c1e21)]">
        <h2 className="text-sm font-semibold text-foreground">
          {filterISO3
            ? `${filtered[0]?.country || filterISO3} Leagues`
            : "All Leagues"}
        </h2>
        {filterISO3 && (
          <button
            onClick={onClearFilter}
            className="text-[11px] text-muted-foreground border border-border rounded px-2 py-0.5 hover:text-foreground hover:border-[var(--primary)]"
          >
            Show all
          </button>
        )}
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={`sticky top-0 bg-background px-3 py-2 text-[11px] uppercase tracking-wider font-medium text-muted-foreground border-b border-border cursor-pointer select-none whitespace-nowrap hover:text-foreground ${col.align === "right" ? "text-right" : "text-left"}`}
                >
                  {col.label}
                  {sortCol === col.key && (
                    <span className="ml-1 text-[9px]">
                      {sortAsc ? "▲" : "▼"}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((lg, idx) => {
              const tierStyle = TIER_COLORS[lg.universe_tier] || TIER_COLORS.base;
              const isExpanded = expandedIdx === idx;
              return (
                <>
                  <tr
                    key={lg.league_id}
                    onClick={() =>
                      setExpandedIdx(isExpanded ? null : idx)
                    }
                    className="cursor-pointer hover:bg-[rgba(71,151,255,0.04)] border-b border-border/30"
                  >
                    <td className="px-3 py-2 whitespace-nowrap">
                      <span
                        className="inline-block mr-1.5 text-[10px] transition-transform"
                        style={{
                          transform: isExpanded
                            ? "rotate(90deg)"
                            : "rotate(0deg)",
                        }}
                      >
                        ▶
                      </span>
                      {lg.league_name}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap">
                      {lg.country}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums whitespace-nowrap">
                      {lg.eligible_matches.toLocaleString()}
                    </td>
                    <td
                      className="px-3 py-2 text-right tabular-nums"
                      style={{ color: pctColor(lg.p0_pct) }}
                    >
                      {lg.p0_pct}
                    </td>
                    <td
                      className="px-3 py-2 text-right tabular-nums"
                      style={{ color: pctColor(lg.p1_pct) }}
                    >
                      {lg.p1_pct}
                    </td>
                    <td
                      className="px-3 py-2 text-right tabular-nums"
                      style={{ color: pctColor(lg.p2_pct) }}
                    >
                      {lg.p2_pct}
                    </td>
                    <td
                      className="px-3 py-2 text-right tabular-nums font-semibold"
                      style={{ color: pctColor(lg.coverage_total_pct) }}
                    >
                      {lg.coverage_total_pct}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className="inline-block px-2 py-0.5 rounded text-[11px] font-medium"
                        style={{
                          background: tierStyle.bg,
                          color: tierStyle.text,
                        }}
                      >
                        {TIER_LABELS[lg.universe_tier] || lg.universe_tier}
                      </span>
                    </td>
                  </tr>
                  {isExpanded && lg.dimensions && (
                    <tr key={`${lg.league_id}-dims`}>
                      <td
                        colSpan={8}
                        className="px-3 py-1 bg-[var(--surface,#1c1e21)]"
                      >
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-x-6 gap-y-1 py-2 pl-6 text-[12px] text-muted-foreground">
                          {Object.entries(lg.dimensions).map(([key, dim]) => (
                            <div key={key} className="flex justify-between">
                              <span>{key.replace(/_/g, " ")}</span>
                              <span
                                className="tabular-nums ml-2"
                                style={{ color: pctColor(dim.pct) }}
                              >
                                {dim.pct}%{" "}
                                <span className="text-[10px] text-muted-foreground">
                                  ({dim.numerator}/{dim.denominator})
                                </span>
                              </span>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
