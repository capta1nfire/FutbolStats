"use client";

import type { CoverageLeague } from "@/lib/types/coverage-map";

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

/** Reusable league coverage detail card(s) — used by coverage map drawer & league detail sheet */
export function CoverageDetailContent({ leagues }: { leagues: CoverageLeague[] }) {
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
