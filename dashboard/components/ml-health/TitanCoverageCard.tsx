"use client";

import { TitanCoverage } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";
import { DegradedAlert } from "./DegradedAlert";
import { Layers } from "lucide-react";

interface TitanCoverageCardProps {
  data: TitanCoverage | null;
}

/**
 * Format percentage for display
 */
function formatPct(pct: number | undefined): string {
  if (pct === undefined) return "—";
  return `${pct.toFixed(1)}%`;
}

/**
 * Get color class based on percentage
 */
function getPctColor(pct: number | undefined): string {
  if (pct === undefined) return "text-muted-foreground";
  if (pct >= 100) return "text-[var(--status-success-text)]";
  if (pct >= 70) return "text-[var(--status-warning-text)]";
  return "text-[var(--status-error-text)]";
}

/**
 * Check if league name is unmapped (backend returns "League <id>" pattern)
 */
function isUnmappedLeague(name: string | undefined, leagueId: number): boolean {
  if (!name) return true;
  return name === `League ${leagueId}`;
}

/**
 * TITAN Coverage Card
 *
 * Shows feature_matrix materialization:
 * - By season: tier1..tier1d with complete/total/pct
 * - By league: name, league_id, tier1_pct, tier1b_pct
 */
export function TitanCoverageCard({ data }: TitanCoverageCardProps) {
  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Layers className="h-4 w-4" />
          <span className="text-sm">TITAN Coverage unavailable</span>
        </div>
      </div>
    );
  }

  const seasons = Object.entries(data.by_season ?? {}).sort((a, b) => a[0].localeCompare(b[0]));
  const leagues = data.by_league ?? [];
  const tierKeys = ["tier1", "tier1b", "tier1c", "tier1d"] as const;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-purple-400" />
          <h3 className="text-sm font-semibold text-foreground">TITAN Coverage</h3>
          <span className="text-xs text-muted-foreground">(Feature Matrix)</span>
        </div>
        <StatusBadge status={data.status} />
      </div>

      {/* Degraded alert */}
      {data._degraded && (
        <div className="mb-4">
          <DegradedAlert error={data._error} />
        </div>
      )}

      {/* By Season Table */}
      <div className="mb-4">
        <p className="text-xs font-medium text-muted-foreground mb-2">By Season</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-muted-foreground border-b border-border">
                <th className="text-left py-2 pr-4">Season</th>
                {tierKeys.map((tier) => (
                  <th key={tier} className="text-right py-2 px-2 uppercase">
                    {tier.replace("tier", "T")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {seasons.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-2 text-center text-muted-foreground">
                    No season data
                  </td>
                </tr>
              ) : (
                seasons.map(([season, tiers]) => (
                  <tr key={season} className="border-b border-border/50">
                    <td className="py-2 pr-4 font-medium text-foreground">{season}</td>
                    {tierKeys.map((tier) => {
                      const tierData = tiers[tier];
                      if (!tierData) {
                        return (
                          <td key={tier} className="py-2 px-2 text-right text-muted-foreground">
                            —
                          </td>
                        );
                      }
                      return (
                        <td key={tier} className="py-2 px-2 text-right">
                          <div className={`tabular-nums ${getPctColor(tierData.pct)}`}>
                            {formatPct(tierData.pct)}
                          </div>
                          <div className="text-[10px] text-muted-foreground">
                            {tierData.complete}/{tierData.total}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* By League Table */}
      <div>
        <p className="text-xs font-medium text-muted-foreground mb-2">By League</p>
        <div className="overflow-x-auto max-h-[200px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card">
              <tr className="text-xs text-muted-foreground border-b border-border">
                <th className="text-left py-2 pr-4">League</th>
                <th className="text-right py-2 px-2">ID</th>
                <th className="text-right py-2 px-2">T1</th>
                <th className="text-right py-2 pl-2">T1b</th>
              </tr>
            </thead>
            <tbody>
              {leagues.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-2 text-center text-muted-foreground">
                    No league data
                  </td>
                </tr>
              ) : (
                leagues.map((league) => {
                  const unmapped = isUnmappedLeague(league.name, league.league_id);
                  return (
                  <tr key={league.league_id} className="border-b border-border/50">
                    <td className="py-2 pr-4 text-foreground truncate max-w-[120px]">
                      <span className="flex items-center gap-1.5">
                        {league.name || `ID ${league.league_id}`}
                        {unmapped && (
                          <span className="text-[10px] px-1 py-0.5 bg-muted text-muted-foreground rounded">
                            unmapped
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">
                      {league.league_id}
                    </td>
                    <td className={`py-2 px-2 text-right tabular-nums ${getPctColor(league.tier1_pct)}`}>
                      {formatPct(league.tier1_pct)}
                    </td>
                    <td className={`py-2 pl-2 text-right tabular-nums ${getPctColor(league.tier1b_pct)}`}>
                      {formatPct(league.tier1b_pct)}
                    </td>
                  </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
