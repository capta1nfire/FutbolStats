"use client";

import { SotaStatsCoverage } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";
import { DegradedAlert } from "./DegradedAlert";
import { Database } from "lucide-react";

interface SotaStatsCoverageCardProps {
  data: SotaStatsCoverage | null;
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
 * Note: These are display hints, not thresholds - backend determines status
 */
function getPctColor(pct: number | undefined): string {
  if (pct === undefined) return "text-muted-foreground";
  if (pct >= 100) return "text-[var(--status-success-text)]";
  if (pct >= 70) return "text-[var(--status-warning-text)]";
  return "text-[var(--status-error-text)]";
}

/**
 * SOTA Stats Coverage Card
 *
 * Shows root cause coverage:
 * - By season: total_matches_ft, with_stats_pct, marked_no_stats_pct, shots_present_pct
 * - By league: name, league_id, with_stats_pct
 */
export function SotaStatsCoverageCard({ data }: SotaStatsCoverageCardProps) {
  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Database className="h-4 w-4" />
          <span className="text-sm">SOTA Stats Coverage unavailable</span>
        </div>
      </div>
    );
  }

  const seasons = Object.entries(data.by_season ?? {}).sort((a, b) => a[0].localeCompare(b[0]));
  const leagues = data.by_league ?? [];

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-sm font-semibold text-foreground">SOTA Stats Coverage</h3>
          <span className="text-xs text-muted-foreground">(Root Cause)</span>
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
                <th className="text-right py-2 px-2">FT Matches</th>
                <th className="text-right py-2 px-2">With Stats</th>
                <th className="text-right py-2 px-2">No Stats</th>
                <th className="text-right py-2 pl-2">Shots</th>
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
                seasons.map(([season, stats]) => (
                  <tr key={season} className="border-b border-border/50">
                    <td className="py-2 pr-4 font-medium text-foreground">{season}</td>
                    <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">
                      {stats.total_matches_ft.toLocaleString()}
                    </td>
                    <td className={`py-2 px-2 text-right tabular-nums ${getPctColor(stats.with_stats_pct)}`}>
                      {formatPct(stats.with_stats_pct)}
                    </td>
                    <td className={`py-2 px-2 text-right tabular-nums ${stats.marked_no_stats_pct > 0 ? "text-[var(--status-warning-text)]" : "text-muted-foreground"}`}>
                      {formatPct(stats.marked_no_stats_pct)}
                    </td>
                    <td className={`py-2 pl-2 text-right tabular-nums ${getPctColor(stats.shots_present_pct)}`}>
                      {formatPct(stats.shots_present_pct)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* By League Table */}
      <div>
        <p className="text-xs font-medium text-muted-foreground mb-2">By League (Top 5)</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-muted-foreground border-b border-border">
                <th className="text-left py-2 pr-4">League</th>
                <th className="text-right py-2 px-2">ID</th>
                <th className="text-right py-2 pl-2">With Stats</th>
              </tr>
            </thead>
            <tbody>
              {leagues.length === 0 ? (
                <tr>
                  <td colSpan={3} className="py-2 text-center text-muted-foreground">
                    No league data
                  </td>
                </tr>
              ) : (
                leagues.map((league) => (
                  <tr key={league.league_id} className="border-b border-border/50">
                    <td className="py-2 pr-4 text-foreground truncate max-w-[150px]">
                      {league.name || `ID ${league.league_id}`}
                    </td>
                    <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">
                      {league.league_id}
                    </td>
                    <td className={`py-2 pl-2 text-right tabular-nums ${getPctColor(league.with_stats_pct)}`}>
                      {formatPct(league.with_stats_pct)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
