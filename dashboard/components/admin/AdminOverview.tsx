"use client";

import { useAdminOverview } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";

export function AdminOverview() {
  const { data, isLoading, error } = useAdminOverview();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader size="md" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        Failed to load overview
      </div>
    );
  }

  const c = data.counts;
  const cov = data.coverage_summary;
  const topLeagues = data.top_leagues_by_matches ?? [];

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-lg font-semibold">Admin Overview</h2>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard label="Total Leagues" value={c?.leagues_total} />
        <SummaryCard label="Active" value={c?.leagues_active} />
        <SummaryCard label="Teams" value={c?.teams_total} />
        <SummaryCard label="Matches" value={c?.matches_total} />
        <SummaryCard label="Predictions" value={c?.predictions_total} />
        <SummaryCard label="Season 25/26" value={c?.matches_25_26} />
      </div>

      {/* Coverage */}
      {cov && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-muted-foreground">Coverage</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg border border-border bg-card p-3">
              <p className="text-xs text-muted-foreground">With Stats</p>
              <p className="text-xl font-semibold mt-1">{cov.matches_with_stats_pct}%</p>
            </div>
            <div className="rounded-lg border border-border bg-card p-3">
              <p className="text-xs text-muted-foreground">TITAN Tier 1 (25/26)</p>
              <p className="text-xl font-semibold mt-1">{cov.titan_tier1_25_26_pct}%</p>
            </div>
          </div>
        </div>
      )}

      {/* Top leagues */}
      {topLeagues.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-muted-foreground">Top Leagues by Matches</h3>
          <div className="space-y-1">
            {topLeagues.map((l) => (
              <div key={l.league_id} className="flex items-center justify-between text-sm">
                <span>{l.name}</span>
                <span className="font-mono text-muted-foreground">{l.matches.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-2xl font-semibold mt-1">{value != null ? value.toLocaleString() : "—"}</p>
    </div>
  );
}
