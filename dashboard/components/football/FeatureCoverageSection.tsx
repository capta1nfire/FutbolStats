"use client";

import { Database, ShieldCheck, ShieldAlert, ShieldX } from "lucide-react";
import type { FeatureCoverage, FeatureCoverageSource } from "@/lib/types";

interface FeatureCoverageSectionProps {
  coverage: FeatureCoverage;
}

/** Color class by percentage threshold */
function pctColor(pct: number): string {
  if (pct >= 90) return "text-[var(--status-success-text)]";
  if (pct >= 60) return "text-[var(--status-warning-text)]";
  return "text-[var(--status-error-text)]";
}

function barColor(pct: number): string {
  if (pct >= 90) return "bg-[var(--status-success-text)]";
  if (pct >= 60) return "bg-[var(--status-warning-text)]";
  return "bg-[var(--status-error-text)]";
}

/** Kill-switch status badge */
function KillswitchBadge({
  status,
  count,
  min,
  lookbackDays,
}: {
  status: "ok" | "warning" | "blocked";
  count: number;
  min: number;
  lookbackDays: number;
}) {
  const config = {
    ok: {
      icon: <ShieldCheck className="h-3.5 w-3.5" />,
      label: "Eligible",
      bg: "bg-[var(--status-success-text)]/10",
      text: "text-[var(--status-success-text)]",
    },
    warning: {
      icon: <ShieldAlert className="h-3.5 w-3.5" />,
      label: "Low Data",
      bg: "bg-[var(--status-warning-text)]/10",
      text: "text-[var(--status-warning-text)]",
    },
    blocked: {
      icon: <ShieldX className="h-3.5 w-3.5" />,
      label: "Blocked",
      bg: "bg-[var(--status-error-text)]/10",
      text: "text-[var(--status-error-text)]",
    },
  }[status];

  return (
    <div className={`flex items-center gap-2 rounded-md px-2.5 py-1.5 ${config.bg}`}>
      <span className={config.text}>{config.icon}</span>
      <span className={`text-xs font-medium ${config.text}`}>{config.label}</span>
      <span className="text-xs text-muted-foreground ml-auto">
        {count} FT league matches / {lookbackDays}d (min: {min})
      </span>
    </div>
  );
}

/** Compact coverage bar row */
function CoverageRow({ label, source }: { label: string; source: FeatureCoverageSource }) {
  const clampedPct = Math.max(0, Math.min(100, source.pct));

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground w-14 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor(clampedPct)}`}
          style={{ width: `${clampedPct}%` }}
        />
      </div>
      <span className={`text-xs font-medium w-10 text-right ${pctColor(clampedPct)}`}>
        {clampedPct}%
      </span>
    </div>
  );
}

/** Tier chip badge */
function TierChip({ label, source }: { label: string; source: FeatureCoverageSource }) {
  const clampedPct = Math.max(0, Math.min(100, source.pct));

  return (
    <div
      className={`flex items-center gap-1 rounded px-2 py-0.5 bg-muted/50 ${pctColor(clampedPct)}`}
    >
      <span className="text-[10px] font-medium">{label}</span>
      <span className="text-[10px] font-bold">{clampedPct}%</span>
    </div>
  );
}

const TIER_LABELS: Record<string, string> = {
  tier1: "T1",
  tier1b: "T1b",
  tier1c: "T1c",
  tier1d: "T1d",
  tier2: "T2",
  tier3: "T3",
};

/**
 * Feature Coverage Section for Team Detail (ATI P0).
 *
 * 3 layers:
 * 1. Kill-switch status (eligible/warning/blocked)
 * 2. Source coverage bars (odds, xG, lineup, XI, form, H2H)
 * 3. TITAN tier chips (T1/T1b/T1c/T1d/T2/T3)
 */
export function FeatureCoverageSection({ coverage }: FeatureCoverageSectionProps) {
  const { killswitch, sources, tiers } = coverage;
  const lookbackDays = killswitch?.lookback_days ?? 90;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
          <Database className="h-4 w-4 text-muted-foreground" />
          Feature Coverage
        </h4>
        <span className="text-[10px] text-muted-foreground">
          Last {lookbackDays}d
        </span>
      </div>

      {/* Layer 1: Kill-switch */}
      {killswitch && (
        <KillswitchBadge
          status={killswitch.status}
          count={killswitch.ft_league_matches}
          min={killswitch.min_required}
          lookbackDays={killswitch.lookback_days}
        />
      )}

      {/* Layer 2: Source coverage */}
      {sources && (
        <div className="space-y-1.5">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
            Sources ({sources.total_matches} league matches in TITAN)
          </p>
          <CoverageRow label="Odds" source={sources.odds} />
          <CoverageRow label="xG" source={sources.xg} />
          <CoverageRow label="Lineup" source={sources.lineup} />
          <CoverageRow label="XI" source={sources.xi_depth} />
          <CoverageRow label="Form" source={sources.form} />
          <CoverageRow label="H2H" source={sources.h2h} />
        </div>
      )}

      {/* Layer 3: Tier chips */}
      {tiers && (
        <div className="space-y-1.5">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
            TITAN Tiers
          </p>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(tiers).map(([key, source]) => (
              <TierChip key={key} label={TIER_LABELS[key] ?? key} source={source} />
            ))}
          </div>
        </div>
      )}

      {/* No data state */}
      {!killswitch && !sources && !tiers && (
        <p className="text-xs text-muted-foreground">No coverage data available</p>
      )}
    </div>
  );
}
