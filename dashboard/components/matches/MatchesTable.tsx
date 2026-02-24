"use client";

import { useMemo, useCallback } from "react";
import { MatchSummary, MatchScore, ProbabilitySet, MatchStatus, ValueBet } from "@/lib/types";
import { StatusDot } from "./StatusDot";
import { TeamLogo } from "@/components/ui/team-logo";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { Copy, Crosshair, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import { useRegion } from "@/components/providers/RegionProvider";
import { getPredictionPick, getProbabilityCellClasses, computeGap20, type Outcome } from "@/lib/predictions";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// =============================================================================
// Model Accuracy Calculation
// =============================================================================

type ModelKey = "market" | "consensus" | "pinnacle" | "modelA" | "shadow" | "sensorB" | "familyS" | "extA" | "extB" | "extC" | "extD";

interface ModelAccuracyStats {
  correct: number;
  total: number;
  accuracy: number | null;
}

function calculateModelAccuracies(
  data: MatchSummary[]
): Record<ModelKey, ModelAccuracyStats> {
  const stats: Record<ModelKey, ModelAccuracyStats> = {
    market: { correct: 0, total: 0, accuracy: null },
    consensus: { correct: 0, total: 0, accuracy: null },
    pinnacle: { correct: 0, total: 0, accuracy: null },
    modelA: { correct: 0, total: 0, accuracy: null },
    shadow: { correct: 0, total: 0, accuracy: null },
    sensorB: { correct: 0, total: 0, accuracy: null },
    familyS: { correct: 0, total: 0, accuracy: null },
    extA: { correct: 0, total: 0, accuracy: null },
    extB: { correct: 0, total: 0, accuracy: null },
    extC: { correct: 0, total: 0, accuracy: null },
    extD: { correct: 0, total: 0, accuracy: null },
  };

  for (const match of data) {
    if (match.status !== "ft") continue;

    const outcome = getOutcomeFromScore(match.score);
    if (!outcome) continue;

    const models: { key: ModelKey; probs: ProbabilitySet | undefined }[] = [
      { key: "market", probs: match.market },
      { key: "consensus", probs: match.consensus },
      { key: "pinnacle", probs: match.pinnacle },
      { key: "modelA", probs: match.modelA },
      { key: "shadow", probs: match.shadow },
      { key: "sensorB", probs: match.sensorB },
      { key: "familyS", probs: match.familyS },
      { key: "extA", probs: match.extA },
      { key: "extB", probs: match.extB },
      { key: "extC", probs: match.extC },
      { key: "extD", probs: match.extD },
    ];

    for (const { key, probs } of models) {
      if (!probs) continue;

      stats[key].total++;
      const pickResult = getPredictionPick(
        { home: probs.home, draw: probs.draw, away: probs.away },
        outcome
      );
      if (pickResult.isCorrect) {
        stats[key].correct++;
      }
    }
  }

  for (const key of Object.keys(stats) as ModelKey[]) {
    if (stats[key].total > 0) {
      stats[key].accuracy = (stats[key].correct / stats[key].total) * 100;
    }
  }

  return stats;
}

function getOutcomeFromScore(score?: MatchScore): Outcome | null {
  if (!score) return null;
  if (score.home > score.away) return "home";
  if (score.home < score.away) return "away";
  return "draw";
}

// =============================================================================
// Helper Components
// =============================================================================

/** Static model descriptions for header tooltips */
const MODEL_TOOLTIPS: Record<string, string> = {
  "Market": "Frozen closing odds (de-vigged implied probabilities)",
  "Consensus": "Median of de-vigged bookmaker odds (fair market consensus)",
  "Pinnacle": "Pinnacle sharp line (implied probabilities)",
  "Sensor B": "Sensor B — calibration diagnostic",
  "Ext A": "Experimental slot A",
  "Ext B": "Experimental slot B",
  "Ext C": "Experimental slot C",
  "Ext D": "Experimental slot D",
};

/** Autopsy tag visual config — Financial Autopsy (6 mutually exclusive post-match tags) */
const AUTOPSY_CONFIG: Record<string, { label: string; color: string; descEs: string }> = {
  sharp_win:     { label: "SHARP",    color: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30", descEs: "Acierto con ventaja. CLV positivo: el modelo acertó y tenía edge de timing." },
  routine_win:   { label: "ROUTINE",  color: "bg-sky-500/15 text-sky-400 border-sky-500/30",            descEs: "Acierto estándar. xG alineado con el resultado, sin edge especial." },
  lucky_win:     { label: "LUCKY",    color: "bg-amber-500/15 text-amber-400 border-amber-500/30",      descEs: "Acierto con suerte. xG decía otra cosa, el resultado nos favoreció." },
  sharp_loss:    { label: "SHARP L",  color: "bg-purple-500/15 text-purple-400 border-purple-500/30",   descEs: "Fallo con edge. CLV positivo pero la varianza nos jugó en contra." },
  variance_loss: { label: "VARIANCE", color: "bg-slate-500/15 text-slate-400 border-slate-500/30",      descEs: "Fallo justo. xG nos respaldaba, fue varianza normal." },
  blind_spot:    { label: "BLIND",    color: "bg-red-500/15 text-red-400 border-red-500/30",            descEs: "Punto ciego. CLV negativo, xG en contra — fallo sistemático." },
};

/** Extract dynamic model versions from the first match that has data for each model */
function getDynamicTooltips(data: MatchSummary[]): Record<string, string> {
  const tooltips: Record<string, string> = {};
  for (const m of data) {
    if (!tooltips["Model A"] && m.modelAVersion) {
      tooltips["Model A"] = m.modelAVersion;
    }
    if (!tooltips["Shadow"] && m.shadowVersion) {
      tooltips["Shadow"] = m.shadowVersion;
    }
    if (!tooltips["Family S"] && m.familySVersion) {
      tooltips["Family S"] = m.familySVersion;
    }
    if (tooltips["Model A"] && tooltips["Shadow"] && tooltips["Family S"]) break;
  }
  return tooltips;
}

function ModelHeader({ label, stats, dynamicTooltip }: { label: string; stats: ModelAccuracyStats; dynamicTooltip?: string }) {
  const hasStats = stats.accuracy !== null && stats.total > 0;
  const tooltipText = dynamicTooltip || MODEL_TOOLTIPS[label];

  return (
    <div className="flex items-center justify-center gap-1.5 leading-tight whitespace-nowrap">
      {tooltipText ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="cursor-help border-b border-dotted border-muted-foreground/50">{label}</span>
          </TooltipTrigger>
          <TooltipContent side="bottom" sideOffset={4}>
            <p className="text-xs">{tooltipText}</p>
          </TooltipContent>
        </Tooltip>
      ) : (
        <span>{label}</span>
      )}
      {hasStats && <span>({stats.correct})</span>}
      {hasStats && (
        <span
          className="text-[10px] font-medium px-1.5 py-0.5 rounded tabular-nums bg-muted text-muted-foreground"
          title={`${stats.correct}/${stats.total} correct`}
        >
          {stats.accuracy!.toFixed(1)}%
        </span>
      )}
    </div>
  );
}

function getStatusColor(status: MatchStatus): { text: string; hover: string } {
  switch (status) {
    case "live":
      return { text: "text-success", hover: "hover:text-success/80" };
    case "ht":
      return { text: "text-warning", hover: "hover:text-warning/80" };
    case "postponed":
      return { text: "text-warning", hover: "hover:text-warning/80" };
    case "cancelled":
      return { text: "text-error", hover: "hover:text-error/80" };
    case "ft":
      return { text: "text-foreground", hover: "hover:text-foreground/80" };
    case "scheduled":
    default:
      return { text: "text-primary", hover: "hover:text-primary-hover" };
  }
}

function CopyableId({ id, status }: { id: number; status: MatchStatus }) {
  const colors = getStatusColor(status);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      navigator.clipboard.writeText(String(id));
      toast(
        <div className="flex items-center gap-2">
          <Copy className={cn("h-4 w-4", colors.text)} />
          <span>
            Match ID <span className={cn("font-medium", colors.text)}>{id}</span> copied
          </span>
        </div>
      );
    },
    [id, colors.text]
  );

  return (
    <button
      onClick={handleClick}
      className={cn("font-mono text-xs cursor-pointer transition-colors", colors.text, colors.hover)}
    >
      {id}
    </button>
  );
}

interface ProbabilityCellProps {
  probs?: ProbabilitySet;
  outcome?: Outcome | null;
  compact?: boolean;
}

function ProbabilityCell({ probs, outcome, compact = false }: ProbabilityCellProps) {
  if (!probs) {
    return <span className="text-muted-foreground text-xs">-</span>;
  }

  const pickResult = getPredictionPick(
    { home: probs.home, draw: probs.draw, away: probs.away },
    outcome ?? null
  );

  if (compact) {
    const outcomes: { key: Outcome; label: string; prob: number }[] = [
      { key: "home", label: "1", prob: probs.home },
      { key: "draw", label: "X", prob: probs.draw },
      { key: "away", label: "2", prob: probs.away },
    ];

    const picks = outcomes.filter((o) => pickResult.topOutcomes.includes(o.key));

    return (
      <div className="text-xs font-mono leading-tight">
        {picks.map((pick) => (
          <div key={pick.key} className={getProbabilityCellClasses(pick.key, pickResult)}>
            {pick.label}: {(pick.prob * 100).toFixed(0)}%
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="text-xs font-mono leading-tight">
      <div className={getProbabilityCellClasses("home", pickResult)}>
        1: {(probs.home * 100).toFixed(0)}%
      </div>
      <div className={getProbabilityCellClasses("draw", pickResult)}>
        X: {(probs.draw * 100).toFixed(0)}%
      </div>
      <div className={getProbabilityCellClasses("away", pickResult)}>
        2: {(probs.away * 100).toFixed(0)}%
      </div>
    </div>
  );
}

// =============================================================================
// Kelly Bet Badge (Trading Core — GDT Glass House)
// =============================================================================

const FLAG_LABELS: Record<string, string> = {
  "MIN_EV_REJECTED": "EV insuficiente",
  "HIGH_ODDS_PENALTY": "Penalización cuota alta",
  "MAX_MATCH_CAP_APPLIED": "Cap de partido",
};

function KellyBet({ vb }: { vb: ValueBet }) {
  const label = vb.outcome === "home" ? "1" : vb.outcome === "draw" ? "X" : "2";
  const evPct = (vb.expectedValue * 100).toFixed(1);
  const hasFlags = vb.stakeFlags && vb.stakeFlags.length > 0;

  const colorClass = vb.suggestedStake === 0
    ? "bg-slate-500/10 text-slate-400 border-slate-500/20"
    : vb.expectedValue >= 0.20
    ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
    : "bg-amber-500/15 text-amber-400 border-amber-500/30";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className={cn(
          "inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-semibold rounded border",
          colorClass
        )}>
          {hasFlags && <ShieldAlert className="h-2.5 w-2.5" />}
          {label} {evPct}% · {vb.stakeUnits}U
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8}>
        <div className="text-xs space-y-0.5">
          <p>EV: {evPct}% | Odds: {vb.marketOdds}</p>
          <p>Kelly bruto: {(vb.kellyRaw * 100).toFixed(2)}%</p>
          <p>Kelly fracción: {(vb.kellyFraction * 100).toFixed(2)}%</p>
          <p className="font-semibold">Stake: {vb.stakeUnits} unidades</p>
          {hasFlags && (
            <p className="text-amber-400">
              {vb.stakeFlags!.map(f => FLAG_LABELS[f] || f).join(" · ")}
            </p>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

// =============================================================================
// VORP Shock Badge (Sprint 3 — Glass House)
// =============================================================================

function VorpBadge({ match }: { match: MatchSummary }) {
  if (!match.vorpApplied) return null;
  const diff = match.talentDeltaDiff;
  const label = diff && diff > 0 ? "HOME+" : diff && diff < 0 ? "AWAY+" : "VORP";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex items-center gap-0.5 rounded-full bg-violet-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-violet-400">
          <Crosshair className="h-3 w-3" />
          {label}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8}>
        <div className="text-xs space-y-0.5">
          <p>VORP Lineup Shock aplicado</p>
          <p>talent_delta_diff: {diff?.toFixed(4) ?? "N/A"}</p>
          <p className="text-violet-300">Probabilidades ajustadas a T-60</p>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

// =============================================================================
// Column Options (for external use)
// =============================================================================

import { ColumnOption } from "@/components/tables";
import { VisibilityState } from "@tanstack/react-table";

export const MATCHES_COLUMN_OPTIONS: ColumnOption[] = [
  { id: "rowIndex", label: "#", enableHiding: false },
  { id: "match", label: "Match", enableHiding: false },
  { id: "leagueName", label: "League", enableHiding: true },
  { id: "kickoffISO", label: "Kickoff", enableHiding: true },
  { id: "score", label: "Score", enableHiding: true },
  { id: "kelly", label: "Kelly", enableHiding: true },
  { id: "elapsed", label: "Elapsed", enableHiding: true },
  { id: "market", label: "Market", enableHiding: true },
  { id: "consensus", label: "Consensus", enableHiding: true },
  { id: "pinnacle", label: "Pinnacle", enableHiding: true },
  { id: "modelA", label: "Model A", enableHiding: true },
  { id: "shadow", label: "Shadow", enableHiding: true },
  { id: "sensorB", label: "Sensor B", enableHiding: true },
  { id: "familyS", label: "Family S", enableHiding: true },
  { id: "extA", label: "Ext A", enableHiding: true },
  { id: "extB", label: "Ext B", enableHiding: true },
  { id: "extC", label: "Ext C", enableHiding: true },
  { id: "extD", label: "Ext D", enableHiding: true },
];

export const MATCHES_DEFAULT_VISIBILITY: VisibilityState = {};

// =============================================================================
// Main Component
// =============================================================================

// Column widths - fixed for horizontal scroll
const ROW_NUM_WIDTH = 70; // Wider to fit #, ID, and status
const MATCH_COL_WIDTH = 220;
const LEAGUE_COL_WIDTH = 180;
const KICKOFF_COL_WIDTH = 100;
const SCORE_COL_WIDTH = 80;
const ELAPSED_COL_WIDTH = 80;
const MODEL_COL_WIDTH = 140;
const KELLY_COL_WIDTH = 130;

interface MatchesTableProps {
  data: MatchSummary[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedMatchId?: number | null;
  onRowClick?: (match: MatchSummary) => void;
  columnVisibility?: VisibilityState;
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
  getLogoUrl?: (teamName: string) => string | null;
  compactPredictions?: boolean;
}

export function MatchesTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedMatchId,
  onRowClick,
  columnVisibility = {},
  getLogoUrl,
  compactPredictions = false,
}: MatchesTableProps) {
  const { formatShortDate, formatTime } = useRegion();
  const modelAccuracies = useMemo(() => calculateModelAccuracies(data), [data]);
  const dynamicTooltips = useMemo(() => getDynamicTooltips(data), [data]);

  // Sort: Live/HT first, then Scheduled, then FT/finished
  const sortedData = useMemo(() => {
    const statusOrder: Record<string, number> = {
      live: 0, ht: 0,
      scheduled: 1,
      ft: 2, postponed: 2, cancelled: 2,
    };
    return [...data].sort((a, b) => {
      const oa = statusOrder[a.status] ?? 1;
      const ob = statusOrder[b.status] ?? 1;
      if (oa !== ob) return oa - ob;
      // Within live group: sort by elapsed descending (closest to FT first)
      if (oa === 0) {
        const ea = (a.elapsed?.min ?? 0) + (a.elapsed?.extra ?? 0);
        const eb = (b.elapsed?.min ?? 0) + (b.elapsed?.extra ?? 0);
        return eb - ea;
      }
      return 0;
    });
  }, [data]);

  // Column visibility helpers
  const isVisible = (colId: string) => columnVisibility[colId] !== false;

  // Min width based on visible columns (prevents "squished" table that never overflows)
  // Mirrors the SOTA FeatureCoverageMatrix behavior: allow natural width + overflow-x when needed.
  const tableMinWidth = useMemo(() => {
    let w = ROW_NUM_WIDTH + MATCH_COL_WIDTH;
    if (isVisible("leagueName")) w += LEAGUE_COL_WIDTH;
    if (isVisible("kickoffISO")) w += KICKOFF_COL_WIDTH;
    if (isVisible("score")) w += SCORE_COL_WIDTH;
    if (isVisible("kelly")) w += KELLY_COL_WIDTH;
    if (isVisible("elapsed")) w += ELAPSED_COL_WIDTH;
    if (isVisible("market")) w += MODEL_COL_WIDTH;
    if (isVisible("consensus")) w += MODEL_COL_WIDTH;
    if (isVisible("pinnacle")) w += MODEL_COL_WIDTH;
    if (isVisible("modelA")) w += MODEL_COL_WIDTH;
    if (isVisible("shadow")) w += MODEL_COL_WIDTH;
    if (isVisible("sensorB")) w += MODEL_COL_WIDTH;
    if (isVisible("familyS")) w += MODEL_COL_WIDTH;
    if (isVisible("extA")) w += MODEL_COL_WIDTH;
    if (isVisible("extB")) w += MODEL_COL_WIDTH;
    if (isVisible("extC")) w += MODEL_COL_WIDTH;
    if (isVisible("extD")) w += MODEL_COL_WIDTH;
    return w;
  }, [columnVisibility]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader size="md" />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-center">
          <p className="text-sm text-error">Failed to load matches</p>
          {onRetry && (
            <Button variant="secondary" size="sm" onClick={onRetry}>
              Retry
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Empty state
  if (data.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">No matches found</p>
      </div>
    );
  }

  return (
    <TooltipProvider delayDuration={150}>
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Single scroll container (like SOTA matrix) so header + body scroll together */}
        <div className="flex-1 overflow-auto">
          <table
            className="border-collapse text-sm"
            // Force the table to keep its intended width (prevents shrink-to-fit that kills overflow-x)
            style={{ width: tableMinWidth, minWidth: tableMinWidth }}
          >
            {/* Fixed column widths to keep header/body perfectly aligned */}
            <colgroup>
              <col style={{ width: ROW_NUM_WIDTH }} />
              <col style={{ width: MATCH_COL_WIDTH }} />
              {isVisible("leagueName") && <col style={{ width: LEAGUE_COL_WIDTH }} />}
              {isVisible("kickoffISO") && <col style={{ width: KICKOFF_COL_WIDTH }} />}
              {isVisible("score") && <col style={{ width: SCORE_COL_WIDTH }} />}
              {isVisible("kelly") && <col style={{ width: KELLY_COL_WIDTH }} />}
              {isVisible("elapsed") && <col style={{ width: ELAPSED_COL_WIDTH }} />}
              {isVisible("market") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("consensus") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("pinnacle") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("modelA") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("shadow") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("sensorB") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("familyS") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("extA") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("extB") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("extC") && <col style={{ width: MODEL_COL_WIDTH }} />}
              {isVisible("extD") && <col style={{ width: MODEL_COL_WIDTH }} />}
            </colgroup>

            {/* Sticky header */}
            <thead className="sticky top-0 z-20 bg-background">
              <tr className="border-b border-border">
                {/* # column - sticky left (includes row number, match ID, and status) */}
                <th
                  className="sticky left-0 z-30 px-2 py-3 text-center font-semibold text-muted-foreground text-xs box-border"
                  style={{
                    width: ROW_NUM_WIDTH,
                    minWidth: ROW_NUM_WIDTH,
                    maxWidth: ROW_NUM_WIDTH,
                    backgroundColor: "var(--background)",
                  }}
                >
                  #
                </th>
                {/* Match column - sticky left */}
                <th
                  className="sticky z-30 px-3 py-3 text-left font-semibold text-muted-foreground text-sm border-r border-border relative after:absolute after:top-0 after:right-0 after:bottom-0 after:w-4 after:translate-x-full after:bg-gradient-to-r after:from-black/20 after:to-transparent after:pointer-events-none box-border"
                  style={{
                    left: ROW_NUM_WIDTH,
                    width: MATCH_COL_WIDTH,
                    minWidth: MATCH_COL_WIDTH,
                    maxWidth: MATCH_COL_WIDTH,
                    backgroundColor: "var(--background)",
                  }}
                >
                  Match
                </th>
                {/* Scrollable columns - each with fixed minWidth for horizontal scroll */}
                {isVisible("leagueName") && (
                  <th
                    className="px-3 py-3 text-left font-semibold text-muted-foreground text-sm whitespace-nowrap"
                    style={{ minWidth: LEAGUE_COL_WIDTH }}
                  >
                    League
                  </th>
                )}
                {isVisible("kickoffISO") && (
                  <th
                    className="px-3 py-3 text-left font-semibold text-muted-foreground text-sm whitespace-nowrap"
                    style={{ minWidth: KICKOFF_COL_WIDTH }}
                  >
                    Kickoff
                  </th>
                )}
                {isVisible("score") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-muted-foreground text-sm whitespace-nowrap"
                    style={{ minWidth: SCORE_COL_WIDTH }}
                  >
                    Score
                  </th>
                )}
                {isVisible("kelly") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-muted-foreground text-sm whitespace-nowrap"
                    style={{ minWidth: KELLY_COL_WIDTH }}
                  >
                    Kelly
                  </th>
                )}
                {isVisible("elapsed") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-muted-foreground text-sm whitespace-nowrap"
                    style={{ minWidth: ELAPSED_COL_WIDTH }}
                  >
                    Elapsed
                  </th>
                )}
                {isVisible("market") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Market" stats={modelAccuracies.market} />
                  </th>
                )}
                {isVisible("consensus") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Consensus" stats={modelAccuracies.consensus} />
                  </th>
                )}
                {isVisible("pinnacle") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Pinnacle" stats={modelAccuracies.pinnacle} />
                  </th>
                )}
                {isVisible("modelA") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Model A" stats={modelAccuracies.modelA} dynamicTooltip={dynamicTooltips["Model A"]} />
                  </th>
                )}
                {isVisible("shadow") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Shadow" stats={modelAccuracies.shadow} dynamicTooltip={dynamicTooltips["Shadow"]} />
                  </th>
                )}
                {isVisible("sensorB") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Sensor B" stats={modelAccuracies.sensorB} />
                  </th>
                )}
                {isVisible("familyS") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Family S" stats={modelAccuracies.familyS} dynamicTooltip={dynamicTooltips["Family S"]} />
                  </th>
                )}
                {isVisible("extA") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Ext A" stats={modelAccuracies.extA} />
                  </th>
                )}
                {isVisible("extB") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Ext B" stats={modelAccuracies.extB} />
                  </th>
                )}
                {isVisible("extC") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Ext C" stats={modelAccuracies.extC} />
                  </th>
                )}
                {isVisible("extD") && (
                  <th
                    className="px-3 py-3 text-center font-semibold text-sm whitespace-nowrap"
                    style={{ minWidth: MODEL_COL_WIDTH }}
                  >
                    <ModelHeader label="Ext D" stats={modelAccuracies.extD} />
                  </th>
                )}
              </tr>
            </thead>

            {/* Body */}
            <tbody>
              {sortedData.map((match, idx) => {
                const isSelected = selectedMatchId === match.id;
                const outcome = match.status === "ft" ? getOutcomeFromScore(match.score) : null;

                return (
                  <tr
                    key={match.id}
                    onClick={() => onRowClick?.(match)}
                    className={cn(
                      "border-b border-border transition-colors",
                      onRowClick && "cursor-pointer",
                      isSelected
                        ? "bg-accent"
                        : "hover:bg-accent/50"
                    )}
                  >
                    {/* # cell - sticky (row number, match ID, status) */}
                    <td
                      className="sticky left-0 z-10 px-2 py-2.5 text-center box-border"
                      style={{
                        width: ROW_NUM_WIDTH,
                        minWidth: ROW_NUM_WIDTH,
                        maxWidth: ROW_NUM_WIDTH,
                        backgroundColor: isSelected ? "var(--accent)" : "var(--background)",
                      }}
                    >
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="text-[10px] text-muted-foreground/50">#{idx + 1}</span>
                        <CopyableId id={match.id} status={match.status} />
                        <StatusDot status={match.status} />
                      </div>
                    </td>
                    {/* Match cell - sticky */}
                    <td
                      className="sticky z-10 px-3 py-2.5 border-r border-border relative after:absolute after:top-0 after:right-0 after:bottom-0 after:w-4 after:translate-x-full after:bg-gradient-to-r after:from-black/20 after:to-transparent after:pointer-events-none box-border"
                      style={{
                        left: ROW_NUM_WIDTH,
                        width: MATCH_COL_WIDTH,
                        minWidth: MATCH_COL_WIDTH,
                        backgroundColor: isSelected ? "var(--accent)" : "var(--background)",
                        maxWidth: MATCH_COL_WIDTH,
                      }}
                    >
                      <div className="flex flex-col gap-0.5 leading-tight">
                        <div className="flex items-center gap-1.5">
                          <TeamLogo
                            src={getLogoUrl?.(match.home) ?? null}
                            teamName={match.home}
                            size={16}
                          />
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="font-medium text-foreground truncate max-w-[180px]">
                                {match.homeDisplayName}
                              </span>
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              <p>{match.home}</p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <TeamLogo
                            src={getLogoUrl?.(match.away) ?? null}
                            teamName={match.away}
                            size={16}
                          />
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="text-muted-foreground text-sm truncate max-w-[180px]">
                                {match.awayDisplayName}
                              </span>
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              <p>{match.away}</p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                      </div>
                    </td>

                    {/* League cell */}
                    {isVisible("leagueName") && (
                      <td className="px-3 py-2.5" style={{ minWidth: LEAGUE_COL_WIDTH }}>
                        <span className="text-muted-foreground text-sm truncate block max-w-[150px]">
                          {match.leagueName}
                        </span>
                      </td>
                    )}

                    {/* Kickoff cell */}
                    {isVisible("kickoffISO") && (
                      <td className="px-3 py-2.5" style={{ minWidth: KICKOFF_COL_WIDTH }}>
                        <div className="text-sm whitespace-nowrap">
                          <div className="text-foreground">
                            {formatShortDate(match.kickoffISO)}
                          </div>
                          <div className="text-muted-foreground text-xs">
                            {formatTime(match.kickoffISO)}
                          </div>
                        </div>
                      </td>
                    )}

                    {/* Score cell — colored by divergence when applicable */}
                    {isVisible("score") && (() => {
                      const gap20 = match.modelA && match.market
                        ? computeGap20(match.modelA, match.market)
                        : null;
                      const divCategory = gap20?.category;
                      const isDiverge = divCategory === "DISAGREE";
                      const isSFAV = divCategory === "STRONG_FAV_DISAGREE";
                      const hasDiv = isDiverge || isSFAV;

                      // Post-match: did the model win the discrepancy?
                      const isFinished = match.status === "ft";
                      const FAV_TO_OUTCOME = ["home", "draw", "away"] as const;
                      const modelWon = isFinished && hasDiv && gap20 && outcome
                        ? outcome === FAV_TO_OUTCOME[gap20.modelFav]
                        : null;

                      const aTag = match.autopsyTag ? AUTOPSY_CONFIG[match.autopsyTag] : null;

                      return (
                        <td className="px-3 py-2.5 text-center" style={{ minWidth: SCORE_COL_WIDTH }}>
                          <div className="flex flex-col items-center gap-0.5">
                          {match.score ? (
                            hasDiv && gap20 ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span
                                    className={cn(
                                      "inline-flex items-center justify-center font-mono font-bold rounded px-1.5 py-0.5 text-[10px]",
                                      modelWon === true
                                        ? "bg-emerald-500/15 text-emerald-500 border border-emerald-500/25"
                                        : modelWon === false
                                          ? "bg-destructive/15 text-destructive border border-destructive/25"
                                          : isSFAV
                                            ? "bg-destructive/15 text-destructive border border-destructive/25"
                                            : "bg-warning/10 text-warning border border-warning/20",
                                    )}
                                  >
                                    {match.score.home} - {match.score.away}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent side="top" sideOffset={8}>
                                  <p className="text-xs">
                                    Model {["1","X","2"][gap20.modelFav]} vs Market {["1","X","2"][gap20.marketFav]}
                                    {isSFAV && ` (gap ${(Math.abs(gap20.gapOnModelFav)*100).toFixed(0)}pp, mkt fav ${(gap20.marketFavProb*100).toFixed(0)}%)`}
                                    {modelWon === true && " — Model won"}
                                    {modelWon === false && " — Market won"}
                                  </p>
                                </TooltipContent>
                              </Tooltip>
                            ) : (
                              <span className="font-mono font-medium">
                                {match.score.home} - {match.score.away}
                              </span>
                            )
                          ) : (
                            isSFAV && gap20 ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span
                                    className="inline-flex items-center justify-center font-mono font-bold rounded px-1.5 py-0.5 text-[10px] bg-destructive/15 text-destructive border border-destructive/25"
                                  >
                                    SFAV
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent side="top" sideOffset={8}>
                                  <p className="text-xs">
                                    Model {["1","X","2"][gap20.modelFav]} vs Market {["1","X","2"][gap20.marketFav]}
                                    {` (gap ${(Math.abs(gap20.gapOnModelFav)*100).toFixed(0)}pp, mkt fav ${(gap20.marketFavProb*100).toFixed(0)}%)`}
                                  </p>
                                </TooltipContent>
                              </Tooltip>
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )
                          )}
                          {aTag && (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className={cn("inline-flex items-center justify-center rounded px-1 py-px text-[8px] font-semibold uppercase tracking-wide border cursor-default", aTag.color)}>
                                  {aTag.label}
                                </span>
                              </TooltipTrigger>
                              <TooltipContent side="bottom" sideOffset={4}>
                                <p className="text-xs max-w-[220px]">{aTag.descEs}</p>
                              </TooltipContent>
                            </Tooltip>
                          )}
                          </div>
                        </td>
                      );
                    })()}

                    {/* Kelly cell (Trading Core + VORP) */}
                    {isVisible("kelly") && (
                      <td className="px-2 py-2.5" style={{ minWidth: KELLY_COL_WIDTH }}>
                        {(match.vorpApplied || (match.valueBets && match.valueBets.length > 0)) ? (
                          <div className="flex flex-col gap-1">
                            <VorpBadge match={match} />
                            {match.valueBets?.map((vb) => (
                              <KellyBet key={vb.outcome} vb={vb} />
                            ))}
                          </div>
                        ) : (
                          <span className="text-muted-foreground text-xs text-center block">—</span>
                        )}
                      </td>
                    )}

                    {/* Elapsed cell */}
                    {isVisible("elapsed") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: ELAPSED_COL_WIDTH }}>
                        {match.elapsed ? (
                          <span className="text-sm text-muted-foreground">
                            {match.elapsed.min}&apos;{match.elapsed.extra ? ` +${match.elapsed.extra}` : ""}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                    )}

                    {/* Market cell */}
                    {isVisible("market") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.market} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Consensus cell */}
                    {isVisible("consensus") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.consensus} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Pinnacle cell */}
                    {isVisible("pinnacle") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.pinnacle} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Model A cell */}
                    {isVisible("modelA") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.modelA} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Shadow cell */}
                    {isVisible("shadow") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.shadow} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Sensor B cell */}
                    {isVisible("sensorB") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.sensorB} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Family S cell */}
                    {isVisible("familyS") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.familyS} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Ext A cell */}
                    {isVisible("extA") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.extA} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Ext B cell */}
                    {isVisible("extB") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.extB} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Ext C cell */}
                    {isVisible("extC") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.extC} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}

                    {/* Ext D cell */}
                    {isVisible("extD") && (
                      <td className="px-3 py-2.5 text-center" style={{ minWidth: MODEL_COL_WIDTH }}>
                        <ProbabilityCell probs={match.extD} outcome={outcome} compact={compactPredictions} />
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </TooltipProvider>
  );
}
