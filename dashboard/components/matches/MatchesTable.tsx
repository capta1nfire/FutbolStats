"use client";

import { useMemo, useCallback } from "react";
import { ColumnDef, VisibilityState } from "@tanstack/react-table";
import { MatchSummary, MatchScore, ProbabilitySet, MatchStatus } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { ColumnOption } from "@/components/tables";
import { StatusDot } from "./StatusDot";
import { TeamLogo } from "@/components/ui/team-logo";
import { toast } from "sonner";
import { Copy } from "lucide-react";
import { cn } from "@/lib/utils";
import { useRegion } from "@/components/providers/RegionProvider";
import { getPredictionPick, getProbabilityCellClasses, type Outcome } from "@/lib/predictions";

// =============================================================================
// Model Accuracy Calculation
// =============================================================================

type ModelKey = "market" | "modelA" | "shadow" | "sensorB";

interface ModelAccuracyStats {
  correct: number;
  total: number;
  accuracy: number | null; // null if no finished matches
}

/**
 * Calculate accuracy stats for all models based on finished matches
 */
function calculateModelAccuracies(
  data: MatchSummary[]
): Record<ModelKey, ModelAccuracyStats> {
  const stats: Record<ModelKey, ModelAccuracyStats> = {
    market: { correct: 0, total: 0, accuracy: null },
    modelA: { correct: 0, total: 0, accuracy: null },
    shadow: { correct: 0, total: 0, accuracy: null },
    sensorB: { correct: 0, total: 0, accuracy: null },
  };

  for (const match of data) {
    // Only count finished matches
    if (match.status !== "ft") continue;

    const outcome = getOutcomeFromMatch(match);
    if (!outcome) continue;

    // Check each model
    const models: { key: ModelKey; probs: ProbabilitySet | undefined }[] = [
      { key: "market", probs: match.market },
      { key: "modelA", probs: match.modelA },
      { key: "shadow", probs: match.shadow },
      { key: "sensorB", probs: match.sensorB },
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

  // Calculate percentages
  for (const key of Object.keys(stats) as ModelKey[]) {
    if (stats[key].total > 0) {
      stats[key].accuracy = (stats[key].correct / stats[key].total) * 100;
    }
  }

  return stats;
}

/**
 * Get outcome from match (helper for accuracy calculation)
 */
function getOutcomeFromMatch(match: MatchSummary): Outcome | null {
  if (!match.score) return null;
  if (match.score.home > match.score.away) return "home";
  if (match.score.home < match.score.away) return "away";
  return "draw";
}

/**
 * Header component with accuracy badge
 */
function ModelHeader({
  label,
  stats,
}: {
  label: string;
  stats: ModelAccuracyStats;
}) {
  const hasStats = stats.accuracy !== null && stats.total > 0;

  return (
    <div className="flex items-center gap-1.5 leading-tight">
      <span>{label}</span>
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

/**
 * Get color classes for a match status
 */
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
    case "scheduled":
    case "ft":
    default:
      return { text: "text-primary", hover: "hover:text-primary-hover" };
  }
}

/**
 * Copyable ID cell - click to copy to clipboard with toast feedback
 * Color matches the status icon color
 */
function CopyableId({ id, status }: { id: number; status: MatchStatus }) {
  const colors = getStatusColor(status);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation(); // Prevent row click
      navigator.clipboard.writeText(String(id));
      toast(
        <div className="flex items-center gap-2">
          <Copy className={cn("h-4 w-4", colors.text)} />
          <span>
            Match ID <span className={cn("font-medium", colors.text)}>{id}</span> copied to clipboard
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

/**
 * Determine actual match outcome from score
 * Returns: "home" | "draw" | "away" | null (if no score yet)
 */
function getOutcome(score?: MatchScore): Outcome | null {
  if (!score) return null;
  if (score.home > score.away) return "home";
  if (score.home < score.away) return "away";
  return "draw";
}

interface ProbabilityCellProps {
  probs?: ProbabilitySet;
  outcome?: Outcome | null;
  /** Compact mode: show only the predicted pick */
  compact?: boolean;
}

/**
 * Cell component for displaying 1X2 probability distribution
 *
 * Handles ties in max probability using epsilon comparison.
 * When match is finished:
 * - Green: single pick was correct
 * - Amber: tied pick, one of them correct (fair handling of ties)
 * - Red: pick(s) were wrong
 * - No color: outcomes the model didn't predict
 *
 * Compact mode: shows only the predicted pick (the max probability outcome)
 */
function ProbabilityCell({ probs, outcome, compact = false }: ProbabilityCellProps) {
  if (!probs) {
    return <div className="text-center"><span className="text-muted-foreground text-xs">-</span></div>;
  }

  // Use the new pick calculation that handles ties fairly
  const pickResult = getPredictionPick(
    { home: probs.home, draw: probs.draw, away: probs.away },
    outcome ?? null
  );

  // Compact mode: show only the pick(s)
  if (compact) {
    const outcomes: { key: Outcome; label: string; prob: number }[] = [
      { key: "home", label: "1", prob: probs.home },
      { key: "draw", label: "X", prob: probs.draw },
      { key: "away", label: "2", prob: probs.away },
    ];

    // Filter to only show top picks
    const picks = outcomes.filter((o) => pickResult.topOutcomes.includes(o.key));

    return (
      <div className="text-xs font-mono leading-tight inline-block">
        {picks.map((pick) => (
          <div key={pick.key} className={getProbabilityCellClasses(pick.key, pickResult)}>
            {pick.label}: {(pick.prob * 100).toFixed(0)}%
          </div>
        ))}
      </div>
    );
  }

  // Full mode: show all 3 outcomes
  return (
    <div className="text-xs font-mono leading-tight inline-block">
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

interface MatchesTableProps {
  data: MatchSummary[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedMatchId?: number | null;
  onRowClick?: (match: MatchSummary) => void;
  /** Column visibility state */
  columnVisibility?: VisibilityState;
  /** Called when column visibility changes */
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
  /** Function to get team logo URL by team name */
  getLogoUrl?: (teamName: string) => string | null;
  /** Compact predictions mode: show only the predicted pick */
  compactPredictions?: boolean;
}

/**
 * Column options for Customize Columns panel
 * Maps column IDs to human-readable labels
 */
export const MATCHES_COLUMN_OPTIONS: ColumnOption[] = [
  { id: "rowIndex", label: "#", enableHiding: false }, // Row number, always visible
  { id: "id", label: "ID", enableHiding: true }, // Status + Match ID
  { id: "match", label: "Match", enableHiding: false }, // Always visible
  { id: "leagueName", label: "League", enableHiding: true },
  { id: "kickoffISO", label: "Kickoff", enableHiding: true },
  { id: "score", label: "Score", enableHiding: true },
  { id: "elapsed", label: "Elapsed", enableHiding: true },
  { id: "market", label: "Market", enableHiding: true },
  { id: "modelA", label: "Model A", enableHiding: true },
  { id: "shadow", label: "Shadow", enableHiding: true },
  { id: "sensorB", label: "Sensor B", enableHiding: true },
];

/**
 * Default column visibility for Matches table
 */
export const MATCHES_DEFAULT_VISIBILITY: VisibilityState = {
  // All columns visible by default
};

export function MatchesTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedMatchId,
  onRowClick,
  columnVisibility,
  onColumnVisibilityChange,
  getLogoUrl,
  compactPredictions = false,
}: MatchesTableProps) {
  const { formatShortDate, formatTime } = useRegion();

  // Calculate model accuracies from finished matches
  const modelAccuracies = useMemo(() => calculateModelAccuracies(data), [data]);

  const columns: ColumnDef<MatchSummary>[] = useMemo(
    () => [
      {
        id: "rowIndex",
        header: "#",
        size: 45,
        cell: ({ row }) => (
          <span className="text-xs text-muted-foreground tabular-nums">
            {row.index + 1}
          </span>
        ),
        enableSorting: false,
      },
      {
        id: "id",
        header: "ID",
        size: 100,
        cell: ({ row }) => (
          <div className="flex items-center gap-2">
            <StatusDot status={row.original.status} />
            <CopyableId id={row.original.id} status={row.original.status} />
          </div>
        ),
        enableSorting: true,
      },
      {
        id: "match",
        header: "Match",
        size: 220,
        cell: ({ row }) => (
          <div className="flex flex-col gap-0.5 leading-tight">
            <div className="flex items-center gap-1.5">
              <TeamLogo
                src={getLogoUrl?.(row.original.home) ?? null}
                teamName={row.original.home}
                size={16}
              />
              <span className="font-medium text-foreground truncate">{row.original.home}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <TeamLogo
                src={getLogoUrl?.(row.original.away) ?? null}
                teamName={row.original.away}
                size={16}
              />
              <span className="text-muted-foreground text-sm truncate">{row.original.away}</span>
            </div>
          </div>
        ),
        enableSorting: false,
      },
      {
        accessorKey: "leagueName",
        header: "League",
        size: 180,
        cell: ({ row }) => (
          <span className="text-muted-foreground text-sm truncate block">
            {row.original.leagueName}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "kickoffISO",
        header: "Kickoff",
        size: 90,
        cell: ({ row }) => {
          const isoUtc = row.original.kickoffISO;
          return (
            <div className="text-sm">
              <div className="text-foreground">
                {formatShortDate(isoUtc)}
              </div>
              <div className="text-muted-foreground text-xs">
                {formatTime(isoUtc)}
              </div>
            </div>
          );
        },
        enableSorting: true,
      },
      {
        id: "score",
        header: "Score",
        size: 70,
        cell: ({ row }) => {
          const score = row.original.score;
          if (!score) {
            return <span className="text-muted-foreground">-</span>;
          }
          return (
            <span className="font-mono font-medium">
              {score.home} - {score.away}
            </span>
          );
        },
        enableSorting: false,
      },
      {
        id: "elapsed",
        header: "Elapsed",
        size: 70,
        cell: ({ row }) => {
          const elapsed = row.original.elapsed;
          if (!elapsed) {
            return <span className="text-muted-foreground">-</span>;
          }
          return (
            <span className="text-sm text-muted-foreground">
              {elapsed.min}&apos;{elapsed.extra ? ` +${elapsed.extra}` : ""}
            </span>
          );
        },
        enableSorting: false,
      },
      {
        id: "market",
        header: () => <ModelHeader label="Market" stats={modelAccuracies.market} />,
        size: 110,
        meta: { cellClassName: "text-center", headerClassName: "text-center" },
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.market}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
            compact={compactPredictions}
          />
        ),
        enableSorting: false,
      },
      {
        id: "modelA",
        header: () => <ModelHeader label="Model A" stats={modelAccuracies.modelA} />,
        size: 110,
        meta: { cellClassName: "text-center", headerClassName: "text-center" },
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.modelA}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
            compact={compactPredictions}
          />
        ),
        enableSorting: false,
      },
      {
        id: "shadow",
        header: () => <ModelHeader label="Shadow" stats={modelAccuracies.shadow} />,
        size: 110,
        meta: { cellClassName: "text-center", headerClassName: "text-center" },
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.shadow}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
            compact={compactPredictions}
          />
        ),
        enableSorting: false,
      },
      {
        id: "sensorB",
        header: () => <ModelHeader label="Sensor B" stats={modelAccuracies.sensorB} />,
        size: 110,
        meta: { cellClassName: "text-center", headerClassName: "text-center" },
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.sensorB}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
            compact={compactPredictions}
          />
        ),
        enableSorting: false,
      },
    ],
    [getLogoUrl, formatShortDate, formatTime, modelAccuracies, compactPredictions]
  );

  return (
    <DataTable
      columns={columns}
      data={data}
      isLoading={isLoading}
      error={error}
      onRetry={onRetry}
      selectedRowId={selectedMatchId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No matches found"
      columnVisibility={columnVisibility}
      onColumnVisibilityChange={onColumnVisibilityChange}
    />
  );
}
