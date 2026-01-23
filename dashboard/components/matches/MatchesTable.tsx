"use client";

import { useMemo } from "react";
import { ColumnDef, VisibilityState } from "@tanstack/react-table";
import { MatchSummary, MatchScore, ProbabilitySet } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { ColumnOption } from "@/components/tables";
import { StatusDot } from "./StatusDot";
import { TeamLogo } from "@/components/ui/team-logo";

/**
 * Determine actual match outcome from score
 * Returns: "home" | "draw" | "away" | null (if no score yet)
 */
function getOutcome(score?: MatchScore): "home" | "draw" | "away" | null {
  if (!score) return null;
  if (score.home > score.away) return "home";
  if (score.home < score.away) return "away";
  return "draw";
}

interface ProbabilityCellProps {
  probs?: ProbabilitySet;
  outcome?: "home" | "draw" | "away" | null;
}

/**
 * Cell component for displaying 1X2 probability distribution
 * Shows pick with highest probability highlighted
 * When match is finished (outcome provided), shows green/red badges
 */
function ProbabilityCell({ probs, outcome }: ProbabilityCellProps) {
  if (!probs) {
    return <span className="text-muted-foreground text-xs">-</span>;
  }

  // Find the pick with highest probability (the model's prediction)
  const maxProb = Math.max(probs.home, probs.draw, probs.away);

  // Style for each outcome based on whether it matches the actual result
  const getStyle = (probType: "home" | "draw" | "away", prob: number) => {
    const isPick = prob === maxProb;

    // If match not finished yet, just highlight the pick
    if (!outcome) {
      return isPick ? "text-foreground font-medium" : "text-muted-foreground";
    }

    // Match finished - show result badges
    const isCorrect = outcome === probType;

    if (isCorrect) {
      // Green badge for correct outcome
      return "text-success font-medium";
    } else {
      // Red badge for wrong outcomes
      return "text-error/70";
    }
  };

  return (
    <div className="flex flex-col text-xs font-mono leading-tight">
      <span className={getStyle("home", probs.home)}>
        1: {(probs.home * 100).toFixed(0)}%
      </span>
      <span className={getStyle("draw", probs.draw)}>
        X: {(probs.draw * 100).toFixed(0)}%
      </span>
      <span className={getStyle("away", probs.away)}>
        2: {(probs.away * 100).toFixed(0)}%
      </span>
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
}

/**
 * Column options for Customize Columns panel
 * Maps column IDs to human-readable labels
 */
export const MATCHES_COLUMN_OPTIONS: ColumnOption[] = [
  { id: "status", label: "Status", enableHiding: false }, // Always visible
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
}: MatchesTableProps) {
  const columns: ColumnDef<MatchSummary>[] = useMemo(
    () => [
      {
        accessorKey: "status",
        header: "Status",
        size: 100,
        cell: ({ row }) => <StatusDot status={row.original.status} />,
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
          const date = new Date(row.original.kickoffISO);
          return (
            <div className="text-sm">
              <div className="text-foreground">
                {date.toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })}
              </div>
              <div className="text-muted-foreground text-xs">
                {date.toLocaleTimeString("en-US", {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
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
        header: "Market",
        size: 90,
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.market}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
          />
        ),
        enableSorting: false,
      },
      {
        id: "modelA",
        header: "Model A",
        size: 90,
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.modelA}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
          />
        ),
        enableSorting: false,
      },
      {
        id: "shadow",
        header: "Shadow",
        size: 90,
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.shadow}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
          />
        ),
        enableSorting: false,
      },
      {
        id: "sensorB",
        header: "Sensor B",
        size: 90,
        cell: ({ row }) => (
          <ProbabilityCell
            probs={row.original.sensorB}
            outcome={row.original.status === "ft" ? getOutcome(row.original.score) : null}
          />
        ),
        enableSorting: false,
      },
    ],
    [getLogoUrl]
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
