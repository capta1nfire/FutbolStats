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
  { id: "id", label: "ID", enableHiding: false }, // Status + Match ID combined, always visible
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
  const { formatShortDate, formatTime } = useRegion();

  const columns: ColumnDef<MatchSummary>[] = useMemo(
    () => [
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
    [getLogoUrl, formatShortDate, formatTime]
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
