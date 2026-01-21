"use client";

import { useMemo } from "react";
import { ColumnDef, VisibilityState } from "@tanstack/react-table";
import { MatchSummary } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { ColumnOption } from "@/components/tables";
import { Badge } from "@/components/ui/badge";
import { StatusDot } from "./StatusDot";

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
  { id: "prediction", label: "Prediction", enableHiding: true },
  { id: "model", label: "Model", enableHiding: true },
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
}: MatchesTableProps) {
  const columns: ColumnDef<MatchSummary>[] = useMemo(
    () => [
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => <StatusDot status={row.original.status} />,
        enableSorting: true,
      },
      {
        id: "match",
        header: "Match",
        cell: ({ row }) => (
          <div className="font-medium text-foreground">
            {row.original.home}{" "}
            <span className="text-muted-foreground">vs</span>{" "}
            {row.original.away}
          </div>
        ),
        enableSorting: false,
      },
      {
        accessorKey: "leagueName",
        header: "League",
        cell: ({ row }) => (
          <span className="text-muted-foreground text-sm">
            {row.original.leagueName}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "kickoffISO",
        header: "Kickoff",
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
        id: "prediction",
        header: "Prediction",
        cell: ({ row }) => {
          const prediction = row.original.prediction;
          if (!prediction) {
            return <span className="text-muted-foreground text-xs">-</span>;
          }

          const pickLabel =
            prediction.pick === "home"
              ? row.original.home.split(" ")[0]
              : prediction.pick === "away"
              ? row.original.away.split(" ")[0]
              : "Draw";

          const prob = prediction.probs
            ? prediction.probs[prediction.pick]
            : null;

          return (
            <Badge variant="secondary" className="text-xs font-normal">
              {pickLabel}
              {prob && (
                <span className="ml-1 text-muted-foreground">
                  {(prob * 100).toFixed(0)}%
                </span>
              )}
            </Badge>
          );
        },
        enableSorting: false,
      },
      {
        id: "model",
        header: "Model",
        cell: ({ row }) => {
          const prediction = row.original.prediction;
          if (!prediction) {
            return <span className="text-muted-foreground text-xs">-</span>;
          }
          return (
            <span className="text-xs text-muted-foreground font-mono">
              {prediction.model}
            </span>
          );
        },
        enableSorting: false,
      },
    ],
    []
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
