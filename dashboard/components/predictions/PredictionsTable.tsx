"use client";

import { useMemo } from "react";
import { ColumnDef, VisibilityState } from "@tanstack/react-table";
import { PredictionRow } from "@/lib/types";
import { DataTable, ColumnOption } from "@/components/tables";
import { PredictionStatusBadge } from "./PredictionStatusBadge";
import { ModelBadge } from "./ModelBadge";
import { PickBadge } from "./PickBadge";
import { TeamLogo } from "@/components/ui/team-logo";
import { formatDistanceToNow } from "@/lib/utils";

interface PredictionsTableProps {
  data: PredictionRow[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  selectedPredictionId?: number | null;
  onRowClick?: (prediction: PredictionRow) => void;
  columnVisibility?: VisibilityState;
  onColumnVisibilityChange?: (visibility: VisibilityState) => void;
  /** Function to get team logo URL by team name */
  getLogoUrl?: (teamName: string) => string | null;
}

/**
 * Column options for Customize Columns panel
 */
export const PREDICTIONS_COLUMN_OPTIONS: ColumnOption[] = [
  { id: "status", label: "Status", enableHiding: false },
  { id: "matchLabel", label: "Match", enableHiding: false },
  { id: "kickoffISO", label: "Kickoff", enableHiding: true },
  { id: "model", label: "Model", enableHiding: true },
  { id: "probs", label: "Probabilities", enableHiding: true },
  { id: "pick", label: "Pick", enableHiding: false },
  { id: "generatedAt", label: "Generated", enableHiding: true },
];

/**
 * Default column visibility for Predictions table
 */
export const PREDICTIONS_DEFAULT_VISIBILITY: VisibilityState = {};

export function PredictionsTable({
  data,
  isLoading,
  error,
  onRetry,
  selectedPredictionId,
  onRowClick,
  columnVisibility,
  onColumnVisibilityChange,
  getLogoUrl,
}: PredictionsTableProps) {
  const columns = useMemo<ColumnDef<PredictionRow>[]>(
    () => [
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => (
          <PredictionStatusBadge status={row.original.status} />
        ),
        enableSorting: true,
      },
      {
        accessorKey: "matchLabel",
        header: "Match",
        cell: ({ row }) => (
          <div className="space-y-0.5">
            <div className="flex items-center gap-1.5">
              <TeamLogo
                src={getLogoUrl?.(row.original.home) ?? null}
                teamName={row.original.home}
                size={14}
              />
              <span className="text-sm font-medium text-foreground truncate">
                {row.original.home}
              </span>
              <span className="text-muted-foreground text-xs">vs</span>
              <TeamLogo
                src={getLogoUrl?.(row.original.away) ?? null}
                teamName={row.original.away}
                size={14}
              />
              <span className="text-sm text-muted-foreground truncate">
                {row.original.away}
              </span>
            </div>
            <div className="text-xs text-muted-foreground">
              {row.original.leagueName}
            </div>
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "kickoffISO",
        header: "Kickoff",
        cell: ({ row }) => (
          <span className="text-sm text-muted-foreground">
            {formatDistanceToNow(row.original.kickoffISO)}
          </span>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "model",
        header: "Model",
        cell: ({ row }) => <ModelBadge model={row.original.model} />,
        enableSorting: true,
      },
      {
        id: "probs",
        header: "Probabilities",
        cell: ({ row }) => {
          const probs = row.original.probs;
          if (!probs) {
            return <span className="text-muted-foreground">-</span>;
          }
          return (
            <div className="text-xs font-mono text-muted-foreground">
              <span className="text-foreground">{(probs.home * 100).toFixed(0)}%</span>
              {" / "}
              <span>{(probs.draw * 100).toFixed(0)}%</span>
              {" / "}
              <span>{(probs.away * 100).toFixed(0)}%</span>
            </div>
          );
        },
        enableSorting: false,
      },
      {
        accessorKey: "pick",
        header: "Pick",
        cell: ({ row }) => {
          const pick = row.original.pick;
          if (!pick) {
            return <span className="text-muted-foreground">-</span>;
          }
          const isCorrect =
            row.original.status === "evaluated" && row.original.result
              ? row.original.pick === row.original.result
              : undefined;
          return <PickBadge pick={pick} isCorrect={isCorrect} />;
        },
        enableSorting: true,
      },
      {
        accessorKey: "generatedAt",
        header: "Generated",
        cell: ({ row }) =>
          row.original.generatedAt ? (
            <span className="text-sm text-muted-foreground">
              {formatDistanceToNow(row.original.generatedAt)}
            </span>
          ) : (
            <span className="text-muted-foreground">-</span>
          ),
        enableSorting: true,
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
      selectedRowId={selectedPredictionId}
      onRowClick={onRowClick}
      getRowId={(row) => row.id}
      emptyMessage="No predictions found"
      columnVisibility={columnVisibility}
      onColumnVisibilityChange={onColumnVisibilityChange}
    />
  );
}
