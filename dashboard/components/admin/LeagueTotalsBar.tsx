"use client";

import { Badge } from "@/components/ui/badge";
import type { AdminLeaguesTotals } from "@/lib/types";

interface LeagueTotalsBarProps {
  totals: AdminLeaguesTotals;
  unmappedCount: number;
}

export function LeagueTotalsBar({ totals, unmappedCount }: LeagueTotalsBarProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-b border-border text-xs text-muted-foreground shrink-0 flex-wrap">
      <Chip label="Total" value={totals.total_in_db} />
      <Chip label="Active" value={totals.active} />
      <Chip label="Seed" value={totals.seed} />
      <Chip label="Observed" value={totals.observed} />
      <Chip label="In Matches" value={totals.in_matches} />
      <Chip label="With TITAN" value={totals.with_titan_data} />
      {unmappedCount > 0 && (
        <Badge variant="destructive" className="text-[10px]">
          {unmappedCount} unmapped
        </Badge>
      )}
    </div>
  );
}

function Chip({ label, value }: { label: string; value: number }) {
  return (
    <span>
      <span className="text-muted-foreground">{label}:</span>{" "}
      <span className="font-mono font-medium text-foreground">{value}</span>
    </span>
  );
}
