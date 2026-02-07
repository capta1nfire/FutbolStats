"use client";

import { AlertCircle } from "lucide-react";
import type { PlayerInjury } from "@/lib/types/squad";

interface InjuryListProps {
  injuries: PlayerInjury[];
  compact?: boolean;
}

function getInjuryStyle(type: string) {
  if (type === "Missing Fixture") {
    return {
      bg: "bg-[var(--status-error-bg)]",
      text: "text-[var(--status-error-text)]",
      border: "border-[var(--status-error-border)]",
      label: "Out",
    };
  }
  // Questionable, Doubtful, or any other type
  return {
    bg: "bg-[var(--status-warning-bg)]",
    text: "text-[var(--status-warning-text)]",
    border: "border-[var(--status-warning-border)]",
    label: type === "Questionable" ? "GTD" : "Doubt",
  };
}

export function InjuryList({ injuries, compact = false }: InjuryListProps) {
  if (injuries.length === 0) {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <AlertCircle className="h-3.5 w-3.5" />
        No absences reported
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {injuries.map((injury, i) => {
        const style = getInjuryStyle(injury.injury_type);
        return (
          <div
            key={`${injury.player_name}-${i}`}
            className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm"
          >
            <span
              className={`inline-flex items-center rounded-full px-1.5 py-0.5 text-[10px] font-medium border ${style.bg} ${style.text} ${style.border}`}
            >
              {style.label}
            </span>
            <span className="truncate font-medium">{injury.player_name}</span>
            {!compact && injury.injury_reason && (
              <span className="truncate text-xs text-muted-foreground">
                {injury.injury_reason}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
