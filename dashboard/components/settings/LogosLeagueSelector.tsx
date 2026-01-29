"use client";

import { useLogosLeagues } from "@/lib/hooks";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Loader2, AlertTriangle } from "lucide-react";

interface LogosLeagueSelectorProps {
  selectedLeagueId: number | null;
  onSelect: (leagueId: number | null) => void;
}

export function LogosLeagueSelector({
  selectedLeagueId,
  onSelect,
}: LogosLeagueSelectorProps) {
  const { data: leagues, isLoading, error } = useLogosLeagues();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Loading leagues...</span>
      </div>
    );
  }

  if (error || !leagues) {
    return (
      <div className="flex items-center gap-2 text-sm text-[var(--status-error-text)]">
        <AlertTriangle className="h-4 w-4" />
        <span>Failed to load leagues</span>
      </div>
    );
  }

  const selectedLeague = leagues.find((l) => l.leagueId === selectedLeagueId);

  return (
    <div className="space-y-3">
      <Select
        value={selectedLeagueId?.toString() || ""}
        onValueChange={(v) => onSelect(v ? parseInt(v, 10) : null)}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Select a league..." />
        </SelectTrigger>
        <SelectContent>
          {leagues.map((league) => (
            <SelectItem key={league.leagueId} value={league.leagueId.toString()}>
              <div className="flex items-center justify-between w-full gap-4">
                <span>
                  {league.country} - {league.name}
                </span>
                <span className="text-xs text-muted-foreground">
                  {league.teamCount} teams
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {selectedLeague && (
        <div className="flex items-center gap-3 text-sm">
          <Badge variant="outline" className="bg-surface">
            {selectedLeague.teamCount} teams
          </Badge>
          {selectedLeague.pendingCount > 0 && (
            <Badge
              variant="outline"
              className="bg-[var(--status-warning-bg)] text-[var(--status-warning-text)] border-[var(--status-warning-border)]"
            >
              {selectedLeague.pendingCount} pending
            </Badge>
          )}
          {selectedLeague.readyCount > 0 && (
            <Badge
              variant="outline"
              className="bg-[var(--status-success-bg)] text-[var(--status-success-text)] border-[var(--status-success-border)]"
            >
              {selectedLeague.readyCount} ready
            </Badge>
          )}
          {selectedLeague.errorCount > 0 && (
            <Badge
              variant="outline"
              className="bg-[var(--status-error-bg)] text-[var(--status-error-text)] border-[var(--status-error-border)]"
            >
              {selectedLeague.errorCount} errors
            </Badge>
          )}
        </div>
      )}
    </div>
  );
}
