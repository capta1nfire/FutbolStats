"use client";

import { useState, useMemo } from "react";
import { usePlayersView } from "@/lib/hooks";
import { InjuryList } from "@/components/squad";
import { Loader } from "@/components/ui/loader";
import { SearchInput } from "@/components/ui/search-input";
import { AlertCircle, ChevronDown, ChevronRight } from "lucide-react";

interface PlayersViewProps {
  onTeamSelect?: (teamId: number) => void;
}

export function PlayersView({ onTeamSelect }: PlayersViewProps) {
  const { data, isLoading, error } = usePlayersView();
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const filtered = useMemo(() => {
    if (!data?.leagues) return [];
    if (!search.trim()) return data.leagues;

    const q = search.toLowerCase();
    return data.leagues
      .map((league) => ({
        ...league,
        teams: league.teams
          .map((team) => ({
            ...team,
            injuries: team.injuries.filter(
              (inj) =>
                inj.player_name.toLowerCase().includes(q) ||
                inj.injury_reason?.toLowerCase().includes(q)
            ),
          }))
          .filter((team) =>
            team.injuries.length > 0 || team.name.toLowerCase().includes(q)
          ),
      }))
      .filter(
        (league) =>
          league.teams.length > 0 || league.name.toLowerCase().includes(q)
      );
  }, [data, search]);

  const toggleExpand = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <AlertCircle className="mx-auto mb-2 h-8 w-8 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Failed to load player data</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Active Absences</h2>
            <p className="text-xs text-muted-foreground">
              {data?.total_absences ?? 0} players &middot; Upcoming 14 days
            </p>
          </div>
          <div className="w-64">
            <SearchInput
              value={search}
              onChange={setSearch}
              placeholder="Search players..."
            />
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {filtered.length === 0 ? (
          <div className="flex h-40 items-center justify-center">
            <p className="text-sm text-muted-foreground">
              {search ? "No players match your search" : "No active absences"}
            </p>
          </div>
        ) : (
          filtered.map((league) => (
            <div key={league.league_id} className="space-y-2">
              {/* League header */}
              <button
                onClick={() => toggleExpand(`league-${league.league_id}`)}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm font-medium hover:bg-muted/50 transition-colors"
              >
                {expanded.has(`league-${league.league_id}`) ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                )}
                <span>{league.name}</span>
                <span className="text-xs text-muted-foreground">
                  ({league.absences_count})
                </span>
              </button>

              {/* Teams (auto-expanded or toggled) */}
              {(expanded.has(`league-${league.league_id}`) || search.trim()) && (
                <div className="ml-6 space-y-3">
                  {league.teams.map((team) => (
                    <div key={team.team_id} className="space-y-1">
                      <button
                        onClick={() => onTeamSelect?.(team.team_id)}
                        className="flex items-center gap-2 text-sm font-medium hover:underline"
                      >
                        {team.name}
                        <span className="text-xs text-muted-foreground">
                          ({team.injuries.length})
                        </span>
                      </button>
                      <div className="ml-2">
                        <InjuryList injuries={team.injuries} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
