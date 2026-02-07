"use client";

import { useState, useMemo } from "react";
import { useManagersView } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { SearchInput } from "@/components/ui/search-input";
import { AlertCircle, User } from "lucide-react";
import type { ManagerEntry } from "@/lib/types/squad";

interface ManagersViewProps {
  onTeamSelect?: (teamId: number) => void;
}

export function ManagersView({ onTeamSelect }: ManagersViewProps) {
  const { data, isLoading, error } = useManagersView();
  const [search, setSearch] = useState("");
  const [showNewOnly, setShowNewOnly] = useState(false);

  const filtered = useMemo(() => {
    if (!data?.managers) return [];
    let list = data.managers;
    if (showNewOnly) {
      list = list.filter((m) => m.is_new);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (m) =>
          m.manager.name.toLowerCase().includes(q) ||
          m.team_name.toLowerCase().includes(q) ||
          m.league_name?.toLowerCase().includes(q)
      );
    }
    return list;
  }, [data, search, showNewOnly]);

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
          <p className="text-sm text-muted-foreground">Failed to load managers data</p>
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
            <h2 className="text-lg font-semibold">Managers</h2>
            <p className="text-xs text-muted-foreground">
              {data?.total_managers ?? 0} active &middot;{" "}
              {data?.new_managers_count ?? 0} new (&lt;60d)
            </p>
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showNewOnly}
                onChange={(e) => setShowNewOnly(e.target.checked)}
                className="rounded border-border"
              />
              New only
            </label>
            <div className="w-56">
              <SearchInput
                value={search}
                onChange={setSearch}
                placeholder="Search managers..."
              />
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-background border-b">
            <tr className="text-left text-xs text-muted-foreground">
              <th className="px-6 py-2 font-medium">League</th>
              <th className="px-4 py-2 font-medium">Team</th>
              <th className="px-4 py-2 font-medium">Manager</th>
              <th className="px-4 py-2 font-medium">Since</th>
              <th className="px-4 py-2 font-medium text-right">Days</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground">
                  {search || showNewOnly ? "No managers match your filters" : "No managers data"}
                </td>
              </tr>
            ) : (
              filtered.map((entry) => (
                <ManagerRow
                  key={`${entry.team_id}-${entry.manager.external_id}`}
                  entry={entry}
                  onTeamClick={() => onTeamSelect?.(entry.team_id)}
                />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ManagerRow({
  entry,
  onTeamClick,
}: {
  entry: ManagerEntry;
  onTeamClick: () => void;
}) {
  return (
    <tr className="hover:bg-muted/30 transition-colors">
      <td className="px-6 py-2.5 text-muted-foreground">{entry.league_name}</td>
      <td className="px-4 py-2.5">
        <button
          onClick={onTeamClick}
          className="font-medium hover:underline"
        >
          {entry.team_name}
        </button>
      </td>
      <td className="px-4 py-2.5">
        <div className="flex items-center gap-2">
          {entry.manager.photo_url ? (
            <img
              src={entry.manager.photo_url}
              alt={entry.manager.name}
              className="h-6 w-6 rounded-full object-cover"
            />
          ) : (
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-muted">
              <User className="h-3 w-3 text-muted-foreground" />
            </div>
          )}
          <span>{entry.manager.name}</span>
          {entry.is_new && (
            <span className="inline-flex items-center rounded-full bg-[var(--status-warning-bg)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--status-warning-text)] border border-[var(--status-warning-border)]">
              NEW
            </span>
          )}
        </div>
      </td>
      <td className="px-4 py-2.5 text-muted-foreground">
        {entry.manager.start_date
          ? new Date(entry.manager.start_date).toLocaleDateString("en-US", {
              month: "short",
              year: "2-digit",
            })
          : "—"}
      </td>
      <td className="px-4 py-2.5 text-right tabular-nums">
        {entry.tenure_days ?? "—"}
      </td>
    </tr>
  );
}
