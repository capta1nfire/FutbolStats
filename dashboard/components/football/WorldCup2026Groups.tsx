"use client";

import Image from "next/image";
import { useWorldCupGroups } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Loader } from "@/components/ui/loader";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  RefreshCw,
  AlertTriangle,
  Trophy,
  Users,
  ArrowLeft,
  ChevronRight,
  Clock,
  Globe,
  XCircle,
} from "lucide-react";
import type { WorldCupGroupWithTeams, WorldCupStandingEntry } from "@/lib/types";

interface WorldCup2026GroupsProps {
  onBack: () => void;
  onGroupSelect: (group: string) => void;
}

/**
 * Team Flag Component
 */
function TeamFlag({ name, logoUrl, size = 16 }: { name: string; logoUrl: string | null; size?: number }) {
  // Try logo_url first, then country flag fallback
  if (logoUrl) {
    return (
      <img
        src={logoUrl}
        alt={name}
        width={size}
        height={size}
        className="rounded-full object-cover"
      />
    );
  }

  const isoCode = getCountryIsoCode(name);
  if (isoCode) {
    return (
      <Image
        src={`/flags/${isoCode}.svg`}
        alt={name}
        width={size}
        height={size}
        className="rounded-full object-cover"
      />
    );
  }

  return <Globe className="text-muted-foreground" style={{ width: size, height: size }} />;
}

/**
 * Group Card Component
 * Uses new backend contract: group.teams[] array
 */
function GroupCard({
  group,
  onSelect,
}: {
  group: WorldCupGroupWithTeams;
  onSelect: () => void;
}) {
  // Get top 2 teams by position for preview
  const topTeams = [...group.teams]
    .sort((a, b) => a.position - b.position)
    .slice(0, 2);

  return (
    <button
      onClick={onSelect}
      className="w-full text-left bg-card border border-border rounded-lg p-4 hover:border-border-hover hover:bg-muted/50 transition-colors group"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <span className="text-sm font-bold text-primary">
              {group.group.replace("Group ", "")}
            </span>
          </div>
          <div>
            <h3 className="text-sm font-medium text-foreground">{group.group}</h3>
            <p className="text-xs text-muted-foreground">
              {group.teams.length} teams
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {/* Top teams preview */}
          {topTeams.length > 0 && (
            <div className="hidden sm:flex items-center gap-1">
              {topTeams.map((team) => (
                <TeamFlag
                  key={team.team_id || team.name}
                  name={team.name}
                  logoUrl={team.logo_url}
                  size={20}
                />
              ))}
            </div>
          )}
          <ChevronRight className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
        </div>
      </div>
    </button>
  );
}

/**
 * WorldCup2026Groups Component (Col 4)
 *
 * Displays World Cup 2026 groups list:
 * - Shows each group with team count
 * - Preview of top 2 teams by position
 */
export function WorldCup2026Groups({ onBack, onGroupSelect }: WorldCup2026GroupsProps) {
  const { data, isLoading, error, refetch } = useWorldCupGroups();

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader size="md" />
      </div>
    );
  }

  // Error state
  if (error || !data) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center max-w-md">
          <AlertTriangle className="h-12 w-12 text-[var(--status-warning-text)]" />
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-2">
              Groups Data Unavailable
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {error?.message || "Unable to fetch World Cup 2026 groups"}
            </p>
          </div>
          <div className="flex gap-2">
            <Button onClick={onBack} variant="outline">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <Button onClick={() => refetch()} variant="secondary">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        {/* Header with Back button */}
        <div className="flex items-start gap-4">
          <Button variant="ghost" size="icon" onClick={onBack} className="shrink-0 mt-1">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <Trophy className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-semibold text-foreground">
                World Cup 2026 Groups
              </h1>
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              {data.totals.groups_count} groups · {data.totals.teams_count} teams
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>

        {/* Status Badge */}
        {data.status === "not_ready" && (
          <div className="flex items-center gap-2 px-3 py-2 bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)] rounded-lg">
            <Clock className="h-4 w-4 text-[var(--status-warning-text)]" />
            <span className="text-sm text-[var(--status-warning-text)]">
              Groups not yet finalized
            </span>
          </div>
        )}
        {data.status === "disabled" && (
          <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 border border-border rounded-lg">
            <XCircle className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              Groups feature disabled
            </span>
          </div>
        )}

        {/* Groups Grid */}
        {data.groups.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.groups.map((group) => (
              <GroupCard
                key={group.group}
                group={group}
                onSelect={() => onGroupSelect(group.group)}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-12 bg-muted/30 rounded-lg">
            <Users className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-sm font-medium text-foreground mb-2">
              No Groups Available
            </h3>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              Group stage draw has not been completed yet.
              Check back after the draw for group compositions.
            </p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
