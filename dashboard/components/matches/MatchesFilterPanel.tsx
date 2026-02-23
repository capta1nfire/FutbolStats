"use client";

import { useMemo } from "react";
import { Trophy, Activity, GitCompareArrows, TrendingUp } from "lucide-react";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { DateRangePicker, type LocalDate } from "./DateRangePicker";
import { MatchSummary, MatchStatus } from "@/lib/types";
import { type DivergenceCategory } from "@/lib/predictions";

/** Status display labels */
const STATUS_LABELS: Record<MatchStatus, string> = {
  scheduled: "Scheduled",
  live: "Live",
  ht: "Half Time",
  ft: "Finished",
  postponed: "Postponed",
  cancelled: "Cancelled",
};

interface MatchesFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  /** Selected date as LocalDate string YYYY-MM-DD */
  selectedDate?: LocalDate;
  /** Callback when selected date changes */
  onDateChange?: (date: LocalDate) => void;
  /** Current matches data (for intelligent filtering) */
  matches?: MatchSummary[];
  /** Selected statuses */
  selectedStatuses?: MatchStatus[];
  /** Callback when status filter changes */
  onStatusChange?: (status: MatchStatus, checked: boolean) => void;
  selectedLeagues: string[];
  /** Selected divergence categories */
  selectedDivergences: DivergenceCategory[];
  searchValue: string;
  onLeagueChange: (league: string, checked: boolean) => void;
  onDivergenceChange: (category: DivergenceCategory, checked: boolean) => void;
  onSearchChange: (value: string) => void;
  /** Callback for "Customize Columns" link click */
  onCustomizeColumnsClick?: () => void;
  /** Whether to show the Customize Columns link */
  showCustomizeColumns?: boolean;
  /** Whether CustomizeColumnsPanel is open (hides collapse button) */
  customizeColumnsOpen?: boolean;
  /** Whether "only value bets" filter is active */
  showOnlyValueBets?: boolean;
  /** Callback when value bet filter changes */
  onValueBetFilterChange?: (checked: boolean) => void;
}

export function MatchesFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedDate,
  onDateChange,
  matches = [],
  selectedStatuses = [],
  onStatusChange,
  selectedLeagues,
  selectedDivergences,
  searchValue,
  onLeagueChange,
  onDivergenceChange,
  onSearchChange,
  onCustomizeColumnsClick,
  showCustomizeColumns = false,
  customizeColumnsOpen = false,
  showOnlyValueBets = false,
  onValueBetFilterChange,
}: MatchesFilterPanelProps) {
  // Get unique leagues from current page matches
  const availableLeagues = useMemo(() => {
    const leagues = new Set<string>();
    for (const match of matches) {
      leagues.add(match.leagueName);
    }
    return Array.from(leagues).sort();
  }, [matches]);

  // Get available statuses from current matches (intelligent filtering)
  const availableStatuses = useMemo(() => {
    const statuses = new Set<MatchStatus>();
    for (const match of matches) {
      statuses.add(match.status);
    }
    const statusOrder: MatchStatus[] = ["scheduled", "live", "ht", "ft", "postponed", "cancelled"];
    return statusOrder.filter((s) => statuses.has(s));
  }, [matches]);

  // Build filter groups
  const filterGroups: FilterGroup[] = useMemo(() => {
    const groups: FilterGroup[] = [];

    // Status filter
    if (availableStatuses.length > 0) {
      groups.push({
        id: "status",
        label: "Status",
        icon: <Activity className="h-4 w-4" strokeWidth={1.5} />,
        options: availableStatuses.map((status) => ({
          id: status,
          label: STATUS_LABELS[status],
          checked: selectedStatuses.includes(status),
        })),
      });
    }

    // League filter
    if (availableLeagues.length > 0) {
      groups.push({
        id: "league",
        label: "League",
        icon: <Trophy className="h-4 w-4" strokeWidth={1.5} />,
        options: availableLeagues.map((league) => ({
          id: league,
          label: league,
          checked: selectedLeagues.includes(league),
        })),
      });
    }

    // Divergence filter (GAP20 model-vs-market)
    groups.push({
      id: "divergence",
      label: "Divergence",
      icon: <GitCompareArrows className="h-4 w-4" strokeWidth={1.5} />,
      options: [
        { id: "AGREE", label: "Agree", checked: selectedDivergences.includes("AGREE") },
        { id: "DISAGREE", label: "Disagree", checked: selectedDivergences.includes("DISAGREE") },
        { id: "STRONG_FAV_DISAGREE", label: "SFAV", checked: selectedDivergences.includes("STRONG_FAV_DISAGREE") },
      ],
    });

    // Trading filter (Kelly value bets)
    groups.push({
      id: "trading",
      label: "Trading",
      icon: <TrendingUp className="h-4 w-4" strokeWidth={1.5} />,
      options: [
        { id: "hasKelly", label: "Con Kelly activo", checked: showOnlyValueBets },
      ],
    });

    return groups;
  }, [
    availableLeagues,
    availableStatuses,
    selectedDivergences,
    selectedLeagues,
    selectedStatuses,
    showOnlyValueBets,
  ]);

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "status") {
      onStatusChange?.(optionId as MatchStatus, checked);
    } else if (groupId === "league") {
      onLeagueChange(optionId, checked);
    } else if (groupId === "divergence") {
      onDivergenceChange(optionId as DivergenceCategory, checked);
    } else if (groupId === "trading") {
      onValueBetFilterChange?.(checked);
    }
  };

  // Date picker as quick filter
  const quickFilterContent = onDateChange ? (
    <DateRangePicker
      value={selectedDate}
      onChange={onDateChange}
    />
  ) : null;

  return (
    <FilterPanel
      title="Matches"
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
      quickFilterContent={quickFilterContent}
      groups={filterGroups}
      onFilterChange={handleFilterChange}
      onSearchChange={onSearchChange}
      searchValue={searchValue}
      onCustomizeColumnsClick={onCustomizeColumnsClick}
      showCustomizeColumns={showCustomizeColumns}
      customizeColumnsOpen={customizeColumnsOpen}
    />
  );
}
