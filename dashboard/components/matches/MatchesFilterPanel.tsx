"use client";

import { useMemo } from "react";
import { Circle, Trophy, Calendar } from "lucide-react";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { MatchStatus } from "@/lib/types";
import { getStatusCountsMock, getLeaguesMock } from "@/lib/mocks";

interface MatchesFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedStatuses: MatchStatus[];
  selectedLeagues: string[];
  searchValue: string;
  onStatusChange: (status: MatchStatus, checked: boolean) => void;
  onLeagueChange: (league: string, checked: boolean) => void;
  onSearchChange: (value: string) => void;
  /** Callback for "Customize Columns" link click */
  onCustomizeColumnsClick?: () => void;
  /** Whether to show the Customize Columns link */
  showCustomizeColumns?: boolean;
  /** Whether CustomizeColumnsPanel is open (hides collapse button) */
  customizeColumnsOpen?: boolean;
}

export function MatchesFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedStatuses,
  selectedLeagues,
  searchValue,
  onStatusChange,
  onLeagueChange,
  onSearchChange,
  onCustomizeColumnsClick,
  showCustomizeColumns = false,
  customizeColumnsOpen = false,
}: MatchesFilterPanelProps) {
  const statusCounts = useMemo(() => getStatusCountsMock(), []);
  const leagues = useMemo(() => getLeaguesMock(), []);

  const filterGroups: FilterGroup[] = useMemo(
    () => [
      {
        id: "status",
        label: "Status",
        icon: <Circle className="h-4 w-4" />,
        options: [
          {
            id: "scheduled",
            label: "Scheduled",
            count: statusCounts.scheduled,
            checked: selectedStatuses.includes("scheduled"),
          },
          {
            id: "live",
            label: "Live",
            count: statusCounts.live,
            checked: selectedStatuses.includes("live"),
          },
          {
            id: "ht",
            label: "Half Time",
            count: statusCounts.ht,
            checked: selectedStatuses.includes("ht"),
          },
          {
            id: "ft",
            label: "Full Time",
            count: statusCounts.ft,
            checked: selectedStatuses.includes("ft"),
          },
        ],
      },
      {
        id: "league",
        label: "League",
        icon: <Trophy className="h-4 w-4" />,
        options: leagues.map((league) => ({
          id: league,
          label: league,
          checked: selectedLeagues.includes(league),
        })),
      },
      {
        id: "date",
        label: "Date Range",
        icon: <Calendar className="h-4 w-4" />,
        options: [
          { id: "today", label: "Today", checked: false },
          { id: "tomorrow", label: "Tomorrow", checked: false },
          { id: "week", label: "This Week", checked: false },
        ],
      },
    ],
    [statusCounts, leagues, selectedStatuses, selectedLeagues]
  );

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "status") {
      onStatusChange(optionId as MatchStatus, checked);
    } else if (groupId === "league") {
      onLeagueChange(optionId, checked);
    }
    // Date range filter not implemented in Phase 0
  };

  return (
    <FilterPanel
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
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
