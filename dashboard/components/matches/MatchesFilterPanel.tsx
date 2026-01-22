"use client";

import { useMemo } from "react";
import { Trophy, Calendar } from "lucide-react";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { getLeaguesMock } from "@/lib/mocks";
import { MatchesViewTabs, MatchesView } from "./MatchesViewTabs";

/** Time range options for filtering */
export type TimeRange = "24h" | "48h" | "7d";

interface MatchesFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  /** Current view: upcoming or finished */
  activeView: MatchesView;
  /** Callback when view changes */
  onViewChange: (view: MatchesView) => void;
  /** Selected time range */
  selectedTimeRange: TimeRange;
  /** Callback when time range changes */
  onTimeRangeChange: (range: TimeRange) => void;
  selectedLeagues: string[];
  searchValue: string;
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
  activeView,
  onViewChange,
  selectedTimeRange,
  onTimeRangeChange,
  selectedLeagues,
  searchValue,
  onLeagueChange,
  onSearchChange,
  onCustomizeColumnsClick,
  showCustomizeColumns = false,
  customizeColumnsOpen = false,
}: MatchesFilterPanelProps) {
  const leagues = useMemo(() => getLeaguesMock(), []);

  // Time range labels depend on active view
  const timeRangeLabels = useMemo(() => {
    if (activeView === "upcoming") {
      return {
        "24h": "Next 24 hours",
        "48h": "Next 48 hours",
        "7d": "Next 7 days",
      };
    } else {
      return {
        "24h": "Last 24 hours",
        "48h": "Last 48 hours",
        "7d": "Last 7 days",
      };
    }
  }, [activeView]);

  const filterGroups: FilterGroup[] = useMemo(
    () => [
      {
        id: "timeRange",
        label: "Time Range",
        icon: <Calendar className="h-4 w-4" strokeWidth={1.5} />,
        options: [
          {
            id: "24h",
            label: timeRangeLabels["24h"],
            checked: selectedTimeRange === "24h",
          },
          {
            id: "48h",
            label: timeRangeLabels["48h"],
            checked: selectedTimeRange === "48h",
          },
          {
            id: "7d",
            label: timeRangeLabels["7d"],
            checked: selectedTimeRange === "7d",
          },
        ],
      },
      {
        id: "league",
        label: "League",
        icon: <Trophy className="h-4 w-4" strokeWidth={1.5} />,
        options: leagues.map((league) => ({
          id: league,
          label: league,
          checked: selectedLeagues.includes(league),
        })),
      },
    ],
    [leagues, selectedLeagues, selectedTimeRange, timeRangeLabels]
  );

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "timeRange" && checked) {
      // Time range is single-select (radio behavior)
      onTimeRangeChange(optionId as TimeRange);
    } else if (groupId === "league") {
      onLeagueChange(optionId, checked);
    }
  };

  // Render tabs in header content slot
  const headerContent = (
    <MatchesViewTabs activeView={activeView} onViewChange={onViewChange} />
  );

  return (
    <FilterPanel
      title="Matches"
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
      headerContent={headerContent}
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
