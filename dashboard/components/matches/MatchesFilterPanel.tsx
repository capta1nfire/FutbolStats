"use client";

import { useMemo } from "react";
import { Trophy, Calendar, Activity } from "lucide-react";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { MatchesViewTabs, MatchesView } from "./MatchesViewTabs";
import { DateRangePicker, DateRangeValue } from "./DateRangePicker";
import { MatchSummary, MatchStatus } from "@/lib/types";

/** Time range options for filtering */
export type TimeRange = "today" | "24h" | "48h" | "7d";

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
  /** Current view: upcoming, finished, or calendar */
  activeView: MatchesView;
  /** Callback when view changes */
  onViewChange: (view: MatchesView) => void;
  /** Selected time range (for upcoming/finished views) */
  selectedTimeRange: TimeRange;
  /** Callback when time range changes */
  onTimeRangeChange: (range: TimeRange) => void;
  /** Custom date range (for calendar view) */
  customDateRange?: DateRangeValue;
  /** Callback when custom date range changes */
  onCustomDateRangeChange?: (range: DateRangeValue) => void;
  /** Current matches data (for intelligent filtering) */
  matches?: MatchSummary[];
  /** Selected statuses (for calendar view) */
  selectedStatuses?: MatchStatus[];
  /** Callback when status filter changes */
  onStatusChange?: (status: MatchStatus, checked: boolean) => void;
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
  customDateRange,
  onCustomDateRangeChange,
  matches = [],
  selectedStatuses = [],
  onStatusChange,
  selectedLeagues,
  searchValue,
  onLeagueChange,
  onSearchChange,
  onCustomizeColumnsClick,
  showCustomizeColumns = false,
  customizeColumnsOpen = false,
}: MatchesFilterPanelProps) {
  // Get unique leagues from current page matches
  // Note: These are from the current page only, not the full dataset
  const availableLeagues = useMemo(() => {
    const leagues = new Set<string>();
    for (const match of matches) {
      leagues.add(match.leagueName);
    }
    return Array.from(leagues).sort();
  }, [matches]);

  // Time range labels depend on active view
  const timeRangeLabels = useMemo(() => {
    if (activeView === "upcoming") {
      return {
        "today": "Today",
        "24h": "Next 24 hours",
        "48h": "Next 48 hours",
        "7d": "Next 7 days",
      };
    } else {
      return {
        "today": "Today",
        "24h": "Last 24 hours",
        "48h": "Last 48 hours",
        "7d": "Last 7 days",
      };
    }
  }, [activeView]);

  // Build filter groups based on active view
  const filterGroups: FilterGroup[] = useMemo(() => {
    const groups: FilterGroup[] = [];

    // All possible statuses for calendar view filter
    const allStatuses: MatchStatus[] = [
      "scheduled",
      "live",
      "ht",
      "ft",
      "postponed",
      "cancelled",
    ];

    // Only show time range filter for upcoming/finished views (not calendar)
    if (activeView !== "calendar") {
      groups.push({
        id: "timeRange",
        label: "Time Range",
        icon: <Calendar className="h-4 w-4" strokeWidth={1.5} />,
        options: [
          {
            id: "today",
            label: timeRangeLabels["today"],
            checked: selectedTimeRange === "today",
          },
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
      });
    }

    // Status filter (only for calendar view)
    // Shows all statuses without counts to avoid confusion with paginated data
    if (activeView === "calendar") {
      groups.push({
        id: "status",
        label: "Status",
        icon: <Activity className="h-4 w-4" strokeWidth={1.5} />,
        options: allStatuses.map((status) => ({
          id: status,
          label: STATUS_LABELS[status],
          checked: selectedStatuses.includes(status),
        })),
      });
    }

    // League filter
    // Shows leagues from current page without counts to avoid confusion
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

    return groups;
  }, [
    activeView,
    availableLeagues,
    selectedLeagues,
    selectedStatuses,
    selectedTimeRange,
    timeRangeLabels,
  ]);

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "timeRange" && checked) {
      // Time range is single-select (radio behavior)
      onTimeRangeChange(optionId as TimeRange);
    } else if (groupId === "status") {
      onStatusChange?.(optionId as MatchStatus, checked);
    } else if (groupId === "league") {
      onLeagueChange(optionId, checked);
    }
  };

  // Render tabs in header content slot
  const headerContent = (
    <MatchesViewTabs activeView={activeView} onViewChange={onViewChange} />
  );

  // Render date range picker for calendar view
  const quickFilterContent = activeView === "calendar" && onCustomDateRangeChange ? (
    <DateRangePicker
      value={customDateRange}
      onChange={onCustomDateRangeChange}
      mode="past" // Calendar view shows all matches (past by default)
    />
  ) : null;

  return (
    <FilterPanel
      title="Matches"
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
      headerContent={headerContent}
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
