"use client";

import { ReactNode } from "react";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { Activity, Database } from "lucide-react";

export type SotaStatusFilter = "ok" | "warning" | "critical";
export type SotaSourceFilter = "understat" | "weather" | "sofascore" | "venue_geo" | "team_profile";

export const SOTA_STATUS_FILTERS: SotaStatusFilter[] = ["ok", "warning", "critical"];
export const SOTA_SOURCE_FILTERS: SotaSourceFilter[] = ["understat", "weather", "sofascore", "venue_geo", "team_profile"];

export const SOTA_STATUS_LABELS: Record<SotaStatusFilter, string> = {
  ok: "OK",
  warning: "Warning",
  critical: "Critical",
};

export const SOTA_SOURCE_LABELS: Record<SotaSourceFilter, string> = {
  understat: "Understat",
  weather: "Weather API",
  sofascore: "Sofascore",
  venue_geo: "Venue Geo",
  team_profile: "Team Profiles",
};

interface SotaFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedStatuses: SotaStatusFilter[];
  selectedSources: SotaSourceFilter[];
  searchValue: string;
  onStatusChange: (status: SotaStatusFilter, checked: boolean) => void;
  onSourceChange: (source: SotaSourceFilter, checked: boolean) => void;
  onSearchChange: (value: string) => void;
  /** Whether to show the Customize Columns link in footer */
  showCustomizeColumns?: boolean;
  /** Callback for "Customize Columns" link click */
  onCustomizeColumnsClick?: () => void;
  /** Whether CustomizeColumnsPanel is currently open */
  customizeColumnsOpen?: boolean;
  /** Optional header content (e.g., view tabs) */
  headerContent?: ReactNode;
}

const statusOptions: { id: SotaStatusFilter; label: string }[] =
  SOTA_STATUS_FILTERS.map((status) => ({
    id: status,
    label: SOTA_STATUS_LABELS[status],
  }));

const sourceOptions: { id: SotaSourceFilter; label: string }[] =
  SOTA_SOURCE_FILTERS.map((source) => ({
    id: source,
    label: SOTA_SOURCE_LABELS[source],
  }));

export function SotaFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedStatuses,
  selectedSources,
  searchValue,
  onStatusChange,
  onSourceChange,
  onSearchChange,
  showCustomizeColumns = false,
  onCustomizeColumnsClick,
  customizeColumnsOpen = false,
  headerContent,
}: SotaFilterPanelProps) {
  const filterGroups: FilterGroup[] = [
    {
      id: "status",
      label: "Status",
      icon: <Activity className="h-4 w-4" strokeWidth={1.5} />,
      options: statusOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedStatuses.includes(opt.id),
      })),
    },
    {
      id: "source",
      label: "Source",
      icon: <Database className="h-4 w-4" strokeWidth={1.5} />,
      options: sourceOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedSources.includes(opt.id),
      })),
    },
  ];

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "status") {
      onStatusChange(optionId as SotaStatusFilter, checked);
    } else if (groupId === "source") {
      onSourceChange(optionId as SotaSourceFilter, checked);
    }
  };

  return (
    <FilterPanel
      title="SOTA"
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
      headerContent={headerContent}
      groups={filterGroups}
      onFilterChange={handleFilterChange}
      onSearchChange={onSearchChange}
      searchValue={searchValue}
      showCustomizeColumns={showCustomizeColumns}
      onCustomizeColumnsClick={onCustomizeColumnsClick}
      customizeColumnsOpen={customizeColumnsOpen}
    />
  );
}
