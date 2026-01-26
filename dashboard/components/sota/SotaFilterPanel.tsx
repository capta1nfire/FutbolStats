"use client";

import { ReactNode } from "react";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { Activity, Database, Layers, BarChart3 } from "lucide-react";
import { CoverageRangeFilter } from "./FeatureCoverageMatrix";

export type SotaStatusFilter = "ok" | "warning" | "critical";
export type SotaSourceFilter = "understat" | "weather" | "sofascore" | "venue_geo" | "team_profile";
export type SotaTierId = "tier1" | "tier1b" | "tier1c" | "tier1d";

export const SOTA_STATUS_FILTERS: SotaStatusFilter[] = ["ok", "warning", "critical"];
export const SOTA_SOURCE_FILTERS: SotaSourceFilter[] = ["understat", "weather", "sofascore", "venue_geo", "team_profile"];
export const SOTA_TIER_IDS: SotaTierId[] = ["tier1", "tier1b", "tier1c", "tier1d"];

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

export const SOTA_TIER_LABELS: Record<SotaTierId, string> = {
  tier1: "Core (PROD)",
  tier1b: "xG (TITAN)",
  tier1c: "Lineup (TITAN)",
  tier1d: "XI Depth (TITAN)",
};

export const COVERAGE_RANGE_FILTERS: CoverageRangeFilter[] = ["all", "100", "70-99", "50-69", "below50"];

export const COVERAGE_RANGE_LABELS: Record<CoverageRangeFilter, string> = {
  all: "All",
  "100": "100%",
  "70-99": "70% - 99%",
  "50-69": "50% - 69%",
  below50: "< 50%",
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
  /** Current view: "enrichment" or "features" */
  activeView?: "enrichment" | "features";
  /** Enabled tiers for Features view */
  enabledTiers?: Set<string>;
  /** Callback for tier toggle */
  onTierChange?: (tierId: string, checked: boolean) => void;
  /** Selected coverage range filter for Features view */
  coverageRangeFilter?: CoverageRangeFilter;
  /** Callback for coverage range filter change */
  onCoverageRangeChange?: (range: CoverageRangeFilter) => void;
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

const tierOptions: { id: SotaTierId; label: string }[] =
  SOTA_TIER_IDS.map((tier) => ({
    id: tier,
    label: SOTA_TIER_LABELS[tier],
  }));

const coverageRangeOptions: { id: CoverageRangeFilter; label: string }[] =
  COVERAGE_RANGE_FILTERS.map((range) => ({
    id: range,
    label: COVERAGE_RANGE_LABELS[range],
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
  activeView = "enrichment",
  enabledTiers,
  onTierChange,
  coverageRangeFilter = "all",
  onCoverageRangeChange,
}: SotaFilterPanelProps) {
  // Build filter groups based on active view
  const filterGroups: FilterGroup[] = activeView === "enrichment"
    ? [
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
      ]
    : [
        {
          id: "tier",
          label: "Tiers",
          icon: <Layers className="h-4 w-4" strokeWidth={1.5} />,
          options: tierOptions.map((opt) => ({
            id: opt.id,
            label: opt.label,
            checked: enabledTiers?.has(opt.id) ?? true,
          })),
        },
        {
          id: "coverage",
          label: "Coverage",
          icon: <BarChart3 className="h-4 w-4" strokeWidth={1.5} />,
          options: coverageRangeOptions.map((opt) => ({
            id: opt.id,
            label: opt.label,
            checked: coverageRangeFilter === opt.id,
          })),
          singleSelect: true,
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
    } else if (groupId === "tier" && onTierChange) {
      onTierChange(optionId, checked);
    } else if (groupId === "coverage" && onCoverageRangeChange && checked) {
      // Single select - only trigger when checked
      onCoverageRangeChange(optionId as CoverageRangeFilter);
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
