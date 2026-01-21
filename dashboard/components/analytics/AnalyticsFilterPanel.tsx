"use client";

import {
  AnalyticsReportType,
  ANALYTICS_REPORT_TYPES,
  ANALYTICS_REPORT_TYPE_LABELS,
} from "@/lib/types";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { BarChart3 } from "lucide-react";

interface AnalyticsFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedTypes: AnalyticsReportType[];
  searchValue: string;
  onTypeChange: (type: AnalyticsReportType, checked: boolean) => void;
  onSearchChange: (value: string) => void;
}

const typeOptions: { id: AnalyticsReportType; label: string }[] =
  ANALYTICS_REPORT_TYPES.map((type) => ({
    id: type,
    label: ANALYTICS_REPORT_TYPE_LABELS[type],
  }));

export function AnalyticsFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedTypes,
  searchValue,
  onTypeChange,
  onSearchChange,
}: AnalyticsFilterPanelProps) {
  const filterGroups: FilterGroup[] = [
    {
      id: "type",
      label: "Report Type",
      icon: <BarChart3 className="h-4 w-4" />,
      options: typeOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedTypes.includes(opt.id),
      })),
    },
  ];

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "type") {
      onTypeChange(optionId as AnalyticsReportType, checked);
    }
  };

  return (
    <FilterPanel
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
      groups={filterGroups}
      onFilterChange={handleFilterChange}
      onSearchChange={onSearchChange}
      searchValue={searchValue}
    />
  );
}
