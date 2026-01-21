"use client";

import {
  DataQualityStatus,
  DataQualityCategory,
  DATA_QUALITY_STATUSES,
  DATA_QUALITY_CATEGORIES,
  DATA_QUALITY_CATEGORY_LABELS,
} from "@/lib/types";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { CheckCircle2, FolderOpen } from "lucide-react";

interface DataQualityFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedStatuses: DataQualityStatus[];
  selectedCategories: DataQualityCategory[];
  searchValue: string;
  onStatusChange: (status: DataQualityStatus, checked: boolean) => void;
  onCategoryChange: (category: DataQualityCategory, checked: boolean) => void;
  onSearchChange: (value: string) => void;
  /** Whether to show the Customize Columns link in footer */
  showCustomizeColumns?: boolean;
  /** Callback for "Customize Columns" link click */
  onCustomizeColumnsClick?: () => void;
  /** Whether CustomizeColumnsPanel is currently open */
  customizeColumnsOpen?: boolean;
}

const statusOptions: { id: DataQualityStatus; label: string }[] = DATA_QUALITY_STATUSES.map(
  (status) => ({
    id: status,
    label: status.charAt(0).toUpperCase() + status.slice(1),
  })
);

const categoryOptions: { id: DataQualityCategory; label: string }[] =
  DATA_QUALITY_CATEGORIES.map((category) => ({
    id: category,
    label: DATA_QUALITY_CATEGORY_LABELS[category],
  }));

export function DataQualityFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedStatuses,
  selectedCategories,
  searchValue,
  onStatusChange,
  onCategoryChange,
  onSearchChange,
  showCustomizeColumns = false,
  onCustomizeColumnsClick,
  customizeColumnsOpen = false,
}: DataQualityFilterPanelProps) {
  const filterGroups: FilterGroup[] = [
    {
      id: "status",
      label: "Status",
      icon: <CheckCircle2 className="h-4 w-4" />,
      options: statusOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedStatuses.includes(opt.id),
      })),
    },
    {
      id: "category",
      label: "Category",
      icon: <FolderOpen className="h-4 w-4" />,
      options: categoryOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedCategories.includes(opt.id),
      })),
    },
  ];

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "status") {
      onStatusChange(optionId as DataQualityStatus, checked);
    } else if (groupId === "category") {
      onCategoryChange(optionId as DataQualityCategory, checked);
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
      showCustomizeColumns={showCustomizeColumns}
      onCustomizeColumnsClick={onCustomizeColumnsClick}
      customizeColumnsOpen={customizeColumnsOpen}
    />
  );
}
