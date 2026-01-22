"use client";

import {
  IncidentStatus,
  IncidentSeverity,
  IncidentType,
  INCIDENT_STATUSES,
  INCIDENT_SEVERITIES,
  INCIDENT_TYPES,
  INCIDENT_TYPE_LABELS,
} from "@/lib/types";
import { FilterPanel, FilterGroup } from "@/components/shell";
import {
  SeverityQuickFilter,
  SeverityFilterItem,
} from "@/components/ui/severity-quick-filter";
import { SeverityLevel } from "@/components/ui/severity-bars";
import { Activity, Tag } from "lucide-react";

interface IncidentsFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedStatuses: IncidentStatus[];
  selectedSeverities: IncidentSeverity[];
  selectedTypes: IncidentType[];
  searchValue: string;
  onStatusChange: (status: IncidentStatus, checked: boolean) => void;
  onSeverityChange: (severity: IncidentSeverity, checked: boolean) => void;
  onTypeChange: (type: IncidentType, checked: boolean) => void;
  onSearchChange: (value: string) => void;
  /** Whether to show the Customize Columns link in footer */
  showCustomizeColumns?: boolean;
  /** Callback for "Customize Columns" link click */
  onCustomizeColumnsClick?: () => void;
  /** Whether CustomizeColumnsPanel is currently open */
  customizeColumnsOpen?: boolean;
  /** Optional counts per severity for quick filter badges */
  severityCounts?: Record<IncidentSeverity, number>;
}

const statusOptions: { id: IncidentStatus; label: string }[] = INCIDENT_STATUSES.map(
  (status) => ({
    id: status,
    label: status.charAt(0).toUpperCase() + status.slice(1),
  })
);

const typeOptions: { id: IncidentType; label: string }[] = INCIDENT_TYPES.map(
  (type) => ({
    id: type,
    label: INCIDENT_TYPE_LABELS[type],
  })
);

/** Map IncidentSeverity to SeverityLevel for bars */
const severityToLevel: Record<IncidentSeverity, SeverityLevel> = {
  info: 1,
  warning: 2,
  critical: 4,
};

/** Severity labels for display */
const severityLabels: Record<IncidentSeverity, string> = {
  info: "Info",
  warning: "Warning",
  critical: "Critical",
};

/** Severity items for quick filter (Incidents has: info, warning, critical - no "high"/level 3) */
const severityFilterItems: SeverityFilterItem[] = INCIDENT_SEVERITIES.map(
  (severity) => ({
    level: severityToLevel[severity],
    id: severity,
    label: severityLabels[severity],
  })
);

export function IncidentsFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedStatuses,
  selectedSeverities,
  selectedTypes,
  searchValue,
  onStatusChange,
  onSeverityChange,
  onTypeChange,
  onSearchChange,
  showCustomizeColumns = false,
  onCustomizeColumnsClick,
  customizeColumnsOpen = false,
  severityCounts,
}: IncidentsFilterPanelProps) {
  // Build filter groups (excluding severity - now handled by quick filter)
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
      id: "type",
      label: "Type",
      icon: <Tag className="h-4 w-4" strokeWidth={1.5} />,
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
    if (groupId === "status") {
      onStatusChange(optionId as IncidentStatus, checked);
    } else if (groupId === "type") {
      onTypeChange(optionId as IncidentType, checked);
    }
  };

  const handleSeverityToggle = (severityId: string) => {
    const severity = severityId as IncidentSeverity;
    const isSelected = selectedSeverities.includes(severity);
    onSeverityChange(severity, !isSelected);
  };

  // Build severity items with optional counts
  const severityItemsWithCounts = severityFilterItems.map((item) => ({
    ...item,
    count: severityCounts?.[item.id as IncidentSeverity],
  }));

  return (
    <FilterPanel
      title="Incidents"
      collapsed={collapsed}
      onToggleCollapse={onToggleCollapse}
      groups={filterGroups}
      onFilterChange={handleFilterChange}
      onSearchChange={onSearchChange}
      searchValue={searchValue}
      showCustomizeColumns={showCustomizeColumns}
      onCustomizeColumnsClick={onCustomizeColumnsClick}
      customizeColumnsOpen={customizeColumnsOpen}
      quickFilterContent={
        <SeverityQuickFilter
          items={severityItemsWithCounts}
          selectedIds={selectedSeverities}
          onToggle={handleSeverityToggle}
        />
      }
    />
  );
}
