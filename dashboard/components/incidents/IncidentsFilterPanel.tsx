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
import { AlertTriangle, Activity, Tag } from "lucide-react";

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
}

const statusOptions: { id: IncidentStatus; label: string }[] = INCIDENT_STATUSES.map(
  (status) => ({
    id: status,
    label: status.charAt(0).toUpperCase() + status.slice(1),
  })
);

const severityOptions: { id: IncidentSeverity; label: string }[] =
  INCIDENT_SEVERITIES.map((severity) => ({
    id: severity,
    label: severity.charAt(0).toUpperCase() + severity.slice(1),
  }));

const typeOptions: { id: IncidentType; label: string }[] = INCIDENT_TYPES.map(
  (type) => ({
    id: type,
    label: INCIDENT_TYPE_LABELS[type],
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
}: IncidentsFilterPanelProps) {
  const filterGroups: FilterGroup[] = [
    {
      id: "status",
      label: "Status",
      icon: <Activity className="h-4 w-4" />,
      options: statusOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedStatuses.includes(opt.id),
      })),
    },
    {
      id: "severity",
      label: "Severity",
      icon: <AlertTriangle className="h-4 w-4" />,
      options: severityOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedSeverities.includes(opt.id),
      })),
    },
    {
      id: "type",
      label: "Type",
      icon: <Tag className="h-4 w-4" />,
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
    } else if (groupId === "severity") {
      onSeverityChange(optionId as IncidentSeverity, checked);
    } else if (groupId === "type") {
      onTypeChange(optionId as IncidentType, checked);
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
