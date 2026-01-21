"use client";

import {
  AuditEventType,
  AuditSeverity,
  AuditActorKind,
  AuditTimeRange,
  AUDIT_EVENT_TYPES,
  AUDIT_EVENT_TYPE_LABELS,
  AUDIT_SEVERITIES,
  AUDIT_SEVERITY_LABELS,
  AUDIT_ACTOR_KINDS,
  AUDIT_ACTOR_KIND_LABELS,
  AUDIT_TIME_RANGES,
  AUDIT_TIME_RANGE_LABELS,
} from "@/lib/types";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { FileText, AlertTriangle, User, Clock } from "lucide-react";

interface AuditFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedTypes: AuditEventType[];
  selectedSeverities: AuditSeverity[];
  selectedActorKinds: AuditActorKind[];
  selectedTimeRange: AuditTimeRange | null;
  searchValue: string;
  onTypeChange: (type: AuditEventType, checked: boolean) => void;
  onSeverityChange: (severity: AuditSeverity, checked: boolean) => void;
  onActorKindChange: (actorKind: AuditActorKind, checked: boolean) => void;
  onTimeRangeChange: (timeRange: AuditTimeRange | null) => void;
  onSearchChange: (value: string) => void;
  /** Whether to show the Customize Columns link in footer */
  showCustomizeColumns?: boolean;
  /** Callback for "Customize Columns" link click */
  onCustomizeColumnsClick?: () => void;
  /** Whether CustomizeColumnsPanel is currently open */
  customizeColumnsOpen?: boolean;
}

const typeOptions: { id: AuditEventType; label: string }[] =
  AUDIT_EVENT_TYPES.map((type) => ({
    id: type,
    label: AUDIT_EVENT_TYPE_LABELS[type],
  }));

const severityOptions: { id: AuditSeverity; label: string }[] =
  AUDIT_SEVERITIES.map((severity) => ({
    id: severity,
    label: AUDIT_SEVERITY_LABELS[severity],
  }));

const actorKindOptions: { id: AuditActorKind; label: string }[] =
  AUDIT_ACTOR_KINDS.map((kind) => ({
    id: kind,
    label: AUDIT_ACTOR_KIND_LABELS[kind],
  }));

const timeRangeOptions: { id: AuditTimeRange; label: string }[] =
  AUDIT_TIME_RANGES.map((range) => ({
    id: range,
    label: AUDIT_TIME_RANGE_LABELS[range],
  }));

export function AuditFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedTypes,
  selectedSeverities,
  selectedActorKinds,
  selectedTimeRange,
  searchValue,
  onTypeChange,
  onSeverityChange,
  onActorKindChange,
  onTimeRangeChange,
  onSearchChange,
  showCustomizeColumns = false,
  onCustomizeColumnsClick,
  customizeColumnsOpen = false,
}: AuditFilterPanelProps) {
  const filterGroups: FilterGroup[] = [
    {
      id: "type",
      label: "Event Type",
      icon: <FileText className="h-4 w-4" />,
      options: typeOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedTypes.includes(opt.id),
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
      id: "actorKind",
      label: "Actor",
      icon: <User className="h-4 w-4" />,
      options: actorKindOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedActorKinds.includes(opt.id),
      })),
    },
    {
      id: "timeRange",
      label: "Time Range",
      icon: <Clock className="h-4 w-4" />,
      options: timeRangeOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedTimeRange === opt.id,
      })),
    },
  ];

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "type") {
      onTypeChange(optionId as AuditEventType, checked);
    } else if (groupId === "severity") {
      onSeverityChange(optionId as AuditSeverity, checked);
    } else if (groupId === "actorKind") {
      onActorKindChange(optionId as AuditActorKind, checked);
    } else if (groupId === "timeRange") {
      // Radio-like behavior: toggle on/off, only one selected at a time
      if (checked) {
        onTimeRangeChange(optionId as AuditTimeRange);
      } else {
        onTimeRangeChange(null);
      }
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
