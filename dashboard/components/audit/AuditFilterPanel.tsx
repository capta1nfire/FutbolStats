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
import {
  SeverityQuickFilter,
  SeverityFilterItem,
} from "@/components/ui/severity-quick-filter";
import { SeverityLevel } from "@/components/ui/severity-bars";
import { FileText, User, Clock } from "lucide-react";

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
  /** Optional counts per severity for quick filter badges */
  severityCounts?: Record<AuditSeverity, number>;
}

const typeOptions: { id: AuditEventType; label: string }[] =
  AUDIT_EVENT_TYPES.map((type) => ({
    id: type,
    label: AUDIT_EVENT_TYPE_LABELS[type],
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

/** Map AuditSeverity to SeverityLevel for bars */
const severityToLevel: Record<AuditSeverity, SeverityLevel> = {
  info: 1,
  warning: 2,
  error: 4,
};

/** Severity items for quick filter (Audit has: info, warning, error - no "high"/level 3) */
const severityFilterItems: SeverityFilterItem[] = AUDIT_SEVERITIES.map(
  (severity) => ({
    level: severityToLevel[severity],
    id: severity,
    label: AUDIT_SEVERITY_LABELS[severity],
  })
);

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
  severityCounts,
}: AuditFilterPanelProps) {
  // Build filter groups (excluding severity - now handled by quick filter)
  const filterGroups: FilterGroup[] = [
    {
      id: "type",
      label: "Event Type",
      icon: <FileText className="h-4 w-4" strokeWidth={1.5} />,
      options: typeOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedTypes.includes(opt.id),
      })),
    },
    {
      id: "actorKind",
      label: "Actor",
      icon: <User className="h-4 w-4" strokeWidth={1.5} />,
      options: actorKindOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedActorKinds.includes(opt.id),
      })),
    },
    {
      id: "timeRange",
      label: "Time Range",
      icon: <Clock className="h-4 w-4" strokeWidth={1.5} />,
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

  const handleSeverityToggle = (severityId: string) => {
    const severity = severityId as AuditSeverity;
    const isSelected = selectedSeverities.includes(severity);
    onSeverityChange(severity, !isSelected);
  };

  // Build severity items with optional counts
  const severityItemsWithCounts = severityFilterItems.map((item) => ({
    ...item,
    count: severityCounts?.[item.id as AuditSeverity],
  }));

  return (
    <FilterPanel
      title="Audit"
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
