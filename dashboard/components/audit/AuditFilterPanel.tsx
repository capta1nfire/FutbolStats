"use client";

import {
  AuditEventType,
  AuditSeverity,
  AUDIT_EVENT_TYPES,
  AUDIT_EVENT_TYPE_LABELS,
  AUDIT_SEVERITIES,
  AUDIT_SEVERITY_LABELS,
} from "@/lib/types";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { FileText, AlertTriangle } from "lucide-react";

interface AuditFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedTypes: AuditEventType[];
  selectedSeverities: AuditSeverity[];
  searchValue: string;
  onTypeChange: (type: AuditEventType, checked: boolean) => void;
  onSeverityChange: (severity: AuditSeverity, checked: boolean) => void;
  onSearchChange: (value: string) => void;
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

export function AuditFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedTypes,
  selectedSeverities,
  searchValue,
  onTypeChange,
  onSeverityChange,
  onSearchChange,
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
