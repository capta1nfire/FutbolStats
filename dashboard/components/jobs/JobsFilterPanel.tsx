"use client";

import { JobStatus, JOB_NAMES } from "@/lib/types";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { Activity, Cog } from "lucide-react";

interface JobsFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedStatuses: JobStatus[];
  selectedJobs: string[];
  searchValue: string;
  onStatusChange: (status: JobStatus, checked: boolean) => void;
  onJobChange: (job: string, checked: boolean) => void;
  onSearchChange: (value: string) => void;
}

const statusOptions: { id: JobStatus; label: string }[] = [
  { id: "running", label: "Running" },
  { id: "success", label: "Success" },
  { id: "failed", label: "Failed" },
  { id: "pending", label: "Pending" },
];

const jobOptions = JOB_NAMES.map((name) => ({
  id: name,
  label: name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" "),
}));

export function JobsFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedStatuses,
  selectedJobs,
  searchValue,
  onStatusChange,
  onJobChange,
  onSearchChange,
}: JobsFilterPanelProps) {
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
      id: "job",
      label: "Job Type",
      icon: <Cog className="h-4 w-4" />,
      options: jobOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedJobs.includes(opt.id),
      })),
    },
  ];

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "status") {
      onStatusChange(optionId as JobStatus, checked);
    } else if (groupId === "job") {
      onJobChange(optionId, checked);
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
