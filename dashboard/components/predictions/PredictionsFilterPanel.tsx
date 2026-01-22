"use client";

import {
  PredictionStatus,
  ModelType,
  PredictionTimeRange,
  PREDICTION_STATUSES,
  PREDICTION_STATUS_LABELS,
  MODEL_TYPES,
  MODEL_TYPE_LABELS,
  PREDICTION_TIME_RANGES,
  PREDICTION_TIME_RANGE_LABELS,
} from "@/lib/types";
import { FilterPanel, FilterGroup } from "@/components/shell";
import { TrendingUp, Cpu, Clock, MapPin } from "lucide-react";

interface PredictionsFilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  selectedStatuses: PredictionStatus[];
  selectedModels: ModelType[];
  selectedLeagues: string[];
  selectedTimeRange: PredictionTimeRange | null;
  searchValue: string;
  availableLeagues: string[];
  onStatusChange: (status: PredictionStatus, checked: boolean) => void;
  onModelChange: (model: ModelType, checked: boolean) => void;
  onLeagueChange: (league: string, checked: boolean) => void;
  onTimeRangeChange: (timeRange: PredictionTimeRange | null) => void;
  onSearchChange: (value: string) => void;
  /** Whether to show the Customize Columns link in footer */
  showCustomizeColumns?: boolean;
  /** Callback for "Customize Columns" link click */
  onCustomizeColumnsClick?: () => void;
  /** Whether CustomizeColumnsPanel is currently open */
  customizeColumnsOpen?: boolean;
}

const statusOptions = PREDICTION_STATUSES.map((status) => ({
  id: status,
  label: PREDICTION_STATUS_LABELS[status],
}));

const modelOptions = MODEL_TYPES.map((model) => ({
  id: model,
  label: MODEL_TYPE_LABELS[model],
}));

const timeRangeOptions = PREDICTION_TIME_RANGES.map((range) => ({
  id: range,
  label: PREDICTION_TIME_RANGE_LABELS[range],
}));

export function PredictionsFilterPanel({
  collapsed,
  onToggleCollapse,
  selectedStatuses,
  selectedModels,
  selectedLeagues,
  selectedTimeRange,
  searchValue,
  availableLeagues,
  onStatusChange,
  onModelChange,
  onLeagueChange,
  onTimeRangeChange,
  onSearchChange,
  showCustomizeColumns = false,
  onCustomizeColumnsClick,
  customizeColumnsOpen = false,
}: PredictionsFilterPanelProps) {
  const leagueOptions = availableLeagues.map((league) => ({
    id: league,
    label: league,
  }));

  const filterGroups: FilterGroup[] = [
    {
      id: "status",
      label: "Status",
      icon: <TrendingUp className="h-4 w-4" strokeWidth={1.5} />,
      options: statusOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedStatuses.includes(opt.id),
      })),
    },
    {
      id: "model",
      label: "Model",
      icon: <Cpu className="h-4 w-4" strokeWidth={1.5} />,
      options: modelOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedModels.includes(opt.id),
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
    {
      id: "league",
      label: "League",
      icon: <MapPin className="h-4 w-4" strokeWidth={1.5} />,
      options: leagueOptions.map((opt) => ({
        id: opt.id,
        label: opt.label,
        checked: selectedLeagues.includes(opt.id),
      })),
    },
  ];

  const handleFilterChange = (
    groupId: string,
    optionId: string,
    checked: boolean
  ) => {
    if (groupId === "status") {
      onStatusChange(optionId as PredictionStatus, checked);
    } else if (groupId === "model") {
      onModelChange(optionId as ModelType, checked);
    } else if (groupId === "league") {
      onLeagueChange(optionId, checked);
    } else if (groupId === "timeRange") {
      // Radio-like behavior
      if (checked) {
        onTimeRangeChange(optionId as PredictionTimeRange);
      } else {
        onTimeRangeChange(null);
      }
    }
  };

  return (
    <FilterPanel
      title="Predictions"
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
