export { SotaFilterPanel } from "./SotaFilterPanel";
export type { SotaStatusFilter, SotaSourceFilter } from "./SotaFilterPanel";
export { SotaViewTabs } from "./SotaViewTabs";
export type { SotaView } from "./SotaViewTabs";
export { FeatureCoverageMatrix } from "./FeatureCoverageMatrix";
export type { CoverageRangeFilter } from "./FeatureCoverageMatrix";
export { FeatureCoverageDetailDrawer } from "./FeatureCoverageDetailDrawer";
export type { FeatureCoverageLeague } from "@/lib/hooks/use-feature-coverage";
export { LeagueFilterPanel } from "./LeagueFilterPanel";
export type { LeagueOption } from "./LeagueFilterPanel";
export {
  SOTA_STATUS_FILTERS,
  SOTA_SOURCE_FILTERS,
  SOTA_STATUS_LABELS,
  SOTA_SOURCE_LABELS,
  COVERAGE_RANGE_FILTERS,
  COVERAGE_RANGE_LABELS,
} from "./SotaFilterPanel";

// Column options for CustomizeColumnsPanel
import { ColumnOption } from "@/components/tables";

export const SOTA_COLUMN_OPTIONS: ColumnOption[] = [
  { id: "status", label: "Status", enableHiding: false },
  { id: "name", label: "Name", enableHiding: false },
  { id: "category", label: "Category", enableHiding: true },
  { id: "current", label: "Current Value", enableHiding: true },
  { id: "threshold", label: "Threshold", enableHiding: true },
  { id: "affected", label: "Affected", enableHiding: true },
  { id: "lastRun", label: "Last Run", enableHiding: true },
];

export const SOTA_DEFAULT_VISIBILITY: Record<string, boolean> = {
  status: true,
  name: true,
  category: false, // Always coverage, no need to show
  current: true,
  threshold: true,
  affected: true,
  lastRun: true,
};
