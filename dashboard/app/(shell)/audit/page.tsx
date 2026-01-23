"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useOpsLogsApi, useOpsLogDetail, useColumnVisibility, usePageSize } from "@/lib/hooks";
import {
  AuditEventRow,
  AuditEventType,
  AuditSeverity,
  AuditActorKind,
  AuditTimeRange,
  AuditFilters,
  AUDIT_EVENT_TYPES,
  AUDIT_SEVERITIES,
  AUDIT_ACTOR_KINDS,
  AUDIT_TIME_RANGES,
} from "@/lib/types";
import {
  AuditTable,
  AuditFilterPanel,
  AuditDetailDrawer,
  AUDIT_COLUMN_OPTIONS,
  AUDIT_DEFAULT_VISIBILITY,
} from "@/components/audit";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseNumericId,
  parseArrayParam,
  parseSingleParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";

const BASE_PATH = "/audit";

/**
 * Audit Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function AuditPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const selectedEventId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );
  const selectedTypes = useMemo(
    () => parseArrayParam<AuditEventType>(searchParams, "type", AUDIT_EVENT_TYPES),
    [searchParams]
  );
  const selectedSeverities = useMemo(
    () => parseArrayParam<AuditSeverity>(searchParams, "severity", AUDIT_SEVERITIES),
    [searchParams]
  );
  const selectedActorKinds = useMemo(
    () => parseArrayParam<AuditActorKind>(searchParams, "actor", AUDIT_ACTOR_KINDS),
    [searchParams]
  );
  const selectedTimeRange = useMemo(
    () => parseSingleParam<AuditTimeRange>(searchParams.get("range"), AUDIT_TIME_RANGES),
    [searchParams]
  );
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

  // Normalize URL if id param is invalid
  const selectedIdParam = searchParams.get("id");
  useEffect(() => {
    if (selectedIdParam && selectedEventId === null) {
      const params = new URLSearchParams(searchParams.toString());
      params.delete("id");
      const search = params.toString();
      router.replace(`${BASE_PATH}${search ? `?${search}` : ""}`, { scroll: false });
    }
  }, [selectedIdParam, selectedEventId, router, searchParams]);

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("audit");

  // Column visibility with localStorage persistence
  const { columnVisibility, setColumnVisibility, setColumnVisible, resetToDefault } = useColumnVisibility(
    "audit",
    AUDIT_DEFAULT_VISIBILITY
  );

  // Handlers for Customize Columns
  const handleCustomizeColumnsClick = useCallback(() => {
    setCustomizeColumnsOpen(true);
  }, []);

  // Done collapses entire Left Rail (UniFi behavior)
  const handleCustomizeColumnsDone = useCallback(() => {
    setLeftRailCollapsed(true);
    setCustomizeColumnsOpen(false);
  }, []);

  const handleLeftRailToggle = useCallback(() => {
    setLeftRailCollapsed((prev) => !prev);
    if (!leftRailCollapsed) {
      setCustomizeColumnsOpen(false);
    }
  }, [leftRailCollapsed]);

  // Construct filters for query
  const filters: AuditFilters = useMemo(() => ({
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    severity: selectedSeverities.length > 0 ? selectedSeverities : undefined,
    actorKind: selectedActorKinds.length > 0 ? selectedActorKinds : undefined,
    timeRange: selectedTimeRange || undefined,
    search: searchValue || undefined,
  }), [selectedTypes, selectedSeverities, selectedActorKinds, selectedTimeRange, searchValue]);

  // Fetch data from real API with mock fallback
  const {
    data: events = [],
    metadata,
    isLoading,
    error,
    isApiDegraded,
    refetch,
  } = useOpsLogsApi(filters, 200);

  // Find selected event from the events list
  const selectedEventRow = useMemo(
    () => events.find((e) => e.id === selectedEventId) ?? null,
    [events, selectedEventId]
  );

  // Get detail from row (ops logs don't have separate detail endpoint)
  const {
    data: selectedEvent,
    isLoading: isLoadingDetail,
  } = useOpsLogDetail(selectedEventRow);

  // Drawer is open when there's a selected event
  const drawerOpen = selectedEventId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
      type?: AuditEventType[];
      severity?: AuditSeverity[];
      actor?: AuditActorKind[];
      range?: AuditTimeRange | null;
      q?: string;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedEventId : overrides.id,
        type: overrides.type ?? selectedTypes,
        severity: overrides.severity ?? selectedSeverities,
        actor: overrides.actor ?? selectedActorKinds,
        range: overrides.range === undefined ? selectedTimeRange : overrides.range,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedEventId, selectedTypes, selectedSeverities, selectedActorKinds, selectedTimeRange, searchValue]
  );

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (event: AuditEventRow) => {
      router.replace(buildUrl({ id: event.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL, preserve filters
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle filter changes
  const handleTypeChange = useCallback(
    (type: AuditEventType, checked: boolean) => {
      const newTypes = toggleArrayValue(selectedTypes, type, checked);
      router.replace(buildUrl({ type: newTypes }), { scroll: false });
    },
    [selectedTypes, router, buildUrl]
  );

  const handleSeverityChange = useCallback(
    (severity: AuditSeverity, checked: boolean) => {
      const newSeverities = toggleArrayValue(selectedSeverities, severity, checked);
      router.replace(buildUrl({ severity: newSeverities }), { scroll: false });
    },
    [selectedSeverities, router, buildUrl]
  );

  const handleActorKindChange = useCallback(
    (actorKind: AuditActorKind, checked: boolean) => {
      const newActorKinds = toggleArrayValue(selectedActorKinds, actorKind, checked);
      router.replace(buildUrl({ actor: newActorKinds }), { scroll: false });
    },
    [selectedActorKinds, router, buildUrl]
  );

  const handleTimeRangeChange = useCallback(
    (timeRange: AuditTimeRange | null) => {
      router.replace(buildUrl({ range: timeRange }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* FilterPanel */}
      <AuditFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        selectedTypes={selectedTypes}
        selectedSeverities={selectedSeverities}
        selectedActorKinds={selectedActorKinds}
        selectedTimeRange={selectedTimeRange}
        searchValue={searchValue}
        onTypeChange={handleTypeChange}
        onSeverityChange={handleSeverityChange}
        onActorKindChange={handleActorKindChange}
        onTimeRangeChange={handleTimeRangeChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
      />

      {/* Customize Columns Panel */}
      <CustomizeColumnsPanel
        open={customizeColumnsOpen && !leftRailCollapsed}
        columns={AUDIT_COLUMN_OPTIONS}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={setColumnVisible}
        onRestore={resetToDefault}
        onDone={handleCustomizeColumnsDone}
        onCollapse={handleLeftRailToggle}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Ops Logs Header */}
        <div className="px-4 py-2 border-b border-border flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            Ops Logs (from backend)
          </span>
          {isApiDegraded && (
            <span className="text-xs text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded">
              degraded - using mock data
            </span>
          )}
          {metadata && !isApiDegraded && (
            <span className="text-xs text-muted-foreground">
              • Last {metadata.sinceMinutes} min • Limit {metadata.limit}
            </span>
          )}
        </div>

        {/* Table */}
        <AuditTable
          data={events}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedEventId={selectedEventId}
          onRowClick={handleRowClick}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisibility}
        />

        {/* Pagination */}
        <Pagination
          currentPage={currentPage}
          totalItems={events.length}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
        />
      </div>

      {/* Detail Drawer (inline on desktop, sheet on mobile) */}
      <AuditDetailDrawer
        event={selectedEvent ?? null}
        open={drawerOpen}
        onClose={handleCloseDrawer}
        isLoading={isLoadingDetail}
      />
    </div>
  );
}

/**
 * Loading fallback for Suspense
 */
function AuditLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Audit Page
 *
 * Master-detail pattern with URL sync (full state):
 * - Canonical: /audit?id=123&type=job_run&severity=error&actor=system&range=24h&q=sync
 * - Uses router.replace with scroll:false
 */
export default function AuditPage() {
  return (
    <Suspense fallback={<AuditLoading />}>
      <AuditPageContent />
    </Suspense>
  );
}
