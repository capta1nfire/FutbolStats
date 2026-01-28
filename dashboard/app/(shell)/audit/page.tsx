"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuditEventsApi, useAuditEventDetail, useColumnVisibility, usePageSize } from "@/lib/hooks";
import {
  AuditEventRow,
  AuditEventType,
  AuditSeverity,
  AuditActorKind,
  AuditTimeRange,
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
import { Database } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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

  // Fetch data from real API with mock fallback
  const {
    events,
    pagination,
    isDegraded,
    isLoading,
    error,
    refetch,
  } = useAuditEventsApi({
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    severity: selectedSeverities.length > 0 ? selectedSeverities : undefined,
    actorKind: selectedActorKinds.length > 0 ? selectedActorKinds : undefined,
    q: searchValue || undefined,
    range: selectedTimeRange || undefined,
    page: currentPage,
    limit: pageSize,
  });

  // Find selected event from the events list
  const selectedEventRow = useMemo(
    () => events.find((e) => e.id === selectedEventId) ?? null,
    [events, selectedEventId]
  );

  // Get detail from row (audit events don't have separate detail endpoint)
  const {
    data: selectedEvent,
    isLoading: isLoadingDetail,
  } = useAuditEventDetail(selectedEventRow);

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

  // Handle filter changes - reset to page 1 when filters change
  const handleTypeChange = useCallback(
    (type: AuditEventType, checked: boolean) => {
      const newTypes = toggleArrayValue(selectedTypes, type, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ type: newTypes }), { scroll: false });
    },
    [selectedTypes, router, buildUrl]
  );

  const handleSeverityChange = useCallback(
    (severity: AuditSeverity, checked: boolean) => {
      const newSeverities = toggleArrayValue(selectedSeverities, severity, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ severity: newSeverities }), { scroll: false });
    },
    [selectedSeverities, router, buildUrl]
  );

  const handleActorKindChange = useCallback(
    (actorKind: AuditActorKind, checked: boolean) => {
      const newActorKinds = toggleArrayValue(selectedActorKinds, actorKind, checked);
      setCurrentPage(1);
      router.replace(buildUrl({ actor: newActorKinds }), { scroll: false });
    },
    [selectedActorKinds, router, buildUrl]
  );

  const handleTimeRangeChange = useCallback(
    (timeRange: AuditTimeRange | null) => {
      setCurrentPage(1);
      router.replace(buildUrl({ range: timeRange }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      setCurrentPage(1);
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle page change
  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page);
  }, []);

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
        {/* Header with mock indicator */}
        <div className="h-12 flex items-center justify-between px-6 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Audit</h1>
          {isDegraded && !isLoading && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)]">
                    <Database className="h-3.5 w-3.5 text-[var(--status-warning-text)]" />
                    <span className="text-[10px] text-[var(--status-warning-text)] font-medium">
                      mock
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <p>Using mock data - backend unavailable</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
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

        {/* Pagination - using real total from backend */}
        <Pagination
          currentPage={currentPage}
          totalItems={pagination.total}
          pageSize={pageSize}
          onPageChange={handlePageChange}
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
