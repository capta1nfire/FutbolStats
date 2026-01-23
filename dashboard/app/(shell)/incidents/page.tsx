"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  useIncidentsApi,
  useIncident,
  useColumnVisibility,
  usePageSize,
  getIncidentsMockSync,
} from "@/lib/hooks";
import {
  Incident,
  IncidentStatus,
  IncidentSeverity,
  IncidentType,
  IncidentFilters,
  INCIDENT_STATUSES,
  INCIDENT_SEVERITIES,
  INCIDENT_TYPES,
} from "@/lib/types";
import {
  IncidentsTable,
  IncidentsFilterPanel,
  IncidentDetailDrawer,
  INCIDENTS_COLUMN_OPTIONS,
  INCIDENTS_DEFAULT_VISIBILITY,
} from "@/components/incidents";
import { CustomizeColumnsPanel, Pagination } from "@/components/tables";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader } from "@/components/ui/loader";

const BASE_PATH = "/incidents";

/**
 * Incidents Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function IncidentsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const selectedIncidentId = useMemo(
    () => parseNumericId(searchParams.get("id")),
    [searchParams]
  );
  const selectedStatuses = useMemo(
    () => parseArrayParam<IncidentStatus>(searchParams, "status", INCIDENT_STATUSES),
    [searchParams]
  );
  const selectedSeverities = useMemo(
    () => parseArrayParam<IncidentSeverity>(searchParams, "severity", INCIDENT_SEVERITIES),
    [searchParams]
  );
  const selectedTypes = useMemo(
    () => parseArrayParam<IncidentType>(searchParams, "type", INCIDENT_TYPES),
    [searchParams]
  );
  const searchValue = useMemo(
    () => searchParams.get("q") ?? "",
    [searchParams]
  );

  // Normalize URL if id param is invalid
  const selectedIdParam = searchParams.get("id");
  useEffect(() => {
    if (selectedIdParam && selectedIncidentId === null) {
      const params = new URLSearchParams(searchParams.toString());
      params.delete("id");
      const search = params.toString();
      router.replace(`${BASE_PATH}${search ? `?${search}` : ""}`, { scroll: false });
    }
  }, [selectedIdParam, selectedIncidentId, router, searchParams]);

  // UI state (non-URL)
  const [leftRailCollapsed, setLeftRailCollapsed] = useState(false);
  const [customizeColumnsOpen, setCustomizeColumnsOpen] = useState(false);

  // Pagination state with localStorage persistence
  const [currentPage, setCurrentPage] = useState(1);
  const { pageSize, setPageSize } = usePageSize("incidents");

  // Column visibility with localStorage persistence
  const {
    columnVisibility,
    setColumnVisibility,
    setColumnVisible,
    resetToDefault,
  } = useColumnVisibility("incidents", INCIDENTS_DEFAULT_VISIBILITY);

  // Construct filters for mock fallback
  const filters: IncidentFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    severity: selectedSeverities.length > 0 ? selectedSeverities : undefined,
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedSeverities, selectedTypes, searchValue]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedStatuses, selectedSeverities, selectedTypes, searchValue]);

  // Fetch data from backend API
  const {
    incidents: apiIncidents,
    pagination,
    isLoading,
    error,
    refetch,
  } = useIncidentsApi({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    severity: selectedSeverities.length > 0 ? selectedSeverities : undefined,
    // Backend uses different type values, so we don't pass frontend types directly
    // The backend will return all types and we filter client-side if needed
    q: searchValue || undefined,
    page: currentPage,
    limit: pageSize,
  });

  // Use API data if available, fallback to mocks
  const mockIncidents = useMemo(() => getIncidentsMockSync(filters), [filters]);
  const incidents = apiIncidents ?? mockIncidents;

  // Find selected incident from current list first
  const { data: mockSelectedIncident } = useIncident(selectedIncidentId);
  const selectedIncident = useMemo(() => {
    if (!selectedIncidentId) return null;
    // First try to find in current list
    const fromList = incidents.find((i) => i.id === selectedIncidentId);
    if (fromList) return fromList;
    // Fallback to mock data
    return mockSelectedIncident ?? null;
  }, [selectedIncidentId, incidents, mockSelectedIncident]);

  // Drawer is open when there's a selected incident
  const drawerOpen = selectedIncidentId !== null;

  // Build URL with current filters
  const buildUrl = useCallback(
    (overrides: {
      id?: number | null;
      status?: IncidentStatus[];
      severity?: IncidentSeverity[];
      type?: IncidentType[];
      q?: string;
    }) => {
      const params = buildSearchParams({
        id: overrides.id === undefined ? selectedIncidentId : overrides.id,
        status: overrides.status ?? selectedStatuses,
        severity: overrides.severity ?? selectedSeverities,
        type: overrides.type ?? selectedTypes,
        q: overrides.q ?? searchValue,
      });
      const search = params.toString();
      return `${BASE_PATH}${search ? `?${search}` : ""}`;
    },
    [selectedIncidentId, selectedStatuses, selectedSeverities, selectedTypes, searchValue]
  );

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (incident: Incident) => {
      router.replace(buildUrl({ id: incident.id }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle drawer close - remove id from URL, preserve filters
  const handleCloseDrawer = useCallback(() => {
    router.replace(buildUrl({ id: null }), { scroll: false });
  }, [router, buildUrl]);

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: IncidentStatus, checked: boolean) => {
      const newStatuses = toggleArrayValue(selectedStatuses, status, checked);
      router.replace(buildUrl({ status: newStatuses }), { scroll: false });
    },
    [selectedStatuses, router, buildUrl]
  );

  const handleSeverityChange = useCallback(
    (severity: IncidentSeverity, checked: boolean) => {
      const newSeverities = toggleArrayValue(selectedSeverities, severity, checked);
      router.replace(buildUrl({ severity: newSeverities }), { scroll: false });
    },
    [selectedSeverities, router, buildUrl]
  );

  const handleTypeChange = useCallback(
    (type: IncidentType, checked: boolean) => {
      const newTypes = toggleArrayValue(selectedTypes, type, checked);
      router.replace(buildUrl({ type: newTypes }), { scroll: false });
    },
    [selectedTypes, router, buildUrl]
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      router.replace(buildUrl({ q: value }), { scroll: false });
    },
    [router, buildUrl]
  );

  // Handle "Customize Columns" link click
  const handleCustomizeColumnsClick = useCallback(() => {
    setCustomizeColumnsOpen(true);
  }, []);

  // Handle "Done" button - collapses entire Left Rail (UniFi behavior)
  const handleCustomizeColumnsDone = useCallback(() => {
    setLeftRailCollapsed(true);
    setCustomizeColumnsOpen(false);
  }, []);

  // Handle Left Rail toggle - close CustomizeColumns when collapsing
  const handleLeftRailToggle = useCallback(() => {
    const newCollapsed = !leftRailCollapsed;
    setLeftRailCollapsed(newCollapsed);
    if (newCollapsed) {
      setCustomizeColumnsOpen(false);
    }
  }, [leftRailCollapsed]);

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* Left Rail: FilterPanel */}
      <IncidentsFilterPanel
        collapsed={leftRailCollapsed}
        onToggleCollapse={handleLeftRailToggle}
        selectedStatuses={selectedStatuses}
        selectedSeverities={selectedSeverities}
        selectedTypes={selectedTypes}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onSeverityChange={handleSeverityChange}
        onTypeChange={handleTypeChange}
        onSearchChange={handleSearchChange}
        showCustomizeColumns={true}
        onCustomizeColumnsClick={handleCustomizeColumnsClick}
        customizeColumnsOpen={customizeColumnsOpen}
      />

      {/* Customize Columns Panel (separate column, hidden when Left Rail collapsed) */}
      <CustomizeColumnsPanel
        open={customizeColumnsOpen && !leftRailCollapsed}
        columns={INCIDENTS_COLUMN_OPTIONS}
        columnVisibility={columnVisibility}
        onColumnVisibilityChange={setColumnVisible}
        onRestore={resetToDefault}
        onDone={handleCustomizeColumnsDone}
        onCollapse={handleLeftRailToggle}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table */}
        <IncidentsTable
          data={incidents}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedIncidentId={selectedIncidentId}
          onRowClick={handleRowClick}
          columnVisibility={columnVisibility}
          onColumnVisibilityChange={setColumnVisibility}
        />

        {/* Pagination */}
        <Pagination
          currentPage={currentPage}
          totalItems={pagination.total}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
        />
      </div>

      {/* Detail Drawer (inline on desktop, sheet on mobile) */}
      <IncidentDetailDrawer
        incident={selectedIncident}
        open={drawerOpen}
        onClose={handleCloseDrawer}
      />
    </div>
  );
}

/**
 * Loading fallback for Suspense
 */
function IncidentsLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Incidents Page
 *
 * Master-detail pattern with URL sync (full state):
 * - Canonical: /incidents?id=123&status=active&severity=critical&type=job_failure&q=sync
 * - Uses router.replace with scroll:false
 */
export default function IncidentsPage() {
  return (
    <Suspense fallback={<IncidentsLoading />}>
      <IncidentsPageContent />
    </Suspense>
  );
}
