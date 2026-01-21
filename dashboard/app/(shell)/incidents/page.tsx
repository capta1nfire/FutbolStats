"use client";

import { Suspense, useState, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useIncidents, useIncident } from "@/lib/hooks";
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
} from "@/components/incidents";
import {
  parseNumericId,
  parseArrayParam,
  buildSearchParams,
  toggleArrayValue,
} from "@/lib/url-state";
import { Loader2 } from "lucide-react";

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
  const [filterCollapsed, setFilterCollapsed] = useState(false);

  // Construct filters for query
  const filters: IncidentFilters = useMemo(() => ({
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    severity: selectedSeverities.length > 0 ? selectedSeverities : undefined,
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    search: searchValue || undefined,
  }), [selectedStatuses, selectedSeverities, selectedTypes, searchValue]);

  // Fetch data
  const {
    data: incidents = [],
    isLoading,
    error,
    refetch,
  } = useIncidents(filters);

  const { data: selectedIncident } = useIncident(selectedIncidentId);

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
        id: overrides.id ?? selectedIncidentId,
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

  return (
    <div className="h-full flex overflow-hidden">
      {/* FilterPanel */}
      <IncidentsFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
        selectedStatuses={selectedStatuses}
        selectedSeverities={selectedSeverities}
        selectedTypes={selectedTypes}
        searchValue={searchValue}
        onStatusChange={handleStatusChange}
        onSeverityChange={handleSeverityChange}
        onTypeChange={handleTypeChange}
        onSearchChange={handleSearchChange}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Incidents</h1>
          <span className="text-sm text-muted-foreground">
            {incidents.length} incidents
          </span>
        </div>

        {/* Table */}
        <IncidentsTable
          data={incidents}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedIncidentId={selectedIncidentId}
          onRowClick={handleRowClick}
        />
      </div>

      {/* Detail Drawer (inline on desktop, sheet on mobile) */}
      <IncidentDetailDrawer
        incident={selectedIncident ?? null}
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
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading incidents...</p>
      </div>
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
