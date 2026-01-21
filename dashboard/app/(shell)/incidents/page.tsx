"use client";

import { Suspense, useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useIncidents, useIncident } from "@/lib/hooks";
import {
  Incident,
  IncidentStatus,
  IncidentSeverity,
  IncidentType,
  IncidentFilters,
} from "@/lib/types";
import {
  IncidentsTable,
  IncidentsFilterPanel,
  IncidentDetailDrawer,
} from "@/components/incidents";
import { Loader2 } from "lucide-react";

/**
 * Parse and validate incident ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
function parseIncidentId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Incidents Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function IncidentsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL state: selected incident ID (sanitized)
  const selectedIdParam = searchParams.get("id");
  const selectedIncidentId = parseIncidentId(selectedIdParam);

  // Normalize URL if id param is invalid
  useEffect(() => {
    if (selectedIdParam && selectedIncidentId === null) {
      // Invalid id in URL → normalize to /incidents
      router.replace("/incidents", { scroll: false });
    }
  }, [selectedIdParam, selectedIncidentId, router]);

  // UI state
  const [filterCollapsed, setFilterCollapsed] = useState(false);
  const [selectedStatuses, setSelectedStatuses] = useState<IncidentStatus[]>([]);
  const [selectedSeverities, setSelectedSeverities] = useState<IncidentSeverity[]>([]);
  const [selectedTypes, setSelectedTypes] = useState<IncidentType[]>([]);
  const [searchValue, setSearchValue] = useState("");

  // Construct filters
  const filters: IncidentFilters = {
    status: selectedStatuses.length > 0 ? selectedStatuses : undefined,
    severity: selectedSeverities.length > 0 ? selectedSeverities : undefined,
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    search: searchValue || undefined,
  };

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

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (incident: Incident) => {
      router.replace(`/incidents?id=${incident.id}`, { scroll: false });
    },
    [router]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace("/incidents", { scroll: false });
  }, [router]);

  // Handle filter changes
  const handleStatusChange = useCallback(
    (status: IncidentStatus, checked: boolean) => {
      setSelectedStatuses((prev) =>
        checked ? [...prev, status] : prev.filter((s) => s !== status)
      );
    },
    []
  );

  const handleSeverityChange = useCallback(
    (severity: IncidentSeverity, checked: boolean) => {
      setSelectedSeverities((prev) =>
        checked ? [...prev, severity] : prev.filter((s) => s !== severity)
      );
    },
    []
  );

  const handleTypeChange = useCallback(
    (type: IncidentType, checked: boolean) => {
      setSelectedTypes((prev) =>
        checked ? [...prev, type] : prev.filter((t) => t !== type)
      );
    },
    []
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
        onSearchChange={setSearchValue}
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
 * Master-detail pattern with:
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (inline on desktop, right, pushes content)
 *
 * URL sync:
 * - Canonical: /incidents?id=123
 * - Uses router.replace with scroll:false
 */
export default function IncidentsPage() {
  return (
    <Suspense fallback={<IncidentsLoading />}>
      <IncidentsPageContent />
    </Suspense>
  );
}
