"use client";

import { Suspense, useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuditEvents, useAuditEvent } from "@/lib/hooks";
import {
  AuditEventRow,
  AuditEventType,
  AuditSeverity,
  AuditFilters,
} from "@/lib/types";
import {
  AuditTable,
  AuditFilterPanel,
  AuditDetailDrawer,
} from "@/components/audit";
import { Loader2 } from "lucide-react";

/**
 * Parse and validate event ID from URL parameter
 * Returns null if invalid (non-numeric, NaN, negative)
 */
function parseEventId(param: string | null): number | null {
  if (!param) return null;
  const parsed = parseInt(param, 10);
  if (isNaN(parsed) || parsed < 0) return null;
  return parsed;
}

/**
 * Audit Page Content
 *
 * Wrapped in Suspense because it uses useSearchParams
 */
function AuditPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // URL state: selected event ID (sanitized)
  const selectedIdParam = searchParams.get("id");
  const selectedEventId = parseEventId(selectedIdParam);

  // Normalize URL if id param is invalid
  useEffect(() => {
    if (selectedIdParam && selectedEventId === null) {
      // Invalid id in URL → normalize to /audit
      router.replace("/audit", { scroll: false });
    }
  }, [selectedIdParam, selectedEventId, router]);

  // UI state
  const [filterCollapsed, setFilterCollapsed] = useState(false);
  const [selectedTypes, setSelectedTypes] = useState<AuditEventType[]>([]);
  const [selectedSeverities, setSelectedSeverities] = useState<AuditSeverity[]>([]);
  const [searchValue, setSearchValue] = useState("");

  // Construct filters
  const filters: AuditFilters = {
    type: selectedTypes.length > 0 ? selectedTypes : undefined,
    severity: selectedSeverities.length > 0 ? selectedSeverities : undefined,
    search: searchValue || undefined,
  };

  // Fetch data
  const {
    data: events = [],
    isLoading,
    error,
    refetch,
  } = useAuditEvents(filters);

  const {
    data: selectedEvent,
    isLoading: isLoadingDetail,
  } = useAuditEvent(selectedEventId);

  // Drawer is open when there's a selected event
  const drawerOpen = selectedEventId !== null;

  // Handle row click - update URL with router.replace (no history entry)
  const handleRowClick = useCallback(
    (event: AuditEventRow) => {
      router.replace(`/audit?id=${event.id}`, { scroll: false });
    },
    [router]
  );

  // Handle drawer close - remove id from URL
  const handleCloseDrawer = useCallback(() => {
    router.replace("/audit", { scroll: false });
  }, [router]);

  // Handle filter changes
  const handleTypeChange = useCallback(
    (type: AuditEventType, checked: boolean) => {
      setSelectedTypes((prev) =>
        checked ? [...prev, type] : prev.filter((t) => t !== type)
      );
    },
    []
  );

  const handleSeverityChange = useCallback(
    (severity: AuditSeverity, checked: boolean) => {
      setSelectedSeverities((prev) =>
        checked ? [...prev, severity] : prev.filter((s) => s !== severity)
      );
    },
    []
  );

  return (
    <div className="h-full flex overflow-hidden">
      {/* FilterPanel */}
      <AuditFilterPanel
        collapsed={filterCollapsed}
        onToggleCollapse={() => setFilterCollapsed(!filterCollapsed)}
        selectedTypes={selectedTypes}
        selectedSeverities={selectedSeverities}
        searchValue={searchValue}
        onTypeChange={handleTypeChange}
        onSeverityChange={handleSeverityChange}
        onSearchChange={setSearchValue}
      />

      {/* Main content: Table */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Table header */}
        <div className="h-12 flex items-center justify-between px-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">Audit Trail</h1>
          <span className="text-sm text-muted-foreground">
            {events.length} events
          </span>
        </div>

        {/* Table */}
        <AuditTable
          data={events}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
          selectedEventId={selectedEventId}
          onRowClick={handleRowClick}
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
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading audit events...</p>
      </div>
    </div>
  );
}

/**
 * Audit Page
 *
 * Master-detail pattern with:
 * - FilterPanel (collapsible, left)
 * - DataTable (center)
 * - DetailDrawer (inline on desktop, right, pushes content)
 *
 * URL sync:
 * - Canonical: /audit?id=123
 * - Uses router.replace with scroll:false
 */
export default function AuditPage() {
  return (
    <Suspense fallback={<AuditLoading />}>
      <AuditPageContent />
    </Suspense>
  );
}
