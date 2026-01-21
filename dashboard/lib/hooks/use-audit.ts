/**
 * Audit hooks using TanStack Query
 */

import { useQuery } from "@tanstack/react-query";
import {
  AuditEventRow,
  AuditEventDetail,
  AuditFilters,
  AuditEventType,
  AuditSeverity,
} from "@/lib/types";
import {
  getAuditEventsMock,
  getAuditEventMock,
  getAuditTypeCountsMock,
  getAuditSeverityCountsMock,
} from "@/lib/mocks";

/**
 * Fetch all audit events with optional filters
 */
export function useAuditEvents(filters?: AuditFilters) {
  return useQuery<AuditEventRow[]>({
    queryKey: ["audit", "events", filters],
    queryFn: () => getAuditEventsMock(filters),
  });
}

/**
 * Fetch a single audit event with full details
 */
export function useAuditEvent(id: number | null) {
  return useQuery<AuditEventDetail | null>({
    queryKey: ["audit", "event", id],
    queryFn: () => (id ? getAuditEventMock(id) : Promise.resolve(null)),
    enabled: id !== null,
  });
}

/**
 * Get type counts (synchronous, for filter badges)
 */
export function useAuditTypeCounts(): Record<AuditEventType, number> {
  return getAuditTypeCountsMock();
}

/**
 * Get severity counts (synchronous, for filter badges)
 */
export function useAuditSeverityCounts(): Record<AuditSeverity, number> {
  return getAuditSeverityCountsMock();
}
