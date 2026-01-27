"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
import {
  Alert,
  AlertsResponse,
  AckAlertsRequest,
  AckAlertsResponse,
} from "@/lib/types/alerts";

const POLLING_INTERVAL_MS = 20_000; // 20 seconds
const SEEN_ALERTS_KEY = "futbolstats_seen_alerts";

export interface UseAlertsResult {
  alerts: Alert[];
  unreadCount: number;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
  markAsRead: (ids: number[]) => Promise<void>;
  markAllAsRead: () => Promise<void>;
  isAcking: boolean;
  /** New critical firing alerts not yet shown as toast */
  newCriticalAlerts: Alert[];
  /** Mark alerts as "seen" (shown toast) to prevent duplicate toasts */
  markAsSeen: (ids: number[]) => void;
}

/**
 * Fetch alerts from API
 */
async function fetchAlerts(): Promise<AlertsResponse> {
  const response = await fetch("/api/alerts?status=firing&limit=50");

  if (!response.ok) {
    throw new Error(`Failed to fetch alerts: ${response.status}`);
  }

  return response.json();
}

/**
 * Acknowledge alerts
 */
async function ackAlerts(body: AckAlertsRequest): Promise<AckAlertsResponse> {
  const response = await fetch("/api/alerts/ack", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Failed to ack alerts: ${response.status}`);
  }

  return response.json();
}

/**
 * Load seen alert IDs from localStorage
 */
function loadSeenAlertIds(): Set<number> {
  if (typeof window === "undefined") return new Set();

  try {
    const stored = localStorage.getItem(SEEN_ALERTS_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) {
        return new Set(parsed);
      }
    }
  } catch {
    // Ignore localStorage errors
  }
  return new Set();
}

/**
 * Save seen alert IDs to localStorage
 */
function saveSeenAlertIds(ids: Set<number>): void {
  if (typeof window === "undefined") return;

  try {
    // Keep only last 200 IDs to prevent localStorage bloat
    const idsArray = Array.from(ids).slice(-200);
    localStorage.setItem(SEEN_ALERTS_KEY, JSON.stringify(idsArray));
  } catch {
    // Ignore localStorage errors
  }
}

/**
 * Hook for fetching and managing alerts
 *
 * Features:
 * - Polls every 20 seconds
 * - Tracks "seen" alerts in localStorage to prevent duplicate toasts
 * - Exposes newCriticalAlerts for toast display
 * - Methods to mark alerts as read via API
 */
export function useAlerts(): UseAlertsResult {
  const queryClient = useQueryClient();
  const [seenIds, setSeenIds] = useState<Set<number>>(() => loadSeenAlertIds());

  // Query for alerts
  const query = useQuery({
    queryKey: ["alerts"],
    queryFn: fetchAlerts,
    staleTime: 10_000, // 10 seconds
    refetchInterval: POLLING_INTERVAL_MS,
    refetchOnWindowFocus: true,
    retry: 1,
  });

  // Mutation for acking alerts
  const ackMutation = useMutation({
    mutationFn: ackAlerts,
    onSuccess: () => {
      // Invalidate alerts query to refetch
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
    },
  });

  // Mark specific alerts as read
  const markAsRead = useCallback(
    async (ids: number[]) => {
      if (ids.length === 0) return;
      await ackMutation.mutateAsync({ ids });
    },
    [ackMutation]
  );

  // Mark all alerts as read
  const markAllAsRead = useCallback(async () => {
    await ackMutation.mutateAsync({ ack_all: true });
  }, [ackMutation]);

  // Mark alerts as "seen" (for toast deduplication)
  const markAsSeen = useCallback((ids: number[]) => {
    if (ids.length === 0) return;
    setSeenIds((prev) => {
      const next = new Set(prev);
      for (const id of ids) {
        next.add(id);
      }
      saveSeenAlertIds(next);
      return next;
    });
  }, []);

  // Calculate new critical alerts (not yet shown as toast)
  const newCriticalAlerts = useMemo(() => {
    const items = query.data?.items ?? [];
    if (items.length === 0) return [];
    return items.filter(
      (alert) =>
        alert.severity === "critical" &&
        alert.status === "firing" &&
        !alert.is_read &&
        !seenIds.has(alert.id)
    );
  }, [query.data?.items, seenIds]);

  return {
    alerts: query.data?.items ?? [],
    unreadCount: query.data?.unread_count ?? 0,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error as Error | null,
    refetch: query.refetch,
    markAsRead,
    markAllAsRead,
    isAcking: ackMutation.isPending,
    newCriticalAlerts,
    markAsSeen,
  };
}
