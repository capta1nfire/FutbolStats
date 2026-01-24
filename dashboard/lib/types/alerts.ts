/**
 * Types for Grafana Alerts
 *
 * These types define the response shapes from:
 * - GET /dashboard/ops/alerts.json
 * - POST /dashboard/ops/alerts/ack
 */

export type AlertStatus = "firing" | "resolved";
export type AlertSeverity = "critical" | "warning" | "info";

/**
 * Single alert item from /dashboard/ops/alerts.json
 */
export interface Alert {
  id: number;
  dedupe_key: string;
  status: AlertStatus;
  severity: AlertSeverity;
  title: string;
  message: string | null;
  starts_at: string | null;
  ends_at: string | null;
  last_seen_at: string | null;
  source_url: string | null;
  is_read: boolean;
  is_ack: boolean;
}

/**
 * Response from GET /dashboard/ops/alerts.json
 */
export interface AlertsResponse {
  unread_count: number;
  items: Alert[];
}

/**
 * Request body for POST /dashboard/ops/alerts/ack
 */
export type AckAlertsRequest = { ids: number[] } | { ack_all: true };

/**
 * Response from POST /dashboard/ops/alerts/ack
 */
export interface AckAlertsResponse {
  status: "ok";
  updated: number;
}
