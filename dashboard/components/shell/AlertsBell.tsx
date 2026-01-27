"use client";

import { useEffect, useRef, useState } from "react";
import { Bell, ExternalLink, Check, AlertTriangle, AlertCircle, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useHasMounted } from "@/lib/hooks";
import { useAlerts } from "@/lib/hooks/use-alerts";
import { Alert, AlertSeverity } from "@/lib/types/alerts";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui/use-toast";

/**
 * Severity icon mapping
 */
const severityIcons: Record<AlertSeverity, React.ReactNode> = {
  critical: <AlertTriangle className="h-4 w-4 text-red-500" />,
  warning: <AlertCircle className="h-4 w-4 text-yellow-500" />,
  info: <Info className="h-4 w-4 text-blue-500" />,
};

/**
 * Severity badge colors
 */
const severityColors: Record<AlertSeverity, string> = {
  critical: "bg-red-500/20 text-red-400 border-red-500/30",
  warning: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  info: "bg-blue-500/20 text-blue-400 border-blue-500/30",
};

/**
 * Format relative time (e.g., "5m ago", "2h ago")
 */
function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) return "";

  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

/**
 * Single alert item in the dropdown
 */
function AlertItem({ alert, onMarkRead }: { alert: Alert; onMarkRead: (id: number) => void }) {
  return (
    <div
      className={cn(
        "p-3 border-b border-border last:border-b-0 hover:bg-muted/50 transition-colors",
        !alert.is_read && "bg-primary/5"
      )}
    >
      <div className="flex items-start gap-2">
        {severityIcons[alert.severity]}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded border font-medium uppercase",
                severityColors[alert.severity]
              )}
            >
              {alert.severity}
            </span>
            <span className="text-[10px] text-muted-foreground">
              {formatRelativeTime(alert.starts_at || alert.last_seen_at)}
            </span>
          </div>
          <p className="text-sm font-medium text-foreground mt-1 line-clamp-2">
            {alert.title}
          </p>
          {alert.message && (
            <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
              {alert.message}
            </p>
          )}
          <div className="flex items-center gap-2 mt-2">
            {alert.source_url && (
              <a
                href={alert.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[10px] text-primary hover:underline flex items-center gap-1"
              >
                <ExternalLink className="h-3 w-3" />
                View in Grafana
              </a>
            )}
            {!alert.is_read && (
              <button
                onClick={() => onMarkRead(alert.id)}
                className="text-[10px] text-muted-foreground hover:text-foreground flex items-center gap-1 ml-auto"
              >
                <Check className="h-3 w-3" />
                Mark read
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Alerts Bell component with dropdown
 *
 * Features:
 * - Badge showing unread count (firing alerts only)
 * - Dropdown with alert list
 * - Toast for new critical alerts (deduplicated via localStorage)
 * - Mark all as read button
 */
export function AlertsBell() {
  const mounted = useHasMounted();
  const [open, setOpen] = useState(false);
  const {
    alerts,
    unreadCount,
    isLoading,
    markAsRead,
    markAllAsRead,
    isAcking,
    newCriticalAlerts,
    markAsSeen,
  } = useAlerts();
  const { toast } = useToast();
  const shownToastIdsRef = useRef<Set<number>>(new Set());

  // Show toast for new critical alerts
  useEffect(() => {
    for (const alert of newCriticalAlerts) {
      // Skip if already shown in this session
      if (shownToastIdsRef.current.has(alert.id)) continue;

      // Show toast
      toast({
        variant: "destructive",
        title: `Critical Alert: ${alert.title}`,
        description: alert.message || undefined,
        duration: 10000, // 10 seconds
      });

      // Mark as shown in session
      shownToastIdsRef.current.add(alert.id);
    }

    // Mark all as seen in localStorage
    if (newCriticalAlerts.length > 0) {
      markAsSeen(newCriticalAlerts.map((a) => a.id));
    }
  }, [newCriticalAlerts, toast, markAsSeen]);

  const handleMarkRead = async (id: number) => {
    await markAsRead([id]);
  };

  const handleMarkAllRead = async () => {
    await markAllAsRead();
    setOpen(false);
  };

  // Filter to only firing alerts for display
  const firingAlerts = alerts.filter((a) => a.status === "firing");

  // Render placeholder during SSR to prevent hydration mismatch
  if (!mounted) {
    return (
      <Button
        variant="ghost"
        size="icon"
        className="relative"
        aria-label="Notifications"
      >
        <Bell className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />
      </Button>
    );
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative"
          aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ""}`}
        >
          <Bell
            className={cn(
              "h-5 w-5 transition-colors",
              unreadCount > 0 ? "text-foreground" : "text-muted-foreground"
            )}
            strokeWidth={1.5}
          />
          {/* Badge */}
          {unreadCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] px-1 bg-red-500 text-white text-[10px] font-medium rounded-full flex items-center justify-center">
              {unreadCount > 99 ? "99+" : unreadCount}
            </span>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[380px] p-0 max-h-[480px] overflow-hidden flex flex-col"
        align="end"
        sideOffset={8}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
          <h3 className="text-sm font-semibold text-foreground">
            Alerts
            {unreadCount > 0 && (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                ({unreadCount} unread)
              </span>
            )}
          </h3>
          {unreadCount > 0 && (
            <button
              onClick={handleMarkAllRead}
              disabled={isAcking}
              className="text-xs text-primary hover:underline disabled:opacity-50"
            >
              Mark all read
            </button>
          )}
        </div>

        {/* Content */}
        <div className="overflow-y-auto flex-1">
          {isLoading ? (
            <div className="p-8 text-center text-sm text-muted-foreground">
              Loading alerts...
            </div>
          ) : firingAlerts.length === 0 ? (
            <div className="p-8 text-center">
              <Bell className="h-8 w-8 text-muted-foreground mx-auto mb-2" strokeWidth={1} />
              <p className="text-sm text-muted-foreground">No active alerts</p>
              <p className="text-xs text-muted-foreground mt-1">
                You&apos;re all caught up!
              </p>
            </div>
          ) : (
            firingAlerts.map((alert) => (
              <AlertItem
                key={alert.id}
                alert={alert}
                onMarkRead={handleMarkRead}
              />
            ))
          )}
        </div>

        {/* Footer */}
        {firingAlerts.length > 0 && (
          <div className="px-4 py-2 border-t border-border bg-muted/30">
            <p className="text-[10px] text-muted-foreground text-center">
              Showing {firingAlerts.length} firing alert{firingAlerts.length !== 1 ? "s" : ""}
            </p>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
