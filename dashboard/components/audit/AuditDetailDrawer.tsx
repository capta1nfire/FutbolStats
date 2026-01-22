"use client";

import { useState } from "react";
import { AuditEventDetail, AUDIT_EVENT_TYPE_LABELS } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { IconTabs } from "@/components/ui/icon-tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AuditSeverityBadge } from "./AuditSeverityBadge";
import { AuditTypeBadge } from "./AuditTypeBadge";
import {
  Clock,
  User,
  Server,
  FileJson,
  Link2,
  History,
  ExternalLink,
  Info,
} from "lucide-react";

/** Tab definitions for audit detail drawer */
const AUDIT_TABS = [
  { id: "details", icon: <Info />, label: "Details" },
  { id: "payload", icon: <FileJson />, label: "Payload" },
  { id: "context", icon: <Server />, label: "Context" },
];

interface AuditDetailDrawerProps {
  event: AuditEventDetail | null;
  open: boolean;
  onClose: () => void;
  isLoading?: boolean;
}

/**
 * Audit Tab Content - content only, without tabs (for desktop drawer with fixedContent)
 */
function AuditTabContent({ event, activeTab }: { event: AuditEventDetail; activeTab: string }) {
  const timestamp = new Date(event.timestamp);
  const formattedTimestamp = timestamp.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const isUserActor = event.actor.kind === "user";

  return (
    <div className="w-full">
      {/* Details Tab */}
      {activeTab === "details" && (
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            {event.severity && <AuditSeverityBadge severity={event.severity} />}
            <AuditTypeBadge type={event.type} />
          </div>

          {/* Message */}
          <div className="pt-3 border-t border-border">
            <p className="text-sm text-foreground">{event.message}</p>
          </div>

          {/* Metadata */}
          <div className="space-y-3 pt-3 border-t border-border">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Timestamp:</span>
              <span className="text-foreground">{formattedTimestamp}</span>
            </div>

            <div className="flex items-center gap-2 text-sm">
              {isUserActor ? (
                <User className="h-4 w-4 text-muted-foreground" />
              ) : (
                <Server className="h-4 w-4 text-muted-foreground" />
              )}
              <span className="text-muted-foreground">Actor:</span>
              <span className={isUserActor ? "text-accent" : "text-foreground"}>
                {event.actor.name}
                {isUserActor && event.actor.kind === "user" && ` (ID: ${event.actor.id})`}
              </span>
            </div>

            {event.entity && (
              <div className="flex items-center gap-2 text-sm">
                <Link2 className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Entity:</span>
                <span className="text-accent flex items-center gap-1">
                  {event.entity.kind} #{event.entity.id}
                  <ExternalLink className="h-3 w-3" />
                </span>
              </div>
            )}
          </div>

          {/* Quick info */}
          <div className="space-y-2 pt-3 border-t border-border">
            <div className="text-sm">
              <span className="text-muted-foreground">Event ID:</span>{" "}
              <span className="text-foreground font-mono">{event.id}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Type:</span>{" "}
              <span className="text-foreground">{AUDIT_EVENT_TYPE_LABELS[event.type]}</span>
            </div>
          </div>

          {/* Phase 0 notice */}
          <p className="text-xs text-muted-foreground text-center italic pt-4">
            Data from mocks - Phase 0
          </p>
        </div>
      )}

      {/* Payload Tab */}
      {activeTab === "payload" && (
        <div className="bg-surface rounded-lg p-4">
          <div className="flex items-center gap-2 mb-4">
            <FileJson className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Event Payload</span>
          </div>

          {event.payload ? (
            <div className="bg-background rounded-lg p-4 overflow-hidden">
              <pre className="text-xs text-foreground font-mono whitespace-pre-wrap overflow-x-auto">
                {JSON.stringify(event.payload, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="text-center py-8">
              <FileJson className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">No payload data</p>
            </div>
          )}
        </div>
      )}

      {/* Context Tab */}
      {activeTab === "context" && (
        <div className="bg-surface rounded-lg p-4">
          {/* Request Context */}
          {event.context && (
            <>
              <div className="flex items-center gap-2 mb-4">
                <Server className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Request Context</span>
              </div>
              <div className="bg-background rounded-lg p-4 space-y-2">
                {event.context.requestId && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">Request ID:</span>{" "}
                    <span className="text-foreground font-mono text-xs">{event.context.requestId}</span>
                  </div>
                )}
                {event.context.correlationId && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">Correlation ID:</span>{" "}
                    <span className="text-foreground font-mono text-xs">{event.context.correlationId}</span>
                  </div>
                )}
                {event.context.ip && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">IP:</span>{" "}
                    <span className="text-foreground font-mono">{event.context.ip}</span>
                  </div>
                )}
                {event.context.userAgent && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">User Agent:</span>{" "}
                    <span className="text-foreground text-xs">{event.context.userAgent}</span>
                  </div>
                )}
                {event.context.env && (
                  <div className="text-sm">
                    <span className="text-muted-foreground">Environment:</span>{" "}
                    <span className="text-foreground">{event.context.env}</span>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Related Events */}
          {event.related && event.related.length > 0 && (
            <div className={event.context ? "pt-3 border-t border-border mt-4" : ""}>
              <div className="flex items-center gap-2 mb-3">
                <History className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Related Events</span>
              </div>
              <div className="space-y-2">
                {event.related.map((rel) => {
                  const relTime = new Date(rel.timestamp);
                  return (
                    <div
                      key={rel.id}
                      className="bg-background rounded-lg p-3 flex items-start justify-between"
                    >
                      <div className="flex-1">
                        <p className="text-sm text-foreground">{rel.message}</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {relTime.toLocaleString("en-US", {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                      <span className="text-xs text-muted-foreground font-mono">
                        #{rel.id}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {!event.context && (!event.related || event.related.length === 0) && (
            <div className="text-center py-8">
              <Server className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">No context data</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Audit Detail Content - used for mobile sheet (tabs + content together)
 */
function AuditDetailContentMobile({ event }: { event: AuditEventDetail }) {
  const [activeTab, setActiveTab] = useState("details");

  return (
    <div className="w-full space-y-3">
      <IconTabs
        tabs={AUDIT_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />
      <AuditTabContent event={event} activeTab={activeTab} />
    </div>
  );
}

/**
 * Responsive Audit Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer (no reflow, ~400px)
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function AuditDetailDrawer({
  event,
  open,
  onClose,
  isLoading,
}: AuditDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const [activeTab, setActiveTab] = useState("details");
  const eventTitle = event ? `Event #${event.id}` : "Event Details";

  // Loading state content
  const loadingContent = (
    <div className="space-y-4">
      <div className="h-8 bg-surface rounded animate-pulse" />
      <div className="h-24 bg-surface rounded animate-pulse" />
      <div className="h-32 bg-surface rounded animate-pulse" />
    </div>
  );

  // Desktop: overlay drawer with tabs in fixedContent
  if (isDesktop) {
    return (
      <DetailDrawer
        open={open}
        onClose={onClose}
        title={eventTitle}
        fixedContent={
          event && !isLoading && (
            <IconTabs
              tabs={AUDIT_TABS}
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            />
          )
        }
      >
        {isLoading ? (
          loadingContent
        ) : event ? (
          <AuditTabContent event={event} activeTab={activeTab} />
        ) : (
          <p className="text-muted-foreground text-sm">
            Select an event to view details
          </p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {eventTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {isLoading ? (
              loadingContent
            ) : event ? (
              <AuditDetailContentMobile event={event} />
            ) : (
              <p className="text-muted-foreground text-sm">
                Select an event to view details
              </p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
