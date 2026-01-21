"use client";

import { useState } from "react";
import { Incident, INCIDENT_TYPE_LABELS } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Checkbox } from "@/components/ui/checkbox";
import { SeverityBadge } from "./SeverityBadge";
import { IncidentStatusChip } from "./IncidentStatusChip";
import {
  Clock,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  BookOpen,
  History,
} from "lucide-react";

interface IncidentDetailDrawerProps {
  incident: Incident | null;
  open: boolean;
  onClose: () => void;
}

/**
 * Incident Detail Content - shared between desktop drawer and mobile sheet
 */
function IncidentDetailContent({ incident }: { incident: Incident }) {
  // Local state for mock actions (not persisted)
  const [localStatus, setLocalStatus] = useState(incident.status);
  const [runbookSteps, setRunbookSteps] = useState(
    incident.runbook?.steps || []
  );

  const createdAt = new Date(incident.createdAt);
  const formattedCreated = createdAt.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const handleAcknowledge = () => {
    if (localStatus === "active") {
      setLocalStatus("acknowledged");
    }
  };

  const handleResolve = () => {
    if (localStatus !== "resolved") {
      setLocalStatus("resolved");
    }
  };

  const toggleRunbookStep = (stepId: string) => {
    setRunbookSteps((prev) =>
      prev.map((step) =>
        step.id === stepId ? { ...step, done: !step.done } : step
      )
    );
  };

  return (
    <Tabs defaultValue="details" className="w-full">
      <TabsList className="w-full grid grid-cols-3 mb-4">
        <TabsTrigger value="details" className="rounded-full text-xs">
          Details
        </TabsTrigger>
        <TabsTrigger value="runbook" className="rounded-full text-xs">
          Runbook
        </TabsTrigger>
        <TabsTrigger value="history" className="rounded-full text-xs">
          History
        </TabsTrigger>
      </TabsList>

      {/* Details Tab */}
      <TabsContent value="details" className="space-y-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <SeverityBadge severity={incident.severity} />
            <IncidentStatusChip status={localStatus} />
          </div>

          {/* Description */}
          {incident.description && (
            <div className="bg-background rounded-lg p-4">
              <p className="text-sm text-foreground">{incident.description}</p>
            </div>
          )}

          {/* Metadata */}
          <div className="bg-background rounded-lg p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Created:</span>
              <span className="text-foreground">{formattedCreated}</span>
            </div>

            <div className="flex items-center gap-2 text-sm">
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Type:</span>
              <span className="text-foreground">
                {INCIDENT_TYPE_LABELS[incident.type]}
              </span>
            </div>

            {incident.entity && (
              <div className="flex items-center gap-2 text-sm">
                <ExternalLink className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Related:</span>
                <span className="text-accent">
                  {incident.entity.kind} #{incident.entity.id}
                </span>
              </div>
            )}
          </div>

          {/* Quick info */}
          <div className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Incident ID:</span>{" "}
              <span className="text-foreground font-mono">{incident.id}</span>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2 pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleAcknowledge}
              disabled={localStatus !== "active"}
              className="flex-1"
            >
              <CheckCircle2 className="h-4 w-4 mr-2" />
              Acknowledge
            </Button>
            <Button
              variant="default"
              size="sm"
              onClick={handleResolve}
              disabled={localStatus === "resolved"}
              className="flex-1"
            >
              <CheckCircle2 className="h-4 w-4 mr-2" />
              Resolve
            </Button>
          </div>

          {/* Phase 0 notice */}
          <p className="text-xs text-muted-foreground text-center italic">
            Actions are UI mock only - not persisted in Phase 0
          </p>
        </div>
      </TabsContent>

      {/* Runbook Tab */}
      <TabsContent value="runbook" className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Resolution Steps</span>
        </div>

        <div className="space-y-3">
          {runbookSteps.map((step, index) => (
            <div
              key={step.id}
              className="flex items-start gap-3 bg-background rounded-lg p-3"
            >
              <Checkbox
                id={`step-${step.id}`}
                checked={step.done}
                onCheckedChange={() => toggleRunbookStep(step.id)}
                className="mt-0.5"
              />
              <div className="flex-1">
                <label
                  htmlFor={`step-${step.id}`}
                  className={`text-sm cursor-pointer ${
                    step.done
                      ? "text-muted-foreground line-through"
                      : "text-foreground"
                  }`}
                >
                  {index + 1}. {step.text}
                </label>
              </div>
            </div>
          ))}
        </div>

        <div className="text-xs text-muted-foreground mt-4">
          Progress: {runbookSteps.filter((s) => s.done).length} /{" "}
          {runbookSteps.length} steps completed
        </div>
      </TabsContent>

      {/* History Tab */}
      <TabsContent value="history" className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <History className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Timeline</span>
        </div>

        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-2 top-2 bottom-2 w-px bg-border" />

          <div className="space-y-4">
            {incident.timeline?.map((event, index) => {
              const eventTime = new Date(event.ts);
              return (
                <div key={index} className="flex gap-4 relative">
                  <div className="w-4 h-4 rounded-full bg-surface border-2 border-border shrink-0 z-10" />
                  <div className="flex-1 pb-2">
                    <p className="text-sm text-foreground">{event.message}</p>
                    <p className="text-xs text-muted-foreground">
                      {eventTime.toLocaleString("en-US", {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </TabsContent>
    </Tabs>
  );
}

/**
 * Responsive Incident Detail Drawer
 *
 * Desktop (>=1280px): Inline drawer that pushes content
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function IncidentDetailDrawer({
  incident,
  open,
  onClose,
}: IncidentDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const incidentTitle = incident
    ? `Incident #${incident.id}`
    : "Incident Details";

  // Desktop: inline drawer
  if (isDesktop) {
    return (
      <DetailDrawer open={open} onClose={onClose} title={incidentTitle}>
        {incident ? (
          <IncidentDetailContent incident={incident} />
        ) : (
          <p className="text-muted-foreground text-sm">
            Select an incident to view details
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
            {incidentTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {incident ? (
              <IncidentDetailContent incident={incident} />
            ) : (
              <p className="text-muted-foreground text-sm">
                Select an incident to view details
              </p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
