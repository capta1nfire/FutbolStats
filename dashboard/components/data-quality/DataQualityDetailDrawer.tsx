"use client";

import { DataQualityCheckDetail } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DataQualityStatusBadge } from "./DataQualityStatusBadge";
import { DataQualityCategoryBadge } from "./DataQualityCategoryBadge";
import {
  Clock,
  AlertCircle,
  Activity,
  FileText,
  History,
  BarChart3,
} from "lucide-react";

interface DataQualityDetailDrawerProps {
  check: DataQualityCheckDetail | null;
  open: boolean;
  onClose: () => void;
  isLoading?: boolean;
}

/**
 * Data Quality Detail Content - shared between desktop drawer and mobile sheet
 */
function DataQualityDetailContent({ check }: { check: DataQualityCheckDetail }) {
  const lastRunAt = new Date(check.lastRunAt);
  const formattedLastRun = lastRunAt.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <Tabs defaultValue="overview" className="w-full">
      <TabsList className="w-full grid grid-cols-3 mb-4">
        <TabsTrigger value="overview" className="rounded-full text-xs">
          Overview
        </TabsTrigger>
        <TabsTrigger value="affected" className="rounded-full text-xs">
          Affected
        </TabsTrigger>
        <TabsTrigger value="history" className="rounded-full text-xs">
          History
        </TabsTrigger>
      </TabsList>

      {/* Overview Tab */}
      <TabsContent value="overview" className="space-y-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <DataQualityStatusBadge status={check.status} />
            <DataQualityCategoryBadge category={check.category} />
          </div>

          {/* Description */}
          {check.description && (
            <div className="bg-background rounded-lg p-4">
              <p className="text-sm text-foreground">{check.description}</p>
            </div>
          )}

          {/* Metrics */}
          <div className="bg-background rounded-lg p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Current Value:</span>
              <span className="text-foreground font-mono">{check.currentValue ?? "-"}</span>
            </div>

            <div className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Threshold:</span>
              <span className="text-foreground font-mono">{check.threshold ?? "-"}</span>
            </div>

            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Last Run:</span>
              <span className="text-foreground">{formattedLastRun}</span>
            </div>

            <div className="flex items-center gap-2 text-sm">
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Affected Items:</span>
              <span className={check.affectedCount > 0 ? "text-yellow-400" : "text-foreground"}>
                {check.affectedCount}
              </span>
            </div>
          </div>

          {/* Quick info */}
          <div className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Check ID:</span>{" "}
              <span className="text-foreground font-mono">{check.id}</span>
            </div>
          </div>

          {/* Phase 0 notice */}
          <p className="text-xs text-muted-foreground text-center italic pt-4">
            Data from mocks - Phase 0
          </p>
        </div>
      </TabsContent>

      {/* Affected Items Tab */}
      <TabsContent value="affected" className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">
            Affected Items ({check.affectedItems.length})
          </span>
        </div>

        {check.affectedItems.length > 0 ? (
          <div className="space-y-2">
            {check.affectedItems.map((item) => (
              <div
                key={item.id}
                className="bg-background rounded-lg p-3 flex items-center justify-between"
              >
                <div className="flex-1">
                  <p className="text-sm text-foreground">{item.label}</p>
                  <p className="text-xs text-muted-foreground capitalize">
                    {item.kind}
                  </p>
                </div>
                {item.details?.reason && (
                  <span className="text-xs text-yellow-400">
                    {item.details.reason}
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <AlertCircle className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No affected items</p>
          </div>
        )}
      </TabsContent>

      {/* History Tab */}
      <TabsContent value="history" className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <History className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Check History (24h)</span>
        </div>

        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-2 top-2 bottom-2 w-px bg-border" />

          <div className="space-y-3">
            {check.history.slice(0, 10).map((entry, index) => {
              const entryTime = new Date(entry.timestamp);
              return (
                <div key={index} className="flex gap-4 relative">
                  <div
                    className={`w-4 h-4 rounded-full shrink-0 z-10 border-2 ${
                      entry.status === "passing"
                        ? "bg-green-500/20 border-green-500"
                        : entry.status === "warning"
                        ? "bg-yellow-500/20 border-yellow-500"
                        : "bg-red-500/20 border-red-500"
                    }`}
                  />
                  <div className="flex-1 pb-2">
                    <div className="flex items-center gap-2">
                      <DataQualityStatusBadge status={entry.status} showIcon={false} />
                      <span className="text-sm font-mono text-foreground">
                        {entry.value ?? "-"}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {entryTime.toLocaleString("en-US", {
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

        {check.history.length > 10 && (
          <p className="text-xs text-muted-foreground text-center">
            Showing latest 10 of {check.history.length} entries
          </p>
        )}
      </TabsContent>
    </Tabs>
  );
}

/**
 * Responsive Data Quality Detail Drawer
 *
 * Desktop (>=1280px): Inline drawer that pushes content
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function DataQualityDetailDrawer({
  check,
  open,
  onClose,
  isLoading,
}: DataQualityDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const checkTitle = check ? check.name : "Check Details";

  // Loading state content
  const loadingContent = (
    <div className="space-y-4">
      <div className="h-8 bg-surface rounded animate-pulse" />
      <div className="h-24 bg-surface rounded animate-pulse" />
      <div className="h-32 bg-surface rounded animate-pulse" />
    </div>
  );

  // Desktop: inline drawer
  if (isDesktop) {
    return (
      <DetailDrawer open={open} onClose={onClose} title={checkTitle}>
        {isLoading ? (
          loadingContent
        ) : check ? (
          <DataQualityDetailContent check={check} />
        ) : (
          <p className="text-muted-foreground text-sm">
            Select a check to view details
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
            {checkTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {isLoading ? (
              loadingContent
            ) : check ? (
              <DataQualityDetailContent check={check} />
            ) : (
              <p className="text-muted-foreground text-sm">
                Select a check to view details
              </p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
