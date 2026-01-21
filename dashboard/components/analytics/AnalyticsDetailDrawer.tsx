"use client";

import { AnalyticsReportDetail, ANALYTICS_REPORT_TYPE_LABELS } from "@/lib/types";
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
import { ReportStatusBadge } from "./ReportStatusBadge";
import { ReportTypeBadge } from "./ReportTypeBadge";
import {
  Clock,
  BarChart3,
  Table2,
  TrendingUp,
} from "lucide-react";

interface AnalyticsDetailDrawerProps {
  report: AnalyticsReportDetail | null;
  open: boolean;
  onClose: () => void;
  isLoading?: boolean;
}

/**
 * Analytics Detail Content - shared between desktop drawer and mobile sheet
 */
function AnalyticsDetailContent({ report }: { report: AnalyticsReportDetail }) {
  const { row, breakdownTable, seriesPlaceholder } = report;

  const lastUpdated = new Date(row.lastUpdated);
  const formattedLastUpdated = lastUpdated.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <Tabs defaultValue="summary" className="w-full">
      <TabsList className="w-full grid grid-cols-3 mb-4">
        <TabsTrigger value="summary" className="rounded-full text-xs">
          Summary
        </TabsTrigger>
        <TabsTrigger value="breakdown" className="rounded-full text-xs">
          Breakdown
        </TabsTrigger>
        <TabsTrigger value="trends" className="rounded-full text-xs">
          Trends
        </TabsTrigger>
      </TabsList>

      {/* Summary Tab */}
      <TabsContent value="summary" className="space-y-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            {row.status && <ReportStatusBadge status={row.status} />}
            <ReportTypeBadge type={row.type} />
          </div>

          {/* Period */}
          <div className="bg-background rounded-lg p-4">
            <p className="text-sm text-foreground font-medium">{row.periodLabel}</p>
          </div>

          {/* Metrics */}
          <div className="bg-background rounded-lg p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm mb-3">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium text-foreground">Key Metrics</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {Object.entries(row.summary).map(([key, value]) => (
                <div key={key} className="bg-surface rounded p-2">
                  <p className="text-xs text-muted-foreground capitalize">{key}</p>
                  <p className="text-sm font-mono text-foreground">{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Metadata */}
          <div className="bg-background rounded-lg p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Last Updated:</span>
              <span className="text-foreground">{formattedLastUpdated}</span>
            </div>
          </div>

          {/* Quick info */}
          <div className="space-y-2">
            <div className="text-sm">
              <span className="text-muted-foreground">Report ID:</span>{" "}
              <span className="text-foreground font-mono">{row.id}</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Type:</span>{" "}
              <span className="text-foreground">{ANALYTICS_REPORT_TYPE_LABELS[row.type]}</span>
            </div>
          </div>

          {/* Phase 0 notice */}
          <p className="text-xs text-muted-foreground text-center italic pt-4">
            Data from mocks - Phase 0
          </p>
        </div>
      </TabsContent>

      {/* Breakdown Tab */}
      <TabsContent value="breakdown" className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <Table2 className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Data Breakdown</span>
        </div>

        {breakdownTable ? (
          <div className="bg-background rounded-lg overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  {breakdownTable.columns.map((col, i) => (
                    <th
                      key={i}
                      className="px-2 py-2 text-left text-muted-foreground font-medium"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdownTable.rows.map((row, rowIdx) => (
                  <tr
                    key={rowIdx}
                    className="border-b border-border last:border-0"
                  >
                    {row.map((cell, cellIdx) => (
                      <td
                        key={cellIdx}
                        className={`px-2 py-2 ${
                          cellIdx === 0 ? "text-foreground" : "text-muted-foreground font-mono"
                        }`}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8">
            <Table2 className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No breakdown data available</p>
          </div>
        )}
      </TabsContent>

      {/* Trends Tab */}
      <TabsContent value="trends" className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Trend Data</span>
        </div>

        {seriesPlaceholder && seriesPlaceholder.length > 0 ? (
          <div className="space-y-3">
            {seriesPlaceholder.map((series, idx) => (
              <div key={idx} className="bg-background rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-foreground">{series.label}</span>
                  <span className="text-xs text-muted-foreground">
                    {series.points} data points
                  </span>
                </div>
                <div className="h-16 bg-surface rounded flex items-center justify-center">
                  <span className="text-xs text-muted-foreground italic">
                    Chart placeholder - Phase 1
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <TrendingUp className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No trend data available</p>
          </div>
        )}

        <p className="text-xs text-muted-foreground text-center italic">
          Charts will be implemented in Phase 1
        </p>
      </TabsContent>
    </Tabs>
  );
}

/**
 * Responsive Analytics Detail Drawer
 *
 * Desktop (>=1280px): Inline drawer that pushes content
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function AnalyticsDetailDrawer({
  report,
  open,
  onClose,
  isLoading,
}: AnalyticsDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const reportTitle = report ? report.row.title : "Report Details";

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
      <DetailDrawer open={open} onClose={onClose} title={reportTitle}>
        {isLoading ? (
          loadingContent
        ) : report ? (
          <AnalyticsDetailContent report={report} />
        ) : (
          <p className="text-muted-foreground text-sm">
            Select a report to view details
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
            {reportTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {isLoading ? (
              loadingContent
            ) : report ? (
              <AnalyticsDetailContent report={report} />
            ) : (
              <p className="text-muted-foreground text-sm">
                Select a report to view details
              </p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
