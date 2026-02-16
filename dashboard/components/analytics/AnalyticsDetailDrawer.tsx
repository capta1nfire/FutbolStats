"use client";

import { useState } from "react";
import { AnalyticsReportDetail, ANALYTICS_REPORT_TYPE_LABELS } from "@/lib/types";
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
import { ReportStatusBadge } from "./ReportStatusBadge";
import { ReportTypeBadge } from "./ReportTypeBadge";
import {
  Clock,
  BarChart3,
  Table2,
  TrendingUp,
  Info,
} from "lucide-react";

/** Tab definitions for analytics detail drawer */
const ANALYTICS_TABS = [
  { id: "summary", icon: <Info />, label: "Summary" },
  { id: "breakdown", icon: <Table2 />, label: "Breakdown" },
  { id: "trends", icon: <TrendingUp />, label: "Trends" },
];

interface AnalyticsDetailDrawerProps {
  report: AnalyticsReportDetail | null;
  open: boolean;
  onClose: () => void;
  isLoading?: boolean;
}

/**
 * Analytics Tab Content - content only, without tabs (for desktop drawer with fixedContent)
 */
function AnalyticsTabContent({ report, activeTab }: { report: AnalyticsReportDetail; activeTab: string }) {
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
    <div className="w-full">
      {/* Summary Tab */}
      {activeTab === "summary" && (
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            {row.status && <ReportStatusBadge status={row.status} />}
            <ReportTypeBadge type={row.type} />
          </div>

          {/* Period */}
          <div className="pt-3 border-t border-border">
            <p className="text-sm text-foreground font-medium">{row.periodLabel}</p>
          </div>

          {/* Metrics */}
          <div className="space-y-3 pt-3 border-t border-border">
            <div className="flex items-center gap-2 text-sm">
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium text-foreground">Key Metrics</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {Object.entries(row.summary).map(([key, value]) => (
                <div key={key} className="bg-background rounded p-2">
                  <p className="text-xs text-muted-foreground capitalize">{key}</p>
                  <p className="text-sm font-mono text-foreground">{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Metadata */}
          <div className="space-y-3 pt-3 border-t border-border">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Last Updated:</span>
              <span className="text-foreground">{formattedLastUpdated}</span>
            </div>
          </div>

          {/* Quick info */}
          <div className="space-y-2 pt-3 border-t border-border">
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
      )}

      {/* Breakdown Tab */}
      {activeTab === "breakdown" && (
        <div className="bg-surface rounded-lg p-4">
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
                  {breakdownTable.rows.map((tableRow, rowIdx) => (
                    <tr
                      key={rowIdx}
                      className="border-b border-border last:border-0"
                    >
                      {tableRow.map((cell, cellIdx) => (
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
        </div>
      )}

      {/* Trends Tab */}
      {activeTab === "trends" && (
        <div className="bg-surface rounded-lg p-4">
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
                  <div className="h-16 bg-surface rounded flex items-center justify-center border border-border">
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

          <p className="text-xs text-muted-foreground text-center italic pt-3">
            Charts will be implemented in Phase 1
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Analytics Detail Content - used for mobile sheet (tabs + content together)
 */
function AnalyticsDetailContentMobile({ report }: { report: AnalyticsReportDetail }) {
  const [activeTab, setActiveTab] = useState("summary");

  return (
    <div className="w-full space-y-3">
      <IconTabs
        tabs={ANALYTICS_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />
      <AnalyticsTabContent report={report} activeTab={activeTab} />
    </div>
  );
}

/**
 * Responsive Analytics Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer (no reflow, ~400px)
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function AnalyticsDetailDrawer({
  report,
  open,
  onClose,
  isLoading,
}: AnalyticsDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const [activeTab, setActiveTab] = useState("summary");
  const reportTitle = report ? report.row.title : "Report Details";

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
        title={reportTitle}
        fixedContent={
          report && !isLoading && (
            <IconTabs
              tabs={ANALYTICS_TABS}
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            />
          )
        }
      >
        {isLoading ? (
          loadingContent
        ) : report ? (
          <AnalyticsTabContent report={report} activeTab={activeTab} />
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
      <SheetContent side="right" className="w-full sm:max-w-md p-0" data-dev-ref="AnalyticsDetailDrawer">
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
              <AnalyticsDetailContentMobile report={report} />
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
