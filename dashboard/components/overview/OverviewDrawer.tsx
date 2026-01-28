"use client";

import { useCallback, useMemo } from "react";
import { useOverviewDrawer } from "@/lib/hooks/use-overview-drawer";
import { OverviewTab } from "@/lib/overview-drawer";
import { DetailDrawer } from "@/components/shell";
import { IconTabs, IconTab } from "@/components/ui/icon-tabs";
import {
  Info,
  AlertTriangle,
  Clock,
  HelpCircle,
  ExternalLink,
  TrendingUp,
  List,
} from "lucide-react";

// Panel content components
import {
  OverviewDrawerSentry,
  OverviewDrawerPredictions,
  OverviewDrawerMovement,
  OverviewDrawerOverall,
  OverviewDrawerJobs,
  OverviewDrawerFastpath,
  OverviewDrawerPit,
  OverviewDrawerSota,
  OverviewDrawerBudget,
  OverviewDrawerLlm,
  OverviewDrawerUpcoming,
  OverviewDrawerIncidents,
} from "./drawer-panels";

/** Map tab keys to icons for IconTabs */
const TAB_ICONS: Record<OverviewTab, React.ReactNode> = {
  summary: <Info />,
  issues: <AlertTriangle />,
  timeline: <Clock />,
  missing: <HelpCircle />,
  movers: <TrendingUp />,
  runs: <List />,
  links: <ExternalLink />,
};

interface OverviewDrawerProps {
  className?: string;
}

/**
 * Overview Drawer
 *
 * Uses DetailDrawer (canonical) for consistent styling.
 * - Docked right, no backdrop (click-outside closes)
 * - IconTabs for multi-tab panels
 * - ESC to close
 * - Deep-linkable via URL query params
 */
export function OverviewDrawer({ className }: OverviewDrawerProps) {
  const {
    isOpen,
    panel,
    tab,
    panelMeta,
    tabsMeta,
    setTab,
    closeDrawer,
  } = useOverviewDrawer();

  // Build IconTabs from tabsMeta
  const iconTabs: IconTab[] = useMemo(
    () =>
      tabsMeta.map((t) => ({
        id: t.key,
        icon: TAB_ICONS[t.key as OverviewTab] ?? <Info />,
        label: t.label,
      })),
    [tabsMeta]
  );

  const handleTabChange = useCallback(
    (value: string) => setTab(value as OverviewTab),
    [setTab]
  );

  // Render panel content based on current panel
  const panelContent = useMemo(() => {
    if (!panel || !tab) return null;

    switch (panel) {
      case "sentry":
        return <OverviewDrawerSentry tab={tab} />;
      case "predictions":
        return <OverviewDrawerPredictions tab={tab} />;
      case "movement":
        return <OverviewDrawerMovement tab={tab} />;
      case "overall":
        return <OverviewDrawerOverall tab={tab} />;
      case "jobs":
        return <OverviewDrawerJobs tab={tab} />;
      case "fastpath":
        return <OverviewDrawerFastpath tab={tab} />;
      case "pit":
        return <OverviewDrawerPit tab={tab} />;
      case "sota":
        return <OverviewDrawerSota tab={tab} />;
      case "budget":
        return <OverviewDrawerBudget tab={tab} />;
      case "llm":
        return <OverviewDrawerLlm tab={tab} />;
      case "upcoming":
        return <OverviewDrawerUpcoming tab={tab} />;
      case "incidents":
        return <OverviewDrawerIncidents tab={tab} />;
      default:
        return null;
    }
  }, [panel, tab]);

  return (
    <DetailDrawer
      open={isOpen}
      onClose={closeDrawer}
      title={panelMeta?.title ?? "Details"}
      className={className}
      fixedContent={
        tabsMeta.length > 1 && tab ? (
          <IconTabs
            tabs={iconTabs}
            value={tab}
            onValueChange={handleTabChange}
            showLabels
            className="w-full"
          />
        ) : undefined
      }
    >
      {panelContent}
    </DetailDrawer>
  );
}
