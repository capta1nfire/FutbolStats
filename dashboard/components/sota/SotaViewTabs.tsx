"use client";

import { Sparkles, Settings } from "lucide-react";
import { IconTabs } from "@/components/ui/icon-tabs";

export type SotaView = "enrichment" | "features";

/** Tab definitions for SOTA view */
const SOTA_VIEW_TABS = [
  { id: "enrichment", icon: <Sparkles />, label: "Enrichment" },
  { id: "features", icon: <Settings />, label: "Features" },
];

interface SotaViewTabsProps {
  activeView: SotaView;
  onViewChange: (view: SotaView) => void;
}

/**
 * Tab selector for switching between Enrichment and Features views
 * Uses icon-only tabs with tooltips for compact display
 */
export function SotaViewTabs({ activeView, onViewChange }: SotaViewTabsProps) {
  return (
    <IconTabs
      tabs={SOTA_VIEW_TABS}
      value={activeView}
      onValueChange={(value) => onViewChange(value as SotaView)}
      className="w-full"
    />
  );
}
