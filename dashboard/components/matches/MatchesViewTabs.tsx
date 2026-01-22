"use client";

import { Calendar, History } from "lucide-react";
import { IconTabs } from "@/components/ui/icon-tabs";

export type MatchesView = "upcoming" | "finished";

/** Tab definitions for matches view */
const MATCHES_VIEW_TABS = [
  { id: "upcoming", icon: <Calendar />, label: "Upcoming" },
  { id: "finished", icon: <History />, label: "Finished" },
];

interface MatchesViewTabsProps {
  activeView: MatchesView;
  onViewChange: (view: MatchesView) => void;
}

/**
 * Tab selector for switching between Upcoming and Finished matches
 * Uses the standard IconTabs component with labels shown
 */
export function MatchesViewTabs({ activeView, onViewChange }: MatchesViewTabsProps) {
  return (
    <IconTabs
      tabs={MATCHES_VIEW_TABS}
      value={activeView}
      onValueChange={(value) => onViewChange(value as MatchesView)}
      showLabels
      className="w-full"
    />
  );
}
