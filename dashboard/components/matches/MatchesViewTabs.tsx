"use client";

import { Calendar, History, CalendarRange } from "lucide-react";
import { IconTabs } from "@/components/ui/icon-tabs";

export type MatchesView = "upcoming" | "finished" | "calendar";

/** Tab definitions for matches view */
const MATCHES_VIEW_TABS = [
  { id: "upcoming", icon: <Calendar />, label: "Upcoming" },
  { id: "finished", icon: <History />, label: "Finished" },
  { id: "calendar", icon: <CalendarRange />, label: "Calendar" },
];

interface MatchesViewTabsProps {
  activeView: MatchesView;
  onViewChange: (view: MatchesView) => void;
}

/**
 * Tab selector for switching between Upcoming, Finished, and Calendar views
 * Uses icon-only tabs with tooltips for compact display
 */
export function MatchesViewTabs({ activeView, onViewChange }: MatchesViewTabsProps) {
  return (
    <IconTabs
      tabs={MATCHES_VIEW_TABS}
      value={activeView}
      onValueChange={(value) => onViewChange(value as MatchesView)}
      className="w-full"
    />
  );
}
