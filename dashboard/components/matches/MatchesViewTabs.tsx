"use client";

import { CalendarClock, CalendarCheck2, CalendarDays } from "lucide-react";
import { IconTabs } from "@/components/ui/icon-tabs";

export type MatchesView = "upcoming" | "finished" | "calendar";

/** Tab definitions for matches view */
const MATCHES_VIEW_TABS = [
  { id: "calendar", icon: <CalendarDays />, label: "Calendar" },
  { id: "finished", icon: <CalendarCheck2 />, label: "Finished" },
  { id: "upcoming", icon: <CalendarClock />, label: "Upcoming" },
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
