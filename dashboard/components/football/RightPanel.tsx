"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { ShieldUser, Settings, BarChart3, X } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TeamPanelContent } from "./TeamDrawer";
import { LeagueSettingsPanelContent } from "./LeagueSettingsDrawer";
import { CoverageDrawerContent } from "./LeagueDetail";
import type { LeagueInfo } from "@/lib/types/football";

export type PanelTabId = "team" | "settings" | "coverage";

interface PanelTabDef {
  id: PanelTabId;
  icon: ReactNode;
  label: string;
  closeable: boolean;
}

const TAB_DEFS: Record<PanelTabId, PanelTabDef> = {
  team:     { id: "team",     icon: <ShieldUser className="w-3.5 h-3.5" />, label: "Team",     closeable: false },
  settings: { id: "settings", icon: <Settings className="w-3.5 h-3.5" />,   label: "Settings", closeable: true },
  coverage: { id: "coverage", icon: <BarChart3 className="w-3.5 h-3.5" />,  label: "Coverage", closeable: true },
};

interface RightPanelProps {
  tabs: PanelTabId[];
  activeTab: PanelTabId;
  onTabChange: (id: PanelTabId) => void;
  onTabClose: (id: PanelTabId) => void;
  // Team tab data
  teamId: number | null;
  // League settings tab data
  league?: LeagueInfo | null;
  // Coverage tab data
  leagueId: number | null;
}

/**
 * Right Panel — UniFi-style tabbed sidebar.
 *
 * Always shows "Team" tab (persistent, no close).
 * "Settings" and "Coverage" tabs open dynamically with X to close.
 */
export function RightPanel({
  tabs,
  activeTab,
  onTabChange,
  onTabClose,
  teamId,
  league,
  leagueId,
}: RightPanelProps) {
  return (
    <aside
      className="w-[400px] border-l border-border shadow-drawer-left bg-sidebar flex flex-col"
      role="complementary"
      aria-label="Details panel"
    >
      {/* Tab bar */}
      {tabs.length > 1 && (
        <div className="flex items-center h-10 px-2 gap-1 shrink-0 border-b border-border">
          {tabs.map((tabId) => {
            const def = TAB_DEFS[tabId];
            const isActive = activeTab === tabId;
            return (
              <button
                key={tabId}
                onClick={() => onTabChange(tabId)}
                className={cn(
                  "inline-flex items-center gap-1.5 px-2.5 h-7 rounded-md text-xs font-medium",
                  "transition-colors duration-150",
                  isActive
                    ? "bg-accent text-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                )}
              >
                {def.icon}
                <span>{def.label}</span>
                {def.closeable && (
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      onTabClose(tabId);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.stopPropagation();
                        onTabClose(tabId);
                      }
                    }}
                    className="ml-0.5 rounded-sm p-0.5 opacity-50 hover:opacity-100 hover:bg-muted transition-opacity"
                    aria-label={`Close ${def.label}`}
                  >
                    <X className="w-3 h-3" />
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Content area */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {activeTab === "team" && (
          <TeamPanelContent teamId={teamId} />
        )}
        {activeTab === "settings" && league && (
          <LeagueSettingsPanelContent league={league} />
        )}
        {activeTab === "coverage" && leagueId != null && (
          <CoveragePanelContent leagueId={leagueId} />
        )}
      </div>
    </aside>
  );
}

/**
 * Coverage panel content — header + scroll + CoverageDrawerContent
 */
function CoveragePanelContent({ leagueId }: { leagueId: number }) {
  return (
    <>
      <div className="h-14 flex items-center justify-center px-4 shrink-0">
        <h2 className="text-sm font-semibold text-foreground truncate">
          Data Coverage
        </h2>
      </div>
      <ScrollArea className="flex-1 min-h-0">
        <div className="px-3 pt-3 pb-3">
          <CoverageDrawerContent leagueId={leagueId} />
        </div>
      </ScrollArea>
    </>
  );
}
