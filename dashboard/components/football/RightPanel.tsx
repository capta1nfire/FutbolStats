"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { ShieldUser, Settings, BarChart3, User, X } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TeamPanelContent } from "./TeamDrawer";
import { LeagueSettingsPanelContent } from "./LeagueSettingsDrawer";
import { CoverageDrawerContent } from "./LeagueDetail";
import { PlayerDetail } from "./PlayerDetail";
import type { LeagueInfo } from "@/lib/types/football";
import type { TeamSquadPlayerSeasonStats } from "@/lib/types/squad";

export type PanelTabId = "team" | "settings" | "coverage" | "player";

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
  player:   { id: "player",   icon: <User className="w-3.5 h-3.5" />,       label: "Player",   closeable: true },
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
  // Player tab data
  selectedPlayer?: TeamSquadPlayerSeasonStats | null;
  teamName?: string;
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
  selectedPlayer,
  teamName,
}: RightPanelProps) {
  return (
    <aside
      data-dev-ref="RightPanel"
      className="w-[400px] border-l border-border shadow-drawer-left bg-sidebar flex flex-col"
      role="complementary"
      aria-label="Details panel"
    >
      {/* Tab bar */}
      {tabs.length > 1 && (
        <nav
          role="tablist"
          className="inline-flex items-center bg-surface rounded-lg p-1 h-10 min-h-9 mx-2 mt-2 shrink-0"
        >
          {tabs.map((tabId) => {
            const def = TAB_DEFS[tabId];
            const isActive = activeTab === tabId;
            const label = tabId === "player" && selectedPlayer
              ? selectedPlayer.player_name.trim().split(/\s+/).slice(-1)[0] || "Player"
              : def.label;
            return (
              <div key={tabId} className="flex-1">
                <button
                  role="tab"
                  aria-selected={isActive}
                  onClick={() => onTabChange(tabId)}
                  className={cn(
                    "inline-flex items-center justify-center gap-1.5 w-full h-8 min-h-8 px-2 rounded-md",
                    "transition-[background-color,color] duration-150",
                    "focus-visible:outline focus-visible:outline-1 focus-visible:outline-primary",
                    "bg-transparent text-muted-foreground",
                    !isActive && "hover:text-primary",
                    isActive && "bg-accent text-primary"
                  )}
                  style={{ transitionTimingFunction: "cubic-bezier(0.7, 0, 0.3, 1)" }}
                >
                  <span className="flex items-center [&>svg]:w-3.5 [&>svg]:h-3.5">
                    {def.icon}
                  </span>
                  <span className="text-xs font-medium">{label}</span>
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
              </div>
            );
          })}
        </nav>
      )}

      {/* Content area */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {activeTab === "team" && (
          <div data-dev-ref="RightPanel:TeamTab" className="flex-1 min-h-0 flex flex-col overflow-hidden">
            <TeamPanelContent teamId={teamId} />
          </div>
        )}
        {activeTab === "settings" && league && (
          <div data-dev-ref="RightPanel:SettingsTab" className="flex-1 min-h-0 flex flex-col overflow-hidden">
            <LeagueSettingsPanelContent league={league} />
          </div>
        )}
        {activeTab === "coverage" && leagueId != null && (
          <div data-dev-ref="RightPanel:CoverageTab" className="flex-1 min-h-0 flex flex-col overflow-hidden">
            <CoveragePanelContent leagueId={leagueId} />
          </div>
        )}
        {activeTab === "player" && selectedPlayer && (
          <ScrollArea data-dev-ref="RightPanel:PlayerTab" className="flex-1 min-h-0">
            <PlayerDetail player={selectedPlayer} teamName={teamName} />
          </ScrollArea>
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
