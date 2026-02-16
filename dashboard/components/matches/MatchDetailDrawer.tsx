"use client";

import { useState } from "react";
import { MatchSummary } from "@/lib/types";
import { useIsDesktop, useTeamLogos } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import { Loader } from "@/components/ui/loader";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { IconTabs } from "@/components/ui/icon-tabs";
import { ScrollArea } from "@/components/ui/scroll-area";

// Import shared components from MatchDetailContent
import {
  MatchHeader,
  MatchTabContent,
  MatchDetailContent,
  MATCH_TABS,
} from "./MatchDetailContent";

interface MatchDetailDrawerProps {
  match: MatchSummary | null;
  /** True when match is being fetched for deep-link / pagination fallback */
  isLoading?: boolean;
  open: boolean;
  onClose: () => void;
}

/**
 * Responsive Match Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer (no reflow, ~400px)
 * Mobile/Tablet (<1280px): Sheet overlay
 *
 * Uses shared components from MatchDetailContent for consistency
 * with the Overview drawer match panel.
 */
export function MatchDetailDrawer({
  match,
  isLoading = false,
  open,
  onClose,
}: MatchDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const [activeTab, setActiveTab] = useState("overview");
  const { getLogoUrl } = useTeamLogos();
  const matchTitle = match ? `Match ${match.id}` : "Match Details";

  // Desktop: overlay drawer with tabs in fixedContent (prevents tooltip clipping)
  if (isDesktop) {
    return (
      <DetailDrawer
        open={open}
        onClose={onClose}
        title={matchTitle}
        fixedContent={
          match && (
            <div className="space-y-3">
              <MatchHeader match={match} getLogoUrl={getLogoUrl} />
              <IconTabs
                tabs={MATCH_TABS}
                value={activeTab}
                onValueChange={setActiveTab}
                className="w-full"
              />
            </div>
          )
        }
      >
        {match ? (
          <MatchTabContent match={match} activeTab={activeTab} getLogoUrl={getLogoUrl} />
        ) : isLoading ? (
          <div className="h-full flex items-center justify-center py-10">
            <Loader size="md" />
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">Select a match to view details</p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0" data-dev-ref="MatchDetailDrawer">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {matchTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {match ? (
              <MatchDetailContent match={match} />
            ) : isLoading ? (
              <div className="h-full flex items-center justify-center py-10">
                <Loader size="md" />
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Select a match to view details</p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
