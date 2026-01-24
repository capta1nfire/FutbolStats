"use client";

import { useEffect, useRef, useCallback } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useOverviewDrawer } from "@/lib/hooks/use-overview-drawer";
import { OverviewTab } from "@/lib/overview-drawer";

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

interface OverviewDrawerProps {
  className?: string;
}

/**
 * Overview Drawer
 *
 * Overlay drawer that shows detailed content for clicked tiles.
 * - Docked right, no backdrop
 * - Tabs for different views
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

  const drawerRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  // Handle ESC key to close
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeDrawer();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, closeDrawer]);

  // Focus management - focus close button when drawer opens
  useEffect(() => {
    if (isOpen && closeButtonRef.current) {
      // Small delay to ensure drawer is rendered
      setTimeout(() => {
        closeButtonRef.current?.focus();
      }, 100);
    }
  }, [isOpen, panel]);

  // Render panel content based on current panel
  const renderPanelContent = useCallback(() => {
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

  if (!isOpen) return null;

  return (
    <div
      ref={drawerRef}
      className={cn(
        "fixed top-14 right-0 bottom-0 w-[400px] z-40",
        "bg-surface border-l border-border",
        "flex flex-col",
        "animate-in slide-in-from-right duration-200",
        className
      )}
      role="dialog"
      aria-modal="false"
      aria-labelledby="drawer-title"
    >
      {/* Header */}
      <div className="h-12 flex items-center justify-between px-4 border-b border-border shrink-0">
        <h2
          id="drawer-title"
          className="text-sm font-semibold text-foreground"
        >
          {panelMeta?.title ?? "Details"}
        </h2>
        <button
          ref={closeButtonRef}
          onClick={closeDrawer}
          className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          aria-label="Close drawer"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Tabs */}
      {tabsMeta.length > 1 && (
        <div className="px-4 py-2 border-b border-border shrink-0">
          <div className="flex gap-1">
            {tabsMeta.map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key as OverviewTab)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-full transition-colors",
                  t.isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {renderPanelContent()}
      </div>
    </div>
  );
}
