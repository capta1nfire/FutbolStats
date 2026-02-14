"use client";

import { ReactNode, useEffect, useRef, useCallback } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export type DrawerVariant = "inline" | "overlay";

interface DetailDrawerProps {
  open: boolean;
  onClose: () => void;
  title?: ReactNode;
  children?: ReactNode;
  className?: string;
  /** Ref to element that should receive focus when drawer closes */
  returnFocusRef?: React.RefObject<HTMLElement | null>;
  /**
   * Drawer behavior variant:
   * - "inline": pushes content (original behavior)
   * - "overlay": absolute positioned, overlays content (UniFi style)
   * Default: "overlay" (new default for desktop)
   */
  variant?: DrawerVariant;
  /**
   * Fixed content rendered above the scroll area (e.g., tabs).
   * This content won't scroll and tooltips won't be clipped.
   */
  fixedContent?: ReactNode;
  /**
   * Persistent mode: always visible, no close button, no ESC/backdrop.
   * Used for fixed sidebars like Team 360 in Football section.
   */
  persistent?: boolean;
}

/**
 * Detail Drawer
 *
 * Two variants:
 * - "overlay" (default): Absolute positioned, overlays content (UniFi style).
 *   No backdrop, no modal behavior. Table remains interactive where not covered.
 * - "inline": Renders inline as part of the grid, pushing content.
 *
 * Mobile (<1280px): Should use Sheet component instead (handled by parent).
 *
 * UX Features:
 * - ESC key closes drawer
 * - Focus returns to returnFocusRef element when closed
 */
export function DetailDrawer({
  open,
  onClose,
  title,
  children,
  className,
  returnFocusRef,
  variant = "overlay",
  fixedContent,
  persistent = false,
}: DetailDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  // Handle ESC key to close drawer (skip in persistent mode)
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (persistent) return;
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    },
    [onClose, persistent]
  );

  // Setup ESC listener and focus management (skip in persistent mode)
  useEffect(() => {
    if (persistent) return;

    if (open) {
      // Store currently focused element before drawer opened
      previouslyFocusedRef.current = document.activeElement as HTMLElement;

      // Add ESC listener
      document.addEventListener("keydown", handleKeyDown);

      // Focus close button after short delay (allow render)
      const timer = setTimeout(() => {
        closeButtonRef.current?.focus();
      }, 50);

      return () => {
        document.removeEventListener("keydown", handleKeyDown);
        clearTimeout(timer);
      };
    } else {
      // Drawer closed - return focus
      const targetElement = returnFocusRef?.current || previouslyFocusedRef.current;
      if (targetElement && typeof targetElement.focus === "function") {
        targetElement.focus();
      }
    }
  }, [open, handleKeyDown, returnFocusRef, persistent]);

  // In persistent mode, always render. Otherwise, respect open prop.
  if (!persistent && !open) {
    return null;
  }

  return (
    <>
      {/* Invisible backdrop — click outside closes drawer (overlay only, not persistent) */}
      {variant === "overlay" && !persistent && (
        <div
          className="absolute inset-0 z-20"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={cn(
          "bg-sidebar flex flex-col transition-smooth overflow-hidden",
          persistent
            ? "w-[400px] border-l border-border shadow-drawer-left"
            : variant === "overlay"
              ? "absolute right-0 top-0 h-full w-[400px] z-30 shadow-drawer-left"
              : "w-[320px]",
          className
        )}
        role="complementary"
        aria-label="Details panel"
      >
      {/* Header */}
      <div className="h-14 flex items-center justify-center px-4 shrink-0 relative">
        <h2 className="text-sm font-semibold text-foreground truncate">
          {title || "Details"}
        </h2>
        {!persistent && (
          <Button
            ref={closeButtonRef}
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 shrink-0 absolute right-4"
            aria-label="Close details panel"
          >
            <X className="h-4 w-4" strokeWidth={1.5} />
          </Button>
        )}
      </div>

      {/* Fixed content (tabs, etc.) - outside ScrollArea to prevent tooltip clipping */}
      {fixedContent && (
        <div className="px-3 pt-3 shrink-0">
          {fixedContent}
        </div>
      )}

      {/* Scrollable content */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="px-3 pt-3 pb-3">
          {children}
        </div>
      </ScrollArea>
    </aside>
    </>
  );
}
