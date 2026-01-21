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
  title?: string;
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
}: DetailDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  // Handle ESC key to close drawer
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    },
    [onClose]
  );

  // Setup ESC listener and focus management
  useEffect(() => {
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
  }, [open, handleKeyDown, returnFocusRef]);

  if (!open) {
    return null;
  }

  return (
    <aside
      className={cn(
        "bg-sidebar flex flex-col transition-smooth",
        variant === "overlay"
          ? "absolute right-0 top-0 h-full w-[400px] z-30 shadow-elevation-left"
          : "w-[320px]",
        className
      )}
      role="complementary"
      aria-label={title || "Details panel"}
    >
      {/* Header */}
      <div className="h-14 flex items-center px-4 shrink-0 relative">
        <h2 className="text-sm font-semibold text-foreground truncate absolute left-1/2 -translate-x-1/2">
          {title || "Details"}
        </h2>
        <Button
          ref={closeButtonRef}
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 shrink-0 ml-auto"
          aria-label="Close details panel"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content - UniFi style: card inside drawer */}
      <ScrollArea className="flex-1">
        <div className="p-3">
          <div className="bg-surface rounded-lg p-4">
            {children}
          </div>
        </div>
      </ScrollArea>
    </aside>
  );
}
