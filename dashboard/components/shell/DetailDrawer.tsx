"use client";

import { ReactNode } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface DetailDrawerProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children?: ReactNode;
  className?: string;
}

/**
 * Inline Detail Drawer
 *
 * Desktop (>=1280px): Renders inline as part of the grid, pushing content.
 * NO overlay, NO modal behavior. Table remains interactive.
 *
 * Mobile (<1280px): Should use Sheet component instead (handled by parent).
 */
export function DetailDrawer({
  open,
  onClose,
  title,
  children,
  className,
}: DetailDrawerProps) {
  if (!open) {
    return null;
  }

  return (
    <aside
      className={cn(
        "w-[320px] border-l border-border bg-surface flex flex-col transition-smooth",
        className
      )}
    >
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-4 border-b border-border shrink-0">
        <h2 className="text-sm font-semibold text-foreground truncate">
          {title || "Details"}
        </h2>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 shrink-0"
          aria-label="Close details panel"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-4">{children}</div>
      </ScrollArea>
    </aside>
  );
}
