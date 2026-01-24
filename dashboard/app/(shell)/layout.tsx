"use client";

import { ReactNode } from "react";
import { TopBar, IconSidebar } from "@/components/shell";
import { Toaster } from "@/components/ui/toaster";

interface ShellLayoutProps {
  children: ReactNode;
}

/**
 * Shell Layout - 4 Zone Grid
 *
 * Layout structure:
 * ┌─────────────────────────────────────────────────────┐
 * │                     TopBar                          │
 * ├──────┬─────────────────────────────────────────────┤
 * │ Icon │                                             │
 * │ Side │              Main Content                   │
 * │ bar  │    (FilterPanel + Table + Drawer)           │
 * │      │         handled by pages                    │
 * └──────┴─────────────────────────────────────────────┘
 *
 * FilterPanel and DetailDrawer are managed at page level
 * because they vary per section and need URL state access.
 */
export default function ShellLayout({ children }: ShellLayoutProps) {
  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Row 1: TopBar - full width */}
      <TopBar />

      {/* Row 2: IconSidebar + Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Column 1: IconSidebar - fixed width */}
        <IconSidebar />

        {/* Column 2+: Main content area (FilterPanel + Table + Drawer managed by pages) */}
        <main className="flex-1 overflow-hidden">{children}</main>
      </div>
      <Toaster />
    </div>
  );
}
