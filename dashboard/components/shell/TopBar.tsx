"use client";

import { ChevronDown, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AlertsBell } from "./AlertsBell";

export function TopBar() {
  return (
    <header className="h-[50px] bg-accent flex items-center justify-between px-4 shadow-elevation-down relative z-40">
      {/* Left: Logo + Site Switcher */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">FS</span>
          </div>
          <span className="font-semibold text-foreground">FutbolStats</span>
        </div>

        {/* Site Switcher (mock) */}
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground gap-1"
        >
          Production
          <ChevronDown className="h-4 w-4" strokeWidth={1.5} />
        </Button>
      </div>

      {/* Center: App Tabs (mock) */}
      <nav className="hidden md:flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="text-foreground bg-accent rounded-full px-4"
        >
          Ops Dashboard
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground rounded-full px-4"
        >
          API Docs
        </Button>
      </nav>

      {/* Right: Notifications + Avatar */}
      <div className="flex items-center gap-2">
        <AlertsBell />

        {/* Avatar (mock) */}
        <Button
          variant="ghost"
          size="icon"
          className="rounded-full"
          aria-label="User menu"
        >
          <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center">
            <User className="h-4 w-4 text-muted-foreground" strokeWidth={1.5} />
          </div>
        </Button>
      </div>
    </header>
  );
}
