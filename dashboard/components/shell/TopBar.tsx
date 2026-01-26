"use client";

import { ChevronDown, User, Circle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AlertsBell } from "./AlertsBell";

/**
 * Soccer ball icon (simple filled circle with pentagon pattern suggestion)
 */
function SoccerBallIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" fill="currentColor" />
      <path
        d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"
        fill="currentColor"
        opacity="0.2"
      />
      <path
        d="M12 7l1.5 2.5h3l-1.5 2.5 1.5 2.5h-3L12 17l-1.5-2.5h-3l1.5-2.5-1.5-2.5h3z"
        fill="white"
        opacity="0.9"
      />
    </svg>
  );
}

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
          <SoccerBallIcon className="w-4 h-4 text-primary" />
          Football
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
