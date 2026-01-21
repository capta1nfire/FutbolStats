"use client";

import { Bell, ChevronDown, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export function TopBar() {
  return (
    <header className="h-14 border-b border-border bg-sidebar flex items-center justify-between px-4">
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
          <ChevronDown className="h-4 w-4" />
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
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="relative"
                aria-label="Notifications"
              >
                <Bell className="h-5 w-5 text-muted-foreground" />
                {/* Notification badge */}
                <span className="absolute top-1 right-1 w-2 h-2 bg-error rounded-full" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Notifications</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* Avatar (mock) */}
        <Button
          variant="ghost"
          size="icon"
          className="rounded-full"
          aria-label="User menu"
        >
          <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center">
            <User className="h-4 w-4 text-muted-foreground" />
          </div>
        </Button>
      </div>
    </header>
  );
}
