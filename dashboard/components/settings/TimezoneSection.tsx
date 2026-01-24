"use client";

import { useState, useMemo, useEffect } from "react";
import { SettingsSummary } from "@/lib/types";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Clock, Globe2, Check, RotateCcw, Search } from "lucide-react";
import { useRegion } from "@/components/providers/RegionProvider";
import { getSupportedTimeZones, formatCurrentTime } from "@/lib/region";
import { cn } from "@/lib/utils";

interface TimezoneSectionProps {
  settings: SettingsSummary;
}

export function TimezoneSection({ settings }: TimezoneSectionProps) {
  const { region, setRegion, resetRegion } = useRegion();
  const [searchQuery, setSearchQuery] = useState("");
  const [currentTime, setCurrentTime] = useState("");

  // Get supported timezones
  const timezones = useMemo(() => getSupportedTimeZones(), []);

  // Filter timezones based on search
  const filteredTimezones = useMemo(() => {
    if (!searchQuery) return timezones;
    const query = searchQuery.toLowerCase();
    return timezones.filter((tz) => tz.toLowerCase().includes(query));
  }, [timezones, searchQuery]);

  // Update current time every second
  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(formatCurrentTime(region));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, [region]);

  // Handle timezone selection
  const handleTimezoneSelect = (tz: string) => {
    setRegion({ timeZone: tz });
  };

  // Handle hour cycle change
  const handleHourCycleChange = (cycle: "h12" | "h23") => {
    setRegion({ hourCycle: cycle });
  };

  return (
    <div>
      <SettingsSectionHeader
        title="Timezone Settings"
        description="Configure timezone and time format for the dashboard"
      />

      <div className="space-y-6">
        {/* Current Time Preview */}
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Current Time</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={resetRegion}
              className="h-7 text-xs"
            >
              <RotateCcw className="h-3 w-3 mr-1.5" />
              Reset to Browser
            </Button>
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-3xl font-mono text-foreground tabular-nums">
              {currentTime}
            </span>
            <span className="text-sm text-muted-foreground">
              {region.timeZone}
            </span>
          </div>
        </div>

        {/* Hour Format */}
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <Label className="text-sm font-medium">Time Format</Label>
          <div className="flex gap-2">
            <Button
              variant={region.hourCycle === "h12" ? "default" : "outline"}
              size="sm"
              onClick={() => handleHourCycleChange("h12")}
              className="flex-1"
            >
              12-hour (AM/PM)
            </Button>
            <Button
              variant={region.hourCycle === "h23" ? "default" : "outline"}
              size="sm"
              onClick={() => handleHourCycleChange("h23")}
              className="flex-1"
            >
              24-hour
            </Button>
          </div>
        </div>

        {/* Timezone Selection */}
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Globe2 className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Select Timezone</span>
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search timezones..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full h-9 pl-9 pr-3 bg-background border border-border rounded-md text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          {/* Timezone List */}
          <div className="h-64 overflow-y-auto border border-border rounded-md bg-background">
            {filteredTimezones.length === 0 ? (
              <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
                No timezones found
              </div>
            ) : (
              <div className="divide-y divide-border">
                {filteredTimezones.map((tz) => (
                  <button
                    key={tz}
                    onClick={() => handleTimezoneSelect(tz)}
                    className={cn(
                      "w-full px-3 py-2 text-left text-sm hover:bg-surface transition-colors flex items-center justify-between",
                      region.timeZone === tz && "bg-primary/10 text-primary"
                    )}
                  >
                    <span className="font-mono text-xs">{tz}</span>
                    {region.timeZone === tz && (
                      <Check className="h-4 w-4 text-primary" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Server Info */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <div className="flex items-center gap-2 mb-2">
            <Globe2 className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium text-muted-foreground">Server Timezone</span>
          </div>
          <p className="text-sm text-muted-foreground">
            The server operates in UTC. All timestamps in the database and API
            responses are in UTC and converted to your selected timezone for display.
          </p>
        </div>
      </div>
    </div>
  );
}
