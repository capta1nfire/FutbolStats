"use client";

import { SettingsSummary } from "@/lib/types";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Clock, Globe2 } from "lucide-react";

interface TimezoneSectionProps {
  settings: SettingsSummary;
}

export function TimezoneSection({ settings }: TimezoneSectionProps) {
  return (
    <div>
      <SettingsSectionHeader
        title="Timezone Settings"
        description="Configure timezone display for the dashboard"
      />

      <div className="space-y-6">
        {/* Current Timezone */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Display Timezone</span>
          </div>
          <div className="space-y-2">
            <Label htmlFor="timezone" className="text-xs text-muted-foreground">
              All timestamps are stored in UTC and displayed in this timezone
            </Label>
            <Input
              id="timezone"
              value={settings.timezoneDisplay}
              disabled
              className="bg-background max-w-sm"
            />
          </div>
        </div>

        {/* Server Timezone */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center gap-2">
            <Globe2 className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Server Timezone</span>
          </div>
          <p className="text-sm text-muted-foreground">
            The server always operates in UTC. All timestamps in the database,
            API responses, and logs are in UTC.
          </p>
        </div>

        {/* Phase 0 Notice */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <p className="text-sm text-muted-foreground">
            Timezone configuration is not editable in Phase 0. The dashboard
            displays all times in UTC.
          </p>
        </div>
      </div>
    </div>
  );
}
