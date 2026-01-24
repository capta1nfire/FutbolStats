"use client";

import { SettingsSummary, ENVIRONMENT_LABELS } from "@/lib/types";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Server, Globe } from "lucide-react";

interface GeneralSectionProps {
  settings: SettingsSummary;
}

export function GeneralSection({ settings }: GeneralSectionProps) {
  return (
    <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
      <SettingsSectionHeader
        title="General Settings"
        description="Basic configuration for the FutbolStats platform"
      />

      <div className="space-y-6">
        {/* Environment */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Environment</span>
          </div>
          <div className="flex items-center gap-3">
            <Badge
              variant="outline"
              className={
                settings.environment === "prod"
                  ? "bg-success/10 text-success border-success/20"
                  : settings.environment === "staging"
                    ? "bg-warning/10 text-warning border-warning/20"
                    : "text-muted-foreground"
              }
            >
              {ENVIRONMENT_LABELS[settings.environment]}
            </Badge>
            <span className="text-xs text-muted-foreground">
              Last updated: {new Date(settings.lastUpdated).toLocaleString()}
            </span>
          </div>
        </div>

        {/* Narrative Provider */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center gap-2">
            <Globe className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Narrative Provider</span>
          </div>
          <div className="space-y-2">
            <Label htmlFor="narrative-provider" className="text-xs text-muted-foreground">
              LLM provider for match narratives
            </Label>
            <Input
              id="narrative-provider"
              value={settings.narrativeProvider}
              disabled
              className="bg-background max-w-sm"
            />
          </div>
        </div>

        {/* Phase 0 Notice */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <p className="text-sm text-muted-foreground">
            Settings are read-only in Phase 0. Configuration changes are managed
            through environment variables in Railway.
          </p>
        </div>
      </div>
    </div>
  );
}
