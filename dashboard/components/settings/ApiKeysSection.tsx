"use client";

import { SettingsSummary, API_KEY_STATUS_LABELS } from "@/lib/types";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Key, Shield, EyeOff, CheckCircle, XCircle, AlertTriangle } from "lucide-react";

interface ApiKeysSectionProps {
  settings: SettingsSummary;
}

export function ApiKeysSection({ settings }: ApiKeysSectionProps) {
  const getStatusIcon = (status: typeof settings.apiFootballKeyStatus) => {
    switch (status) {
      case "configured":
        return <CheckCircle className="h-4 w-4 text-success" />;
      case "invalid":
        return <XCircle className="h-4 w-4 text-error" />;
      case "missing":
        return <AlertTriangle className="h-4 w-4 text-warning" />;
    }
  };

  const getStatusBadgeClass = (status: typeof settings.apiFootballKeyStatus) => {
    switch (status) {
      case "configured":
        return "bg-success/10 text-success border-success/20";
      case "invalid":
        return "bg-error/10 text-error border-error/20";
      case "missing":
        return "bg-warning/10 text-warning border-warning/20";
    }
  };

  return (
    <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
      <SettingsSectionHeader
        title="API Keys"
        description="Manage external service credentials"
      />

      <div className="space-y-6">
        {/* API-Football Key */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Key className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">API-Football</span>
            </div>
            <div className="flex items-center gap-2">
              {getStatusIcon(settings.apiFootballKeyStatus)}
              <Badge variant="outline" className={getStatusBadgeClass(settings.apiFootballKeyStatus)}>
                {API_KEY_STATUS_LABELS[settings.apiFootballKeyStatus]}
              </Badge>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="api-football-key" className="text-xs text-muted-foreground">
              Used for fetching live scores, fixtures, and statistics
            </Label>
            <div className="relative">
              <Input
                id="api-football-key"
                type="password"
                value="****************************"
                disabled
                className="bg-background max-w-md pr-10"
              />
              <EyeOff className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            </div>
          </div>
        </div>

        {/* RunPod Key */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Key className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">RunPod</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-success" />
              <Badge variant="outline" className="bg-success/10 text-success border-success/20">
                Configured
              </Badge>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="runpod-key" className="text-xs text-muted-foreground">
              Serverless LLM endpoint for narrative generation (backup)
            </Label>
            <div className="relative">
              <Input
                id="runpod-key"
                type="password"
                value="****************************"
                disabled
                className="bg-background max-w-md pr-10"
              />
              <EyeOff className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            </div>
          </div>
        </div>

        {/* Gemini Key */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Key className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Gemini</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-success" />
              <Badge variant="outline" className="bg-success/10 text-success border-success/20">
                Configured
              </Badge>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="gemini-key" className="text-xs text-muted-foreground">
              Primary LLM provider for match narratives
            </Label>
            <div className="relative">
              <Input
                id="gemini-key"
                type="password"
                value="****************************"
                disabled
                className="bg-background max-w-md pr-10"
              />
              <EyeOff className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            </div>
          </div>
        </div>

        {/* Security Notice */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium text-muted-foreground">Security</span>
          </div>
          <p className="text-sm text-muted-foreground">
            API keys are stored securely as environment variables in Railway and
            are never exposed in the dashboard. Keys shown here are masked for
            security.
          </p>
        </div>
      </div>
    </div>
  );
}
