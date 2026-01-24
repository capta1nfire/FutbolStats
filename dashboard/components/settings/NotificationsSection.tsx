"use client";

import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Bell, Mail, MessageSquare, AlertTriangle } from "lucide-react";

export function NotificationsSection() {
  return (
    <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
      <SettingsSectionHeader
        title="Notifications"
        description="Configure alert and notification preferences"
      />

      <div className="space-y-6">
        {/* Email Notifications */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Mail className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Email Notifications</span>
            </div>
            <Badge variant="outline" className="text-muted-foreground">
              Coming soon
            </Badge>
          </div>
          <div className="space-y-3 opacity-50">
            <div className="flex items-center gap-2">
              <Checkbox id="email-incidents" disabled />
              <Label htmlFor="email-incidents" className="text-sm text-muted-foreground">
                Critical incidents
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="email-jobs" disabled />
              <Label htmlFor="email-jobs" className="text-sm text-muted-foreground">
                Job failures
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox id="email-dq" disabled />
              <Label htmlFor="email-dq" className="text-sm text-muted-foreground">
                Data quality alerts
              </Label>
            </div>
          </div>
        </div>

        {/* Slack Notifications */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Slack Integration</span>
            </div>
            <Badge variant="outline" className="text-muted-foreground">
              Coming soon
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground opacity-50">
            Connect a Slack workspace to receive alerts in channels.
          </p>
        </div>

        {/* Alert Thresholds */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Alert Thresholds</span>
            </div>
            <Badge variant="outline" className="text-muted-foreground">
              Coming soon
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground opacity-50">
            Configure custom thresholds for automatic alert generation.
          </p>
        </div>

        {/* Phase 0 Notice */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <div className="flex items-center gap-2 mb-2">
            <Bell className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium text-muted-foreground">Phase 0</span>
          </div>
          <p className="text-sm text-muted-foreground">
            Notification settings are not available in Phase 0. Alerts are
            currently managed through Grafana and Sentry.
          </p>
        </div>
      </div>
    </div>
  );
}
