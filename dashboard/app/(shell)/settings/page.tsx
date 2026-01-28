"use client";

import { useState } from "react";
import { useSettingsSummary } from "@/lib/hooks";
import { SettingsSection } from "@/lib/types";
import {
  SettingsNav,
  GeneralSection,
  TimezoneSection,
  NotificationsSection,
  ApiKeysSection,
  ModelVersionsSection,
  FeatureFlagsSection,
  UsersSection,
  IaFeaturesSection,
} from "@/components/settings";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";

/**
 * Settings Content Renderer
 *
 * Renders the appropriate section based on the active section
 */
function SettingsContent({
  section,
  isLoading,
  error,
  onRetry,
}: {
  section: SettingsSection;
  isLoading: boolean;
  error: Error | null;
  onRetry: () => void;
}) {
  const { data: settings } = useSettingsSummary();

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader size="md" />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center gap-4 text-center">
          <AlertTriangle className="h-8 w-8 text-error" />
          <div>
            <p className="text-sm text-foreground mb-1">Failed to load settings</p>
            <p className="text-xs text-muted-foreground">{error.message}</p>
          </div>
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // No settings data
  if (!settings) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-sm text-muted-foreground">No settings data available</p>
      </div>
    );
  }

  // Render section content
  switch (section) {
    case "general":
      return <GeneralSection settings={settings} />;
    case "timezone":
      return <TimezoneSection settings={settings} />;
    case "notifications":
      return <NotificationsSection />;
    case "api_keys":
      return <ApiKeysSection settings={settings} />;
    case "model_versions":
      return <ModelVersionsSection settings={settings} />;
    case "feature_flags":
      return <FeatureFlagsSection />;
    case "users":
      return <UsersSection />;
    case "ia_features":
      return <IaFeaturesSection />;
    default:
      return (
        <div className="flex items-center justify-center h-64">
          <p className="text-sm text-muted-foreground">Unknown section</p>
        </div>
      );
  }
}

/**
 * Settings Page
 *
 * Two-column layout with navigation sidebar and content area
 * All settings are read-only in Phase 0
 */
export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState<SettingsSection>("general");

  const {
    isLoading,
    error,
    refetch,
  } = useSettingsSummary();

  return (
    <div className="h-full flex overflow-hidden">
      {/* Settings Navigation */}
      <SettingsNav
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="p-6 max-w-3xl">
            <SettingsContent
              section={activeSection}
              isLoading={isLoading}
              error={error}
              onRetry={() => refetch()}
            />
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
