"use client";

import { useState } from "react";
import { useFeatureFlags } from "@/lib/hooks";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { FeatureFlagsTable } from "./FeatureFlagsTable";
import { SearchInput } from "@/components/ui/search-input";
import { ToggleLeft } from "lucide-react";

export function FeatureFlagsSection() {
  const [searchValue, setSearchValue] = useState("");

  const {
    data: flags = [],
    isLoading,
    error,
    refetch,
  } = useFeatureFlags(searchValue ? { search: searchValue } : undefined);

  const enabledCount = flags.filter((f) => f.enabled).length;
  const disabledCount = flags.filter((f) => !f.enabled).length;

  return (
    <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
      <SettingsSectionHeader
        title="Feature Flags"
        description="Toggle features on or off across the platform"
      />

      <div className="space-y-4">
        {/* Summary */}
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <ToggleLeft className="h-4 w-4 text-success" />
            <span className="text-muted-foreground">
              {enabledCount} enabled
            </span>
          </div>
          <div className="flex items-center gap-2">
            <ToggleLeft className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">
              {disabledCount} disabled
            </span>
          </div>
        </div>

        {/* Search */}
        <div className="max-w-sm">
          <SearchInput
            placeholder="Search flags..."
            value={searchValue}
            onChange={setSearchValue}
            className="bg-background"
          />
        </div>

        {/* Table */}
        <FeatureFlagsTable
          data={flags}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
        />

        {/* Phase 0 Notice */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <p className="text-sm text-muted-foreground">
            Feature flags are read-only in Phase 0. Toggles are managed through
            environment variables and require a redeploy to change.
          </p>
        </div>
      </div>
    </div>
  );
}
