"use client";

import { useState, useEffect } from "react";
import { Settings } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { DetailDrawer } from "@/components/shell/DetailDrawer";
import { useAdminLeagueMutation } from "@/lib/hooks/use-admin-league-mutation";
import type { LeagueInfo, LeagueTags } from "@/lib/types/football";

interface LeagueSettingsDrawerProps {
  open: boolean;
  onClose: () => void;
  league: LeagueInfo;
}

/**
 * League Settings Drawer
 *
 * Overlay drawer for configuring league-level settings.
 * Currently supports:
 * - use_short_names: Toggle to show abbreviated team names
 *
 * Settings are stored in admin_leagues.tags JSONB field.
 */
export function LeagueSettingsDrawer({
  open,
  onClose,
  league,
}: LeagueSettingsDrawerProps) {
  const mutation = useAdminLeagueMutation();

  // Local state for settings
  const [useShortNames, setUseShortNames] = useState(
    league.tags?.use_short_names ?? false
  );

  // Sync with props when league changes
  useEffect(() => {
    setUseShortNames(league.tags?.use_short_names ?? false);
  }, [league.tags?.use_short_names]);

  const handleToggle = async (checked: boolean) => {
    setUseShortNames(checked);

    const newTags: LeagueTags = {
      ...league.tags,
      use_short_names: checked,
    };

    // P0 ABE: Hook espera { id, body } no { id, tags }
    mutation.mutate({
      id: league.league_id,
      body: { tags: newTags },
    });
  };

  return (
    <DetailDrawer
      open={open}
      onClose={onClose}
      title={
        <div className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          <span>League Settings</span>
        </div>
      }
      variant="overlay"
    >
      <div className="space-y-6">
        {/* League name header */}
        <div className="text-sm text-muted-foreground">
          {league.name} ({league.country})
        </div>

        {/* Settings section */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium">Display</h3>

          {/* use_short_names toggle */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="short-names">Use short names</Label>
              <p className="text-xs text-muted-foreground">
                Show abbreviated team names (e.g. &quot;America&quot; instead of &quot;America de Cali&quot;)
              </p>
            </div>
            <Switch
              id="short-names"
              checked={useShortNames}
              onCheckedChange={handleToggle}
              disabled={mutation.isPending}
            />
          </div>
        </div>

        {/* Status feedback */}
        {mutation.isPending && (
          <p className="text-xs text-muted-foreground">Saving...</p>
        )}
        {mutation.isError && (
          <p className="text-xs text-destructive">
            Error saving settings. Try again.
          </p>
        )}
      </div>
    </DetailDrawer>
  );
}
