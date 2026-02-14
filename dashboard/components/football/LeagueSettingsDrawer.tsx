"use client";

import { useState, useEffect, useCallback } from "react";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { DetailDrawer } from "@/components/shell/DetailDrawer";
import { useAdminLeagueMutation } from "@/lib/hooks/use-admin-league-mutation";
import { ExternalLink, Loader2 } from "lucide-react";
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
 * Supports:
 * - display_name: Commercial league name (e.g. "Liga BetPlay DIMAYOR")
 * - logo_url: League logo URL
 * - wikipedia_url: Wikipedia reference link
 * - use_short_names: Toggle to show abbreviated team names
 *
 * Settings are stored in admin_leagues columns + tags JSONB.
 */
export function LeagueSettingsDrawer({
  open,
  onClose,
  league,
}: LeagueSettingsDrawerProps) {
  const mutation = useAdminLeagueMutation();

  // Local state for toggle
  const [useShortNames, setUseShortNames] = useState(
    league.tags?.use_short_names ?? false
  );

  // Local state for text fields
  const [displayName, setDisplayName] = useState(league.display_name ?? "");
  const [logoUrl, setLogoUrl] = useState(league.logo_url ?? "");
  const [wikipediaUrl, setWikipediaUrl] = useState(league.wikipedia_url ?? "");

  // Track dirty state
  const isDirty =
    displayName !== (league.display_name ?? "") ||
    logoUrl !== (league.logo_url ?? "") ||
    wikipediaUrl !== (league.wikipedia_url ?? "");

  // Sync with props when league changes
  useEffect(() => {
    setUseShortNames(league.tags?.use_short_names ?? false);
    setDisplayName(league.display_name ?? "");
    setLogoUrl(league.logo_url ?? "");
    setWikipediaUrl(league.wikipedia_url ?? "");
  }, [league.league_id, league.tags?.use_short_names, league.display_name, league.logo_url, league.wikipedia_url]);

  const handleToggle = useCallback(
    (checked: boolean) => {
      setUseShortNames(checked);
      const newTags: LeagueTags = {
        ...league.tags,
        use_short_names: checked,
      };
      mutation.mutate({
        id: league.league_id,
        body: { tags: newTags },
      });
    },
    [league.league_id, league.tags, mutation]
  );

  const handleSaveMetadata = useCallback(() => {
    mutation.mutate({
      id: league.league_id,
      body: {
        display_name: displayName.trim() || null,
        logo_url: logoUrl.trim() || null,
        wikipedia_url: wikipediaUrl.trim() || null,
      },
    });
  }, [league.league_id, displayName, logoUrl, wikipediaUrl, mutation]);

  return (
    <DetailDrawer
      open={open}
      onClose={onClose}
      title="League Settings"
      variant="overlay"
      className="shadow-none"
    >
      <div className="space-y-4">
        {/* League info header */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          {league.logo_url && (
            <img
              src={league.logo_url}
              alt=""
              width={20}
              height={20}
              className="inline-block"
            />
          )}
          {league.name} ({league.country})
        </div>

        {/* Metadata Card */}
        <div className="bg-muted/50 rounded-lg p-3 space-y-3">
          <div className="text-xs text-muted-foreground">League Identity</div>

          {/* Display name */}
          <div className="space-y-1">
            <Label htmlFor="display-name" className="text-xs text-muted-foreground">
              Commercial Name
            </Label>
            <Input
              id="display-name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder={league.name}
              className="h-8 text-sm"
            />
          </div>

          {/* Logo URL */}
          <div className="space-y-1">
            <Label htmlFor="logo-url" className="text-xs text-muted-foreground">
              Logo URL
            </Label>
            <div className="flex items-center gap-2">
              {logoUrl && (
                <img
                  src={logoUrl}
                  alt=""
                  width={24}
                  height={24}
                  className="shrink-0"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              )}
              <Input
                id="logo-url"
                value={logoUrl}
                onChange={(e) => setLogoUrl(e.target.value)}
                placeholder="https://..."
                className="h-8 text-sm flex-1"
              />
            </div>
          </div>

          {/* Wikipedia URL */}
          <div className="space-y-1">
            <Label htmlFor="wikipedia-url" className="text-xs text-muted-foreground">
              Wikipedia
            </Label>
            <div className="flex items-center gap-2">
              <Input
                id="wikipedia-url"
                value={wikipediaUrl}
                onChange={(e) => setWikipediaUrl(e.target.value)}
                placeholder="https://es.wikipedia.org/wiki/..."
                className="h-8 text-sm flex-1"
              />
              {wikipediaUrl && (
                <a
                  href={wikipediaUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground shrink-0"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              )}
            </div>
          </div>

          {/* Save button */}
          {isDirty && (
            <Button
              size="sm"
              onClick={handleSaveMetadata}
              disabled={mutation.isPending}
              className="w-full h-8 text-xs"
            >
              {mutation.isPending ? (
                <Loader2 className="w-3 h-3 animate-spin mr-1" />
              ) : null}
              Save Changes
            </Button>
          )}
        </div>

        {/* Display Settings Card */}
        <div className="bg-muted/50 rounded-lg p-3 space-y-3">
          <div className="text-xs text-muted-foreground">Display</div>

          {/* use_short_names toggle */}
          <div className="flex items-center justify-between gap-4">
            <div className="space-y-0.5">
              <Label htmlFor="short-names" className="text-sm text-foreground">
                Use short names
              </Label>
              <p className="text-xs text-muted-foreground">
                Show abbreviated team names
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
