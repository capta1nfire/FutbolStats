"use client";

import { useState, useEffect, useCallback } from "react";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { DetailDrawer } from "@/components/shell/DetailDrawer";
import { ScrollArea } from "@/components/ui/scroll-area";
import { IconTabs } from "@/components/ui/icon-tabs";
import { useAdminLeagueMutation } from "@/lib/hooks/use-admin-league-mutation";
import { ExternalLink, Info, Image as ImageIcon, Loader2, Settings } from "lucide-react";
import type { LeagueInfo, LeagueTags } from "@/lib/types/football";

const LEAGUE_TABS = [
  { id: "overview", icon: <Info />, label: "Overview" },
  { id: "multimedia", icon: <ImageIcon />, label: "Multimedia" },
  { id: "settings", icon: <Settings />, label: "Settings" },
];

/**
 * League Settings Panel Content — standalone content for the right panel.
 * Contains league header, IconTabs, and tab content.
 */
export function LeagueSettingsPanelContent({ league }: { league: LeagueInfo }) {
  const mutation = useAdminLeagueMutation();
  const [activeTab, setActiveTab] = useState("overview");

  const [useShortNames, setUseShortNames] = useState(
    league.tags?.use_short_names ?? false
  );
  const [displayName, setDisplayName] = useState(league.display_name ?? "");
  const [logoUrl, setLogoUrl] = useState(league.logo_url ?? "");
  const [wikipediaUrl, setWikipediaUrl] = useState(league.wikipedia_url ?? "");
  const [seasonStartMonth, setSeasonStartMonth] = useState(
    league.season_start_month ?? 8
  );

  const isDirtyMetadata =
    displayName !== (league.display_name ?? "") ||
    wikipediaUrl !== (league.wikipedia_url ?? "");

  const isDirtyLogo = logoUrl !== (league.logo_url ?? "");

  useEffect(() => {
    setUseShortNames(league.tags?.use_short_names ?? false);
    setDisplayName(league.display_name ?? "");
    setLogoUrl(league.logo_url ?? "");
    setWikipediaUrl(league.wikipedia_url ?? "");
    setSeasonStartMonth(league.season_start_month ?? 8);
  }, [league.league_id, league.tags?.use_short_names, league.display_name, league.logo_url, league.wikipedia_url, league.season_start_month]);

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

  const handleSeasonStartChange = useCallback(
    (month: number) => {
      setSeasonStartMonth(month);
      mutation.mutate({
        id: league.league_id,
        body: { season_start_month: month },
      });
    },
    [league.league_id, mutation]
  );

  const handleSaveMetadata = useCallback(() => {
    mutation.mutate({
      id: league.league_id,
      body: {
        display_name: displayName.trim() || null,
        wikipedia_url: wikipediaUrl.trim() || null,
      },
    });
  }, [league.league_id, displayName, wikipediaUrl, mutation]);

  const handleSaveLogo = useCallback(() => {
    mutation.mutate({
      id: league.league_id,
      body: { logo_url: logoUrl.trim() || null },
    });
  }, [league.league_id, logoUrl, mutation]);

  return (
    <>
      {/* League header */}
      <div className="px-4 pt-4 pb-2 shrink-0 space-y-3">
        <div className="flex items-center gap-3">
          {league.logo_url && (
            <img
              src={league.logo_url}
              alt=""
              width={32}
              height={32}
              className="shrink-0"
            />
          )}
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-foreground truncate">
              {league.display_name || league.name}
            </h2>
            <p className="text-xs text-muted-foreground">{league.country}</p>
          </div>
        </div>

        {/* Tabs */}
        <IconTabs
          tabs={LEAGUE_TABS}
          value={activeTab}
          onValueChange={setActiveTab}
          className="w-full"
        />
      </div>

      {/* Scrollable tab content */}
      <ScrollArea data-dev-ref="LeagueSettings" className="flex-1 min-h-0">
        <div className="px-3 pt-3 pb-3 space-y-4">
          {/* Overview tab */}
          {activeTab === "overview" && (
            <>
              {/* League Identity */}
              <div className="bg-muted/50 rounded-lg p-3 space-y-3">
                <div className="text-xs text-muted-foreground">League Identity</div>

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

                {isDirtyMetadata && (
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
            </>
          )}

          {/* Multimedia tab */}
          {activeTab === "multimedia" && (
            <>
              <div className="bg-muted/50 rounded-lg p-3 space-y-3">
                <div className="text-xs text-muted-foreground">Logo</div>

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

                {isDirtyLogo && (
                  <Button
                    size="sm"
                    onClick={handleSaveLogo}
                    disabled={mutation.isPending}
                    className="w-full h-8 text-xs"
                  >
                    {mutation.isPending ? (
                      <Loader2 className="w-3 h-3 animate-spin mr-1" />
                    ) : null}
                    Save Logo
                  </Button>
                )}
              </div>
            </>
          )}

          {/* Settings tab */}
          {activeTab === "settings" && (
            <>
              {/* Display Settings */}
              <div className="bg-muted/50 rounded-lg p-3 space-y-3">
                <div className="text-xs text-muted-foreground">Display</div>

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

              {/* Season Configuration */}
              <div className="bg-muted/50 rounded-lg p-3 space-y-3">
                <div className="text-xs text-muted-foreground">Season</div>

                <div className="space-y-1">
                  <Label htmlFor="season-start" className="text-xs text-muted-foreground">
                    Season start month
                  </Label>
                  <select
                    id="season-start"
                    value={seasonStartMonth}
                    onChange={(e) => handleSeasonStartChange(Number(e.target.value))}
                    disabled={mutation.isPending}
                    className="h-8 w-full rounded-md border border-border bg-background px-2 text-sm"
                  >
                    {[
                      { value: 1, label: "January" },
                      { value: 2, label: "February" },
                      { value: 3, label: "March" },
                      { value: 4, label: "April" },
                      { value: 5, label: "May" },
                      { value: 6, label: "June" },
                      { value: 7, label: "July" },
                      { value: 8, label: "August" },
                      { value: 9, label: "September" },
                      { value: 10, label: "October" },
                      { value: 11, label: "November" },
                      { value: 12, label: "December" },
                    ].map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] text-muted-foreground">
                    Used by Coverage Map season filters
                  </p>
                </div>
              </div>
            </>
          )}

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
      </ScrollArea>
    </>
  );
}

interface LeagueSettingsDrawerProps {
  open: boolean;
  onClose: () => void;
  league: LeagueInfo;
}

/**
 * League Settings Drawer (backward-compatible wrapper)
 *
 * Wraps LeagueSettingsPanelContent in a DetailDrawer for standalone use.
 */
export function LeagueSettingsDrawer({
  open,
  onClose,
  league,
}: LeagueSettingsDrawerProps) {
  return (
    <DetailDrawer
      open={open}
      onClose={onClose}
      title="League Settings"
      variant="overlay"
      className="shadow-none"
    >
      <LeagueSettingsPanelContent league={league} />
    </DetailDrawer>
  );
}
