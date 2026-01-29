"use client";

import { Suspense, useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader } from "@/components/ui/loader";
import { Flag } from "lucide-react";
import {
  FootballNav,
  FootballOverview,
  CountryCompetitions,
  LeagueDetail,
  GroupDetail,
  TeamDrawer,
  TournamentsList,
  WorldCup2026Overview,
  WorldCup2026Groups,
  WorldCup2026GroupDetail,
  NationalTeamsCountryDetail,
} from "@/components/football";

/**
 * Parse numeric ID from URL param
 */
function parseNumericId(value: string | null): number | null {
  if (!value) return null;
  const num = parseInt(value, 10);
  return isNaN(num) ? null : num;
}

/**
 * Build URL search params for Football page
 */
function buildFootballUrl(params: {
  category?: string | null;
  country?: string | null;
  league?: number | null;
  group?: number | null;
  team?: number | null;
  worldCupTab?: string | null;
  worldCupGroup?: string | null;
}): string {
  const searchParams = new URLSearchParams();

  if (params.category && params.category !== "overview") {
    searchParams.set("category", params.category);
  }
  if (params.country) {
    searchParams.set("country", params.country);
  }
  if (params.league) {
    searchParams.set("league", params.league.toString());
  }
  if (params.group) {
    searchParams.set("group", params.group.toString());
  }
  if (params.team) {
    searchParams.set("team", params.team.toString());
  }
  if (params.worldCupTab && params.worldCupTab !== "overview") {
    searchParams.set("wcTab", params.worldCupTab);
  }
  if (params.worldCupGroup) {
    searchParams.set("wcGroup", params.worldCupGroup);
  }

  const search = searchParams.toString();
  return `/football${search ? `?${search}` : ""}`;
}

/**
 * Football Page Content
 *
 * URL patterns:
 * - /football                          → overview (default)
 * - /football?category=leagues_by_country → muestra lista países
 * - /football?category=leagues_by_country&country=Paraguay → muestra competiciones
 * - /football?category=leagues_by_country&country=Paraguay&league=123 → detalle liga
 * - /football?category=leagues_by_country&country=Paraguay&group=456 → detalle grupo
 * - /football?...&team=789             → drawer abierto con Team 360
 */
function FootballPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse URL state
  const category = searchParams.get("category") || "overview";
  const country = searchParams.get("country");
  const leagueId = parseNumericId(searchParams.get("league"));
  const groupId = parseNumericId(searchParams.get("group"));
  const teamId = parseNumericId(searchParams.get("team"));
  const worldCupTab = searchParams.get("wcTab") || "overview";
  const worldCupGroup = searchParams.get("wcGroup");

  // Navigation handlers
  const handleCategoryChange = useCallback(
    (newCategory: string) => {
      router.replace(
        buildFootballUrl({
          category: newCategory,
          country: null,
          league: null,
          group: null,
          team: teamId, // preserve team drawer
        }),
        { scroll: false }
      );
    },
    [router, teamId]
  );

  const handleCountrySelect = useCallback(
    (newCountry: string) => {
      router.replace(
        buildFootballUrl({
          category,
          country: newCountry,
          league: null,
          group: null,
          team: teamId,
        }),
        { scroll: false }
      );
    },
    [router, category, teamId]
  );

  const handleLeagueSelect = useCallback(
    (newLeagueId: number) => {
      router.replace(
        buildFootballUrl({
          category,
          country,
          league: newLeagueId,
          group: null,
          team: teamId,
        }),
        { scroll: false }
      );
    },
    [router, category, country, teamId]
  );

  const handleGroupSelect = useCallback(
    (newGroupId: number) => {
      router.replace(
        buildFootballUrl({
          category,
          country,
          league: null,
          group: newGroupId,
          team: teamId,
        }),
        { scroll: false }
      );
    },
    [router, category, country, teamId]
  );

  const handleTeamSelect = useCallback(
    (newTeamId: number) => {
      router.replace(
        buildFootballUrl({
          category,
          country,
          league: leagueId,
          group: groupId,
          team: newTeamId,
        }),
        { scroll: false }
      );
    },
    [router, category, country, leagueId, groupId]
  );

  const handleTeamDrawerClose = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category,
        country,
        league: leagueId,
        group: groupId,
        team: null,
      }),
      { scroll: false }
    );
  }, [router, category, country, leagueId, groupId]);

  const handleBackToCountry = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category,
        country,
        league: null,
        group: null,
        team: teamId,
      }),
      { scroll: false }
    );
  }, [router, category, country, teamId]);

  const handleBackToTournaments = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category: "tournaments_competitions",
        country: null,
        league: null,
        group: null,
        team: teamId,
      }),
      { scroll: false }
    );
  }, [router, teamId]);

  // World Cup navigation handlers
  const handleWorldCupGroupsClick = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category,
        worldCupTab: "groups",
        worldCupGroup: null,
        team: teamId,
      }),
      { scroll: false }
    );
  }, [router, category, teamId]);

  const handleWorldCupGroupSelect = useCallback(
    (groupName: string) => {
      router.replace(
        buildFootballUrl({
          category,
          worldCupTab: "groups",
          worldCupGroup: groupName,
          team: teamId,
        }),
        { scroll: false }
      );
    },
    [router, category, teamId]
  );

  const handleWorldCupBackToGroups = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category,
        worldCupTab: "groups",
        worldCupGroup: null,
        team: teamId,
      }),
      { scroll: false }
    );
  }, [router, category, teamId]);

  const handleWorldCupBackToOverview = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category,
        worldCupTab: "overview",
        worldCupGroup: null,
        team: teamId,
      }),
      { scroll: false }
    );
  }, [router, category, teamId]);

  // Determine what content to show in main area
  const contentView = useMemo(() => {
    if (category === "overview") {
      return "overview";
    }
    if (category === "leagues_by_country") {
      if (leagueId) return "league";
      if (groupId) return "group";
      if (country) return "country";
      return "overview"; // show overview when just category selected
    }
    if (category === "tournaments_competitions") {
      if (leagueId) return "league";
      return "tournaments";
    }
    if (category === "national_teams") {
      if (country) return "nationals_country";
      return "nationals_placeholder";
    }
    if (category === "world_cup_2026") {
      if (worldCupTab === "groups" && worldCupGroup) return "worldcup_group_detail";
      if (worldCupTab === "groups") return "worldcup_groups";
      return "worldcup_overview";
    }
    // Other categories show overview for now
    return "overview";
  }, [category, country, leagueId, groupId, worldCupTab, worldCupGroup]);

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* Col 2: Navigation Panel */}
      <FootballNav
        selectedCategory={category}
        onCategoryChange={handleCategoryChange}
        selectedCountry={country}
        onCountrySelect={handleCountrySelect}
        onTeamSelect={handleTeamSelect}
      />

      {/* Col 3+4: Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {contentView === "overview" && <FootballOverview />}
        {contentView === "country" && country && (
          <CountryCompetitions
            country={country}
            onLeagueSelect={handleLeagueSelect}
            onGroupSelect={handleGroupSelect}
          />
        )}
        {contentView === "league" && leagueId && (
          <LeagueDetail
            leagueId={leagueId}
            onBack={category === "tournaments_competitions" ? handleBackToTournaments : handleBackToCountry}
            onTeamSelect={handleTeamSelect}
          />
        )}
        {contentView === "group" && groupId && (
          <GroupDetail
            groupId={groupId}
            onBack={handleBackToCountry}
            onLeagueSelect={handleLeagueSelect}
            onTeamSelect={handleTeamSelect}
          />
        )}
        {contentView === "nationals_country" && country && (
          <NationalTeamsCountryDetail
            country={country}
            onTeamSelect={handleTeamSelect}
          />
        )}
        {contentView === "nationals_placeholder" && (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-md">
              <Flag className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h2 className="text-lg font-semibold text-foreground mb-2">National Teams</h2>
              <p className="text-sm text-muted-foreground">
                Select a country from the sidebar to view its national teams, competitions, and recent matches.
              </p>
            </div>
          </div>
        )}
        {contentView === "tournaments" && (
          <TournamentsList onLeagueSelect={handleLeagueSelect} />
        )}
        {contentView === "worldcup_overview" && (
          <WorldCup2026Overview onGroupsClick={handleWorldCupGroupsClick} />
        )}
        {contentView === "worldcup_groups" && (
          <WorldCup2026Groups
            onBack={handleWorldCupBackToOverview}
            onGroupSelect={handleWorldCupGroupSelect}
          />
        )}
        {contentView === "worldcup_group_detail" && worldCupGroup && (
          <WorldCup2026GroupDetail
            group={worldCupGroup}
            onBack={handleWorldCupBackToGroups}
            onTeamSelect={handleTeamSelect}
          />
        )}
      </div>

      {/* Col 5: Team Drawer */}
      <TeamDrawer
        teamId={teamId}
        open={teamId !== null}
        onClose={handleTeamDrawerClose}
      />
    </div>
  );
}

/**
 * Loading fallback for Suspense
 */
function FootballLoading() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <Loader size="md" />
    </div>
  );
}

/**
 * Football Navigation Page
 *
 * Entry point for football data exploration:
 * - Overview: Summary stats, upcoming matches, top leagues
 * - Leagues by Country: Browse countries → competitions → league/group details
 * - Team 360: Drawer with detailed team information
 */
export default function FootballPage() {
  return (
    <Suspense fallback={<FootballLoading />}>
      <FootballPageContent />
    </Suspense>
  );
}
