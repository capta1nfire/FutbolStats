"use client";

import { Suspense, useCallback, useMemo, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useFootballTeam, useFootballLeague, useFootballCountries } from "@/lib/hooks";
import { Loader } from "@/components/ui/loader";
import { Flag } from "lucide-react";
import {
  FootballNav,
  FootballOverview,
  LeagueDetail,
  GroupDetail,
  TeamDrawer,
  TournamentsList,
  WorldCup2026Overview,
  WorldCup2026Groups,
  WorldCup2026GroupDetail,
  NationalTeamsCountryDetail,
  PlayersView,
  ManagersView,
} from "@/components/football";
import { LeagueSettingsDrawer } from "@/components/football/LeagueSettingsDrawer";

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
  leagueSettings?: boolean | null;
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
  if (params.leagueSettings) {
    searchParams.set("leagueSettings", "true");
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
  const leagueSettingsOpen = searchParams.get("leagueSettings") === "true";

  // Fetch team data to auto-select primary league
  const { data: teamData } = useFootballTeam(teamId);

  // Fetch league data for settings drawer
  const { data: leagueData } = useFootballLeague(leagueId);

  // Fetch countries data (cached via react-query) for sibling leagues resolution
  const { data: countriesData } = useFootballCountries();

  // Track if we've already auto-selected the league for this team
  const autoSelectedLeagueRef = useRef<number | null>(null);

  // Auto-select primary league when team is selected from search
  // Only runs when: teamId exists, no leagueId selected, team data loaded with leagues
  useEffect(() => {
    if (
      teamId &&
      !leagueId &&
      teamData?.leagues_played &&
      teamData.leagues_played.length > 0 &&
      autoSelectedLeagueRef.current !== teamId
    ) {
      const primaryLeague = teamData.leagues_played[0];
      autoSelectedLeagueRef.current = teamId;

      router.replace(
        buildFootballUrl({
          category: "leagues_by_country",
          country: teamData.team?.country || country,
          league: primaryLeague.league_id,
          group: null,
          team: teamId,
        }),
        { scroll: false }
      );
    }
  }, [teamId, leagueId, teamData, country, router]);

  // Navigation handlers
  const handleCategoryChange = useCallback(
    (newCategory: string) => {
      // Auto-select first country + primary league when entering leagues_by_country
      if (newCategory === "leagues_by_country" && countriesData?.countries?.length) {
        const first = countriesData.countries[0];
        const primaryLeague = first.leagues[0];
        if (primaryLeague) {
          router.replace(
            buildFootballUrl({
              category: newCategory,
              country: first.country,
              league: primaryLeague.league_id,
              group: null,
              team: null,
            }),
            { scroll: false }
          );
          return;
        }
      }
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
    [router, teamId, countriesData]
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

  // Navigate directly to a country's primary league
  const handleCountryLeagueSelect = useCallback(
    (newCountry: string, primaryLeagueId: number) => {
      router.replace(
        buildFootballUrl({
          category,
          country: newCountry,
          league: primaryLeagueId,
          group: null,
          team: null,
        }),
        { scroll: false }
      );
    },
    [router, category]
  );

  // Sibling leagues for current country (sorted by priority from backend)
  const siblingLeagues = useMemo(() => {
    if (!country || !countriesData?.countries) return [];
    const countryData = countriesData.countries.find((c) => c.country === country);
    return countryData?.leagues ?? [];
  }, [country, countriesData]);

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
    (newTeamId: number, teamCountry?: string) => {
      // If teamCountry is provided (from search), navigate to that country's context
      // This ensures the background shows the team's league context
      const targetCountry = teamCountry || country;
      const targetCategory = teamCountry ? "leagues_by_country" : category;

      router.replace(
        buildFootballUrl({
          category: targetCategory,
          country: targetCountry,
          league: teamCountry ? null : leagueId, // Clear league when coming from search
          group: teamCountry ? null : groupId,   // Clear group when coming from search
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
        country: null,
        league: null,
        group: null,
        team: teamId,
      }),
      { scroll: false }
    );
  }, [router, category, teamId]);

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

  // League Settings handlers
  const handleSettingsOpen = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category,
        country,
        league: leagueId,
        group: groupId,
        team: teamId,
        leagueSettings: true,
      }),
      { scroll: false }
    );
  }, [router, category, country, leagueId, groupId, teamId]);

  const handleSettingsClose = useCallback(() => {
    router.replace(
      buildFootballUrl({
        category,
        country,
        league: leagueId,
        group: groupId,
        team: teamId,
        leagueSettings: null,
      }),
      { scroll: false }
    );
  }, [router, category, country, leagueId, groupId, teamId]);

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
      return "overview"; // country without league → show overview
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
    if (category === "players") return "players";
    if (category === "managers") return "managers";
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
        onCountryLeagueSelect={handleCountryLeagueSelect}
        onTeamSelect={handleTeamSelect}
      />

      {/* Col 3+4: Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden bg-background">
        {contentView === "overview" && <FootballOverview />}
        {contentView === "league" && leagueId && (
          <LeagueDetail
            leagueId={leagueId}
            onBack={category === "tournaments_competitions" ? handleBackToTournaments : handleBackToCountry}
            onTeamSelect={handleTeamSelect}
            onSettingsClick={handleSettingsOpen}
            initialTeamId={teamId}
            siblingLeagues={siblingLeagues}
            onLeagueChange={handleLeagueSelect}
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
        {contentView === "players" && (
          <PlayersView onTeamSelect={handleTeamSelect} />
        )}
        {contentView === "managers" && (
          <ManagersView onTeamSelect={handleTeamSelect} />
        )}
      </div>

      {/* League Settings Drawer (overlay) */}
      {leagueData?.league && (
        <LeagueSettingsDrawer
          open={leagueSettingsOpen}
          onClose={handleSettingsClose}
          league={leagueData.league}
        />
      )}

      {/* Col 5: Team Drawer (persistent/fixed) - hidden when LeagueSettings is open */}
      <TeamDrawer
        teamId={teamId}
        open={!leagueSettingsOpen}
        onClose={handleTeamDrawerClose}
        persistent
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
