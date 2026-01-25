"use client";

import { useState } from "react";
import { MatchSummary, MatchWeather, ProbabilitySet, StandingEntry } from "@/lib/types";
import { useIsDesktop, useTeamLogos, useStandings } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import { Loader } from "@/components/ui/loader";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { IconTabs } from "@/components/ui/icon-tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatusDot } from "./StatusDot";
import { TeamLogo } from "@/components/ui/team-logo";
import {
  Calendar,
  TrendingUp,
  TableProperties,
  Info,
  MapPin,
  Sun,
  Moon,
  Cloud,
  CloudSun,
  CloudMoon,
  CloudRain,
  CloudLightning,
  Snowflake,
  Wind,
  Thermometer,
  Droplets,
} from "lucide-react";
import { useRegion } from "@/components/providers/RegionProvider";

/** Map country names to ISO 3166-1 alpha-2 codes for flag emoji */
const COUNTRY_TO_CODE: Record<string, string> = {
  "Argentina": "AR", "Australia": "AU", "Austria": "AT", "Belgium": "BE",
  "Brazil": "BR", "Chile": "CL", "China": "CN", "Colombia": "CO",
  "Croatia": "HR", "Czech-Republic": "CZ", "Denmark": "DK", "Ecuador": "EC",
  "England": "GB", "Finland": "FI", "France": "FR", "Germany": "DE",
  "Greece": "GR", "Hungary": "HU", "Ireland": "IE", "Israel": "IL",
  "Italy": "IT", "Japan": "JP", "Mexico": "MX", "Netherlands": "NL",
  "Norway": "NO", "Paraguay": "PY", "Peru": "PE", "Poland": "PL",
  "Portugal": "PT", "Romania": "RO", "Russia": "RU", "Saudi-Arabia": "SA",
  "Scotland": "GB", "Serbia": "RS", "Slovakia": "SK", "Slovenia": "SI",
  "South-Korea": "KR", "Spain": "ES", "Sweden": "SE", "Switzerland": "CH",
  "Turkey": "TR", "Ukraine": "UA", "Uruguay": "UY", "USA": "US",
  "Venezuela": "VE", "Wales": "GB", "World": "UN",
};

/** Convert country name to flag emoji */
function getCountryFlag(country: string): string {
  const code = COUNTRY_TO_CODE[country];
  if (!code) return "";
  // Convert ISO code to flag emoji using regional indicator symbols
  return code
    .toUpperCase()
    .split("")
    .map((char) => String.fromCodePoint(127397 + char.charCodeAt(0)))
    .join("");
}

/** Get weather icon component based on conditions */
function getWeatherIcon(weather: MatchWeather): React.ComponentType<{ className?: string }> {
  // Rain
  if ((weather.precip_prob ?? 0) > 60 || (weather.precip_mm ?? 0) > 2) {
    return (weather.precip_mm ?? 0) > 5 ? CloudLightning : CloudRain;
  }
  // Snow
  if (weather.temp_c < 2 && (weather.precip_prob ?? 0) > 40) {
    return Snowflake;
  }
  // Strong wind
  if ((weather.wind_ms ?? 0) > 10) {
    return Wind;
  }
  // Cloudy
  if ((weather.cloudcover ?? 0) > 70) {
    return Cloud;
  }
  // Partly cloudy
  if ((weather.cloudcover ?? 0) > 30) {
    return weather.is_daylight ? CloudSun : CloudMoon;
  }
  // Clear
  return weather.is_daylight ? Sun : Moon;
}

/** Venue and Weather info section */
function VenueWeatherSection({ match }: { match: MatchSummary }) {
  const hasVenue = match.venue?.name || match.venue?.city;
  const hasWeather = match.weather?.temp_c !== undefined;

  if (!hasVenue && !hasWeather) return null;

  // Render weather icon inline to avoid component-during-render issue
  const renderWeatherIcon = () => {
    const IconComponent = match.weather ? getWeatherIcon(match.weather) : Sun;
    return <IconComponent className="h-6 w-6 text-muted-foreground" />;
  };

  return (
    <div className="bg-surface rounded-lg p-3">
      <div className="flex items-center justify-between gap-4">
        {/* Venue info (left side) */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <MapPin className="h-4 w-4 text-muted-foreground shrink-0" />
          <div className="min-w-0">
            {match.venue?.name && (
              <div className="text-sm text-foreground truncate">
                {match.venue.name}
              </div>
            )}
            {match.venue?.city && (
              <div className="text-xs text-muted-foreground truncate">
                {match.venue.city}
              </div>
            )}
            {!hasVenue && (
              <div className="text-xs text-muted-foreground">Unknown venue</div>
            )}
          </div>
        </div>

        {/* Weather info (right side) */}
        {hasWeather && match.weather && (
          <div className="flex items-center gap-3 shrink-0">
            {/* Weather icon */}
            {renderWeatherIcon()}

            {/* Temperature */}
            <div className="flex items-center gap-1">
              <Thermometer className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-foreground">
                {Math.round(match.weather.temp_c)}°C
              </span>
            </div>

            {/* Rain probability (if significant) */}
            {(match.weather.precip_prob ?? 0) > 20 && (
              <div className="flex items-center gap-1">
                <Droplets className="h-4 w-4 text-blue-400" />
                <span className="text-sm text-muted-foreground">
                  {Math.round(match.weather.precip_prob ?? 0)}%
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** Standings row component */
function StandingsRow({ row, isHighlighted }: { row: StandingEntry; isHighlighted?: boolean }) {
  return (
    <div className={`grid grid-cols-[24px_1fr_28px_28px_28px_28px_40px_32px_28px] gap-1 items-center font-condensed text-sm px-1 py-1.5 rounded transition-colors ${isHighlighted ? "bg-[#05254d] hover:bg-[#05254d]" : "hover:bg-muted/30"}`}>
      <span className="text-muted-foreground text-center">{row.position}</span>
      <span className="text-foreground truncate">{row.teamName}</span>
      <span className="text-muted-foreground text-center">{row.played}</span>
      <span className="text-muted-foreground text-center">{row.won}</span>
      <span className="text-muted-foreground text-center">{row.drawn}</span>
      <span className="text-muted-foreground text-center">{row.lost}</span>
      <span className="text-muted-foreground text-center">{row.goalsFor}:{row.goalsAgainst}</span>
      <span className={`text-center font-medium ${row.goalDiff > 0 ? "text-success" : row.goalDiff < 0 ? "text-error" : "text-muted-foreground"}`}>
        {row.goalDiff > 0 ? `+${row.goalDiff}` : row.goalDiff}
      </span>
      <span className="text-foreground text-center font-semibold">{row.points}</span>
    </div>
  );
}

/** Standings table component */
function StandingsTable({
  leagueId,
  homeTeamName,
  awayTeamName,
  isLive
}: {
  leagueId: number;
  homeTeamName?: string;
  awayTeamName?: string;
  isLive?: boolean;
}) {
  const { data, isLoading, error } = useStandings(leagueId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader size="md" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-8">
        <TableProperties className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">
          {error?.message || "Standings not available"}
        </p>
      </div>
    );
  }

  if (data.standings.length === 0) {
    return (
      <div className="text-center py-8">
        <TableProperties className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No standings data</p>
      </div>
    );
  }

  return (
    <div className="space-y-1 font-condensed">
      {/* Table header */}
      <div className="grid grid-cols-[24px_1fr_28px_28px_28px_28px_40px_32px_28px] gap-1 text-xs text-muted-foreground px-1 py-1">
        <span></span>
        <span>Club</span>
        <span className="text-center">MP</span>
        <span className="text-center">W</span>
        <span className="text-center">D</span>
        <span className="text-center">L</span>
        <span className="text-center">G</span>
        <span className="text-center">GD</span>
        <span className="text-center">P</span>
      </div>

      {/* Table rows */}
      <div className="space-y-0">
        {data.standings.map((row) => (
          <StandingsRow
            key={row.position}
            row={row}
            isHighlighted={isLive && (row.teamName === homeTeamName || row.teamName === awayTeamName)}
          />
        ))}
      </div>

      {/* Source indicator */}
      {(data.isPlaceholder || data.isCalculated) && (
        <div className="text-center pt-2 border-t border-border">
          <p className="text-xs text-muted-foreground">
            {data.isPlaceholder ? "Provisional data" : "Calculated from results"}
          </p>
        </div>
      )}
    </div>
  );
}

/** Single form badge (W/D/L) - badge style with border and transparent background */
function FormBadge({ result }: { result: string }) {
  const isWin = result === "W";
  const isLoss = result === "L";
  const isDraw = result === "D";

  // Badge style: border color + transparent background
  let borderColor = "border-muted-foreground/50";
  let bgColor = "bg-muted-foreground/10";
  let textColor = "text-muted-foreground";

  if (isWin) {
    borderColor = "border-success";
    bgColor = "bg-success/15";
    textColor = "text-success";
  } else if (isLoss) {
    borderColor = "border-error";
    bgColor = "bg-error/15";
    textColor = "text-error";
  } else if (isDraw) {
    borderColor = "border-muted-foreground/50";
    bgColor = "bg-muted-foreground/15";
    textColor = "text-muted-foreground";
  }

  return (
    <span
      className={`
        inline-flex items-center justify-center w-7 h-7 rounded text-sm font-bold
        border ${borderColor} ${bgColor} ${textColor}
      `}
    >
      {result}
    </span>
  );
}

/** Recent form display for a team */
function TeamRecentForm({
  position,
  teamLogo,
  form,
  points,
  getLogoUrl,
  teamName,
}: {
  position: number | null;
  teamLogo: string | null;
  form: string;
  points: number | null;
  getLogoUrl: (teamName: string) => string | null;
  teamName: string;
}) {
  // Parse form string (e.g., "WLDWW" -> ["W", "L", "D", "W", "W"])
  const formArray = form ? form.split("").slice(0, 5) : [];

  return (
    <div className="flex items-center justify-between py-3 border-b border-border last:border-b-0">
      {/* Position */}
      <span className="text-muted-foreground text-lg w-10">
        #{position ?? "—"}
      </span>

      {/* Team logo */}
      <div className="w-12 flex justify-center">
        <TeamLogo
          src={teamLogo ?? getLogoUrl(teamName)}
          teamName={teamName}
          size={36}
        />
      </div>

      {/* Form badges */}
      <div className="flex items-center gap-1.5 flex-1 justify-center">
        {formArray.length > 0 ? (
          formArray.map((result, idx) => (
            <FormBadge key={idx} result={result} />
          ))
        ) : (
          <span className="text-xs text-muted-foreground">No data</span>
        )}
      </div>

      {/* Points */}
      <div className="flex items-baseline gap-1 w-16 justify-end">
        <span className="text-lg font-bold text-foreground">
          {points ?? 0}
        </span>
        <span className="text-xs text-muted-foreground">Pts</span>
      </div>
    </div>
  );
}

/** Recent form section using standings data */
function RecentFormSection({
  match,
  getLogoUrl,
}: {
  match: MatchSummary;
  getLogoUrl: (teamName: string) => string | null;
}) {
  const { data: standingsData, isLoading } = useStandings(match.leagueId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader size="sm" />
      </div>
    );
  }

  // Find teams in standings
  const homeStanding = standingsData?.standings.find(
    (s) => s.teamName.toLowerCase() === match.home.toLowerCase()
  );
  const awayStanding = standingsData?.standings.find(
    (s) => s.teamName.toLowerCase() === match.away.toLowerCase()
  );

  return (
    <div className="bg-surface rounded-lg p-3">
      <TeamRecentForm
        position={homeStanding?.position ?? null}
        teamLogo={homeStanding?.teamLogo ?? null}
        form={homeStanding?.form ?? ""}
        points={homeStanding?.points ?? null}
        getLogoUrl={getLogoUrl}
        teamName={match.home}
      />
      <TeamRecentForm
        position={awayStanding?.position ?? null}
        teamLogo={awayStanding?.teamLogo ?? null}
        form={awayStanding?.form ?? ""}
        points={awayStanding?.points ?? null}
        getLogoUrl={getLogoUrl}
        teamName={match.away}
      />
    </div>
  );
}

/** Section for displaying a single prediction model */
function PredictionSection({
  label,
  probs,
  home,
  away,
}: {
  label: string;
  probs: ProbabilitySet;
  home: string;
  away: string;
}) {
  const maxProb = Math.max(probs.home, probs.draw, probs.away);
  const pick =
    probs.home === maxProb ? home : probs.draw === maxProb ? "Draw" : away;

  return (
    <div className="space-y-2 pb-3 border-b border-border last:border-b-0">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <Badge variant="secondary" className="text-xs">
          {pick}
        </Badge>
      </div>
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <span className={probs.home === maxProb ? "text-foreground" : "text-muted-foreground"}>
            {home}
          </span>
          <span className={probs.home === maxProb ? "text-foreground font-medium" : "text-muted-foreground"}>
            {(probs.home * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className={probs.draw === maxProb ? "text-foreground" : "text-muted-foreground"}>
            Draw
          </span>
          <span className={probs.draw === maxProb ? "text-foreground font-medium" : "text-muted-foreground"}>
            {(probs.draw * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex justify-between text-xs">
          <span className={probs.away === maxProb ? "text-foreground" : "text-muted-foreground"}>
            {away}
          </span>
          <span className={probs.away === maxProb ? "text-foreground font-medium" : "text-muted-foreground"}>
            {(probs.away * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}

interface MatchDetailDrawerProps {
  match: MatchSummary | null;
  /** True when match is being fetched for deep-link / pagination fallback */
  isLoading?: boolean;
  open: boolean;
  onClose: () => void;
}

/**
 * Match Header - 3 column layout showing teams and score/time
 * Replicates the design: [Home Team] | [Score/Time + Status] | [Away Team]
 */
function MatchHeader({
  match,
  getLogoUrl,
}: {
  match: MatchSummary;
  getLogoUrl: (teamName: string) => string | null;
}) {
  const { formatTime } = useRegion();
  const formattedTime = formatTime(match.kickoffISO);

  // Determine what to show in center: score or time
  const hasScore = match.score !== undefined && match.score !== null;
  const isLive = match.status === "live" || match.status === "ht";

  const countryFlag = getCountryFlag(match.leagueCountry);

  return (
    <div className="bg-surface rounded-lg p-4 space-y-3">
      {/* League header - centered */}
      <div className="flex items-center justify-center">
        <Badge variant="secondary" className="text-xs">
          {countryFlag && <span className="mr-1.5">{countryFlag}</span>}
          {match.leagueName}
        </Badge>
      </div>

      {/* Score/Time row with teams */}
      <div className="flex items-start justify-between">
        {/* Home team + score - aligned to left */}
        <div className="flex items-start gap-6">
          <div className="flex flex-col items-center gap-1">
            <TeamLogo
              src={getLogoUrl(match.home) ?? null}
              teamName={match.home}
              size={48}
            />
            <span className="text-[10px] text-muted-foreground text-center line-clamp-1 max-w-[80px]">
              {match.home}
            </span>
          </div>
          {hasScore && (
            <span className="text-5xl font-bold text-foreground font-condensed">
              {match.score!.home}
            </span>
          )}
        </div>

        {/* Center: status or time */}
        <div className="flex flex-col items-center justify-center gap-1 flex-1 pt-[18px]">
          {hasScore ? (
            <>
              <StatusDot status={match.status} showLabel />
              {isLive && match.elapsed && (
                <div className="text-xs text-muted-foreground">
                  {match.elapsed.min}&apos;
                  {match.elapsed.extra ? ` +${match.elapsed.extra}` : ""}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="text-xl font-medium text-foreground">
                {formattedTime}
              </div>
              <StatusDot status={match.status} showLabel />
            </>
          )}
        </div>

        {/* Away team + score - aligned to right */}
        <div className="flex items-start gap-6">
          {hasScore && (
            <span className="text-5xl font-bold text-foreground font-condensed">
              {match.score!.away}
            </span>
          )}
          <div className="flex flex-col items-center gap-1">
            <TeamLogo
              src={getLogoUrl(match.away) ?? null}
              teamName={match.away}
              size={48}
            />
            <span className="text-[10px] text-muted-foreground text-center line-clamp-1 max-w-[80px]">
              {match.away}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Tab definitions for match detail drawer */
const MATCH_TABS = [
  { id: "overview", icon: <Info />, label: "Overview" },
  { id: "predictions", icon: <TrendingUp />, label: "Predictions" },
  { id: "standings", icon: <TableProperties />, label: "Standings" },
];

/**
 * Tab content only - without tabs component (for desktop drawer with fixedContent)
 */
function MatchTabContent({
  match,
  activeTab,
  getLogoUrl,
}: {
  match: MatchSummary;
  activeTab: string;
  getLogoUrl: (teamName: string) => string | null;
}) {
  const kickoffDate = new Date(match.kickoffISO);
  const formattedDate = kickoffDate.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
  const formattedTime = kickoffDate.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });

  const isLive = match.status === "live" || match.status === "ht";

  return (
    <div className="w-full space-y-3">
      {/* Overview Tab - Venue/Weather Section (first card) */}
      {activeTab === "overview" && (
        <VenueWeatherSection match={match} />
      )}

      {/* Overview Tab - Match Details */}
      {activeTab === "overview" && (
        <div className="bg-surface rounded-lg p-4 space-y-3">
          {/* Match info */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">
                {formattedDate} at {formattedTime}
              </span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Match ID:</span>{" "}
              <span className="text-foreground font-mono">{match.id}</span>
            </div>
          </div>
        </div>
      )}

      {/* Recent Form Section (shown in Overview) */}
      {activeTab === "overview" && (
        <RecentFormSection match={match} getLogoUrl={getLogoUrl} />
      )}

      {/* Predictions Tab */}
      {activeTab === "predictions" && (
        <div className="bg-surface rounded-lg p-4 space-y-4">
          {match.modelA || match.shadow || match.sensorB || match.market ? (
            <>
              {/* Model A */}
              {match.modelA && (
                <PredictionSection
                  label="Model A"
                  probs={match.modelA}
                  home={match.home}
                  away={match.away}
                />
              )}

              {/* Shadow */}
              {match.shadow && (
                <PredictionSection
                  label="Shadow"
                  probs={match.shadow}
                  home={match.home}
                  away={match.away}
                />
              )}

              {/* Sensor B */}
              {match.sensorB && (
                <PredictionSection
                  label="Sensor B"
                  probs={match.sensorB}
                  home={match.home}
                  away={match.away}
                />
              )}

              {/* Market */}
              {match.market && (
                <PredictionSection
                  label="Market"
                  probs={match.market}
                  home={match.home}
                  away={match.away}
                />
              )}
            </>
          ) : (
            <div className="text-center py-8">
              <TrendingUp className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No predictions available</p>
            </div>
          )}
        </div>
      )}

      {/* Standings Tab */}
      {activeTab === "standings" && (
        <div className="bg-surface rounded-lg py-2 px-1">
          <StandingsTable
            leagueId={match.leagueId}
            homeTeamName={match.home}
            awayTeamName={match.away}
            isLive={isLive}
          />
        </div>
      )}
    </div>
  );
}

/**
 * Match Detail Content - used for mobile sheet (tabs + content together)
 */
function MatchDetailContentMobile({
  match,
  getLogoUrl,
}: {
  match: MatchSummary;
  getLogoUrl: (teamName: string) => string | null;
}) {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="w-full space-y-3">
      <MatchHeader match={match} getLogoUrl={getLogoUrl} />
      <IconTabs
        tabs={MATCH_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />
      <MatchTabContent match={match} activeTab={activeTab} getLogoUrl={getLogoUrl} />
    </div>
  );
}

/**
 * Responsive Match Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer (no reflow, ~400px)
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function MatchDetailDrawer({
  match,
  isLoading = false,
  open,
  onClose,
}: MatchDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const [activeTab, setActiveTab] = useState("overview");
  const { getLogoUrl } = useTeamLogos();
  const matchTitle = match ? `Match ID ${match.id}` : "Match Details";

  // Desktop: overlay drawer with tabs in fixedContent (prevents tooltip clipping)
  if (isDesktop) {
    return (
      <DetailDrawer
        open={open}
        onClose={onClose}
        title={matchTitle}
        fixedContent={
          match && (
            <div className="space-y-3">
              <MatchHeader match={match} getLogoUrl={getLogoUrl} />
              <IconTabs
                tabs={MATCH_TABS}
                value={activeTab}
                onValueChange={setActiveTab}
                className="w-full"
              />
            </div>
          )
        }
      >
        {match ? (
          <MatchTabContent match={match} activeTab={activeTab} getLogoUrl={getLogoUrl} />
        ) : isLoading ? (
          <div className="h-full flex items-center justify-center py-10">
            <Loader size="md" />
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">Select a match to view details</p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {matchTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {match ? (
              <MatchDetailContentMobile match={match} getLogoUrl={getLogoUrl} />
            ) : isLoading ? (
              <div className="h-full flex items-center justify-center py-10">
                <Loader size="md" />
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Select a match to view details</p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
