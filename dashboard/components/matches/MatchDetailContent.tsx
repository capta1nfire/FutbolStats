"use client";

/**
 * Shared Match Detail Components
 *
 * Extracted from MatchDetailDrawer for reuse in OverviewDrawer.
 */

import { useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { MatchSummary, MatchWeather, ProbabilitySet, StandingEntry } from "@/lib/types";
import { useStandings, useTeamLogos, useMatchSquad } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import { Loader } from "@/components/ui/loader";
import { IconTabs } from "@/components/ui/icon-tabs";
import { Badge } from "@/components/ui/badge";
import { StatusDot } from "./StatusDot";
import { DivergenceBadge } from "./DivergenceBadge";
import { computeGap20 } from "@/lib/predictions";
import { TeamLogo } from "@/components/ui/team-logo";
import { CountryFlag } from "@/components/ui/country-flag";
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
  Users,
} from "lucide-react";
import { ManagerCard, InjuryList } from "@/components/squad";
import { useRegion } from "@/components/providers/RegionProvider";

/** Get weather icon component based on conditions */
function getWeatherIcon(weather: MatchWeather): React.ComponentType<{ className?: string }> {
  if ((weather.precip_prob ?? 0) > 60 || (weather.precip_mm ?? 0) > 2) {
    return (weather.precip_mm ?? 0) > 5 ? CloudLightning : CloudRain;
  }
  if (weather.temp_c < 2 && (weather.precip_prob ?? 0) > 40) {
    return Snowflake;
  }
  if ((weather.wind_ms ?? 0) > 10) {
    return Wind;
  }
  if ((weather.cloudcover ?? 0) > 70) {
    return Cloud;
  }
  if ((weather.cloudcover ?? 0) > 30) {
    return weather.is_daylight ? CloudSun : CloudMoon;
  }
  return weather.is_daylight ? Sun : Moon;
}

/** Venue and Weather info section */
function VenueWeatherSection({ match }: { match: MatchSummary }) {
  const hasVenue = match.venue?.name || match.venue?.city;
  const hasWeather = match.weather?.temp_c !== undefined;

  if (!hasVenue && !hasWeather) return null;

  const renderWeatherIcon = () => {
    const IconComponent = match.weather ? getWeatherIcon(match.weather) : Sun;
    return <IconComponent className="h-6 w-6 text-muted-foreground" />;
  };

  return (
    <div className="bg-surface rounded-lg p-3">
      <div className="flex items-center justify-between gap-4">
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

        {hasWeather && match.weather && (
          <div className="flex items-center gap-3 shrink-0">
            {renderWeatherIcon()}
            <div className="flex items-center gap-1">
              <Thermometer className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-foreground">
                {Math.round(match.weather.temp_c)}°C
              </span>
            </div>
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
    <div className={`grid grid-cols-[24px_1fr_28px_28px_28px_28px_40px_32px_28px] gap-1 items-center font-condensed text-sm px-1 py-1.5 rounded transition-colors ${isHighlighted ? "bg-[var(--row-selected)] hover:bg-[var(--row-selected)]" : "hover:bg-muted/30"}`}>
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

      <div className="space-y-0">
        {data.standings.map((row) => (
          <StandingsRow
            key={row.position}
            row={row}
            isHighlighted={isLive && (row.teamName === homeTeamName || row.teamName === awayTeamName)}
          />
        ))}
      </div>

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

/** Single form badge (W/D/L) */
function FormBadge({ result }: { result: string }) {
  const isWin = result === "W";
  const isLoss = result === "L";
  const isDraw = result === "D";

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
  const formArray = form ? form.split("").slice(0, 5) : [];

  return (
    <div className="flex items-center justify-between py-3 border-b border-border last:border-b-0">
      <span className="text-muted-foreground text-lg w-10">
        #{position ?? "—"}
      </span>

      <div className="w-12 flex justify-center">
        <TeamLogo
          src={teamLogo ?? getLogoUrl(teamName)}
          teamName={teamName}
          size={36}
        />
      </div>

      <div className="flex items-center gap-1.5 flex-1 justify-center">
        {formArray.length > 0 ? (
          formArray.map((result, idx) => (
            <FormBadge key={idx} result={result} />
          ))
        ) : (
          <span className="text-xs text-muted-foreground">No data</span>
        )}
      </div>

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
  accentClass,
}: {
  label: string;
  probs: ProbabilitySet;
  home: string;
  away: string;
  /** Optional left-border accent class for divergence highlighting */
  accentClass?: string;
}) {
  const maxProb = Math.max(probs.home, probs.draw, probs.away);
  const pick =
    probs.home === maxProb ? home : probs.draw === maxProb ? "Draw" : away;

  return (
    <div className={cn("space-y-2 pb-3 border-b border-border last:border-b-0", accentClass && "rounded-md px-2.5 py-2 -mx-0.5", accentClass)}>
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

/**
 * Match Header - 3 column layout showing teams and score/time
 */
export function MatchHeader({
  match,
  getLogoUrl,
}: {
  match: MatchSummary;
  getLogoUrl: (teamName: string) => string | null;
}) {
  const router = useRouter();
  const { formatTime } = useRegion();
  const formattedTime = formatTime(match.kickoffISO);

  // Handler click en logo → navega a Football con TeamDrawer abierto
  // Usa directamente homeTeamId/awayTeamId del match (no necesita resolver via standings)
  const handleTeamClick = useCallback(
    (teamId: number, e: React.MouseEvent) => {
      e.stopPropagation();

      const params = new URLSearchParams({
        category: "leagues_by_country",
        league: String(match.leagueId),
      });

      if (match.leagueCountry) {
        params.set("country", match.leagueCountry);
      }

      if (teamId > 0) {
        params.set("team", String(teamId));
      }

      router.push(`/football?${params.toString()}`);
    },
    [router, match.leagueId, match.leagueCountry]
  );

  const hasScore = match.score !== undefined && match.score !== null;
  const isLive = match.status === "live" || match.status === "ht";

  // Format date for bottom row
  const kickoffDate = new Date(match.kickoffISO);
  const formattedDate = kickoffDate.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });

  // Venue info
  const venueText = match.venue?.city && match.venue?.name
    ? `${match.venue.city} / ${match.venue.name}`
    : match.venue?.name || match.venue?.city || null;

  // Weather info
  const hasWeather = match.weather?.temp_c !== undefined;
  const WeatherIcon = match.weather ? getWeatherIcon(match.weather) : null;

  return (
    <div className="bg-surface rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground">
        <CountryFlag country={match.leagueCountry} size={14} />
        <span>{match.leagueName}</span>
      </div>

      <div className="flex items-start justify-between">
        <div className="flex items-start gap-6">
          <div className="flex flex-col items-center gap-1">
            <button
              type="button"
              onClick={(e) => handleTeamClick(match.homeTeamId, e)}
              className="cursor-pointer rounded-full transition-transform hover:scale-105 focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
              aria-label={`Ver detalles de ${match.home}`}
            >
              <TeamLogo
                src={getLogoUrl(match.home) ?? null}
                teamName={match.home}
                size={48}
              />
            </button>
            <span className="text-[10px] text-muted-foreground text-center line-clamp-1 max-w-[80px]">
              {match.homeDisplayName}
            </span>
          </div>
          {hasScore && (
            <span className="text-5xl font-bold text-foreground font-condensed">
              {match.score!.home}
            </span>
          )}
        </div>

        <div className="flex flex-col items-center justify-center gap-1 flex-1 pt-[18px]">
          {hasScore ? (
            <>
              {isLive && match.elapsed && (
                <div className="text-xs text-muted-foreground">
                  {match.elapsed.min}&apos;
                  {match.elapsed.extra ? ` +${match.elapsed.extra}` : ""}
                </div>
              )}
              <StatusDot status={match.status} showLabel showIcon={false} />
            </>
          ) : (
            <>
              <div className="text-xl font-medium text-foreground">
                {formattedTime}
              </div>
              <StatusDot status={match.status} showLabel showIcon={false} />
            </>
          )}
        </div>

        <div className="flex items-start gap-6">
          {hasScore && (
            <span className="text-5xl font-bold text-foreground font-condensed">
              {match.score!.away}
            </span>
          )}
          <div className="flex flex-col items-center gap-1">
            <button
              type="button"
              onClick={(e) => handleTeamClick(match.awayTeamId, e)}
              className="cursor-pointer rounded-full transition-transform hover:scale-105 focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
              aria-label={`Ver detalles de ${match.away}`}
            >
              <TeamLogo
                src={getLogoUrl(match.away) ?? null}
                teamName={match.away}
                size={48}
              />
            </button>
            <span className="text-[10px] text-muted-foreground text-center line-clamp-1 max-w-[80px]">
              {match.awayDisplayName}
            </span>
          </div>
        </div>
      </div>

      {/* Bottom rows: Date, Venue, Weather */}
      <div className="flex flex-col gap-1.5 pt-2 border-t border-border text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <Calendar className="h-3.5 w-3.5" />
          <span>{formattedDate}</span>
        </div>

        {venueText && (
          <div className="flex items-center gap-1.5">
            <MapPin className="h-3.5 w-3.5 shrink-0" />
            <span>{venueText}</span>
          </div>
        )}

        {hasWeather && match.weather && WeatherIcon && (
          <div className="flex items-center gap-1.5">
            <WeatherIcon className="h-3.5 w-3.5" />
            <span>{Math.round(match.weather.temp_c)}°C</span>
          </div>
        )}
      </div>
    </div>
  );
}

/** Tab definitions for match detail */
export const MATCH_TABS = [
  { id: "overview", icon: <Info />, label: "Overview" },
  { id: "predictions", icon: <TrendingUp />, label: "Predictions" },
  { id: "standings", icon: <TableProperties />, label: "Standings" },
  { id: "squad", icon: <Users />, label: "Squad" },
];

/**
 * Tab content for match detail
 */
export function MatchTabContent({
  match,
  activeTab,
  getLogoUrl,
}: {
  match: MatchSummary;
  activeTab: string;
  getLogoUrl: (teamName: string) => string | null;
}) {
  const isLive = match.status === "live" || match.status === "ht";

  return (
    <div className="w-full space-y-3">
      {activeTab === "overview" && (
        <RecentFormSection match={match} getLogoUrl={getLogoUrl} />
      )}

      {activeTab === "predictions" && (() => {
        const gap20 = match.modelA && match.market
          ? computeGap20(match.modelA, match.market)
          : null;
        const hasDiv = gap20 && gap20.category !== "AGREE";
        const divAccent = hasDiv
          ? gap20.category === "STRONG_FAV_DISAGREE"
            ? "bg-destructive/8"
            : "bg-warning/8"
          : undefined;

        return (
        <div className="bg-surface rounded-lg p-4 space-y-4">
          {match.modelA || match.shadow || match.sensorB || match.market ? (
            <>
              {match.market && (
                <PredictionSection
                  label="Market"
                  probs={match.market}
                  home={match.homeDisplayName}
                  away={match.awayDisplayName}
                  accentClass={divAccent}
                />
              )}
              {match.modelA && (
                <PredictionSection
                  label="Model A"
                  probs={match.modelA}
                  home={match.homeDisplayName}
                  away={match.awayDisplayName}
                  accentClass={divAccent}
                />
              )}
              {match.shadow && (
                <PredictionSection
                  label="Shadow"
                  probs={match.shadow}
                  home={match.homeDisplayName}
                  away={match.awayDisplayName}
                />
              )}
              {match.sensorB && (
                <PredictionSection
                  label="Sensor B"
                  probs={match.sensorB}
                  home={match.homeDisplayName}
                  away={match.awayDisplayName}
                />
              )}
              {/* Divergence badge legend */}
              {hasDiv && gap20 && (
                <div className="border-t border-border pt-3 mt-1 space-y-1.5">
                  <div className="flex items-center gap-1.5">
                    <DivergenceBadge result={gap20} score={match.score} />
                    <span className="text-xs text-muted-foreground">Model-Market divergence</span>
                  </div>
                  <div className="text-[11px] text-muted-foreground/70 leading-relaxed">
                    <span className="font-semibold text-warning">Yellow</span> = Model and Market pick different favorites.{" "}
                    <span className="font-semibold text-destructive">Red</span> = Strong disagreement (gap {"\u2265"}20pp, market confidence {"\u2265"}45%).{" "}
                    No badge = both agree on the favorite.
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-8">
              <TrendingUp className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No predictions available</p>
            </div>
          )}
        </div>
        );
      })()}

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

      {activeTab === "squad" && (
        <MatchSquadSection matchId={match.id} getLogoUrl={getLogoUrl} />
      )}
    </div>
  );
}

/**
 * Match Squad section — injuries + manager for both teams
 */
function MatchSquadSection({
  matchId,
  getLogoUrl,
}: {
  matchId: number;
  getLogoUrl: (teamName: string) => string | null;
}) {
  const { data, isLoading, error } = useMatchSquad(matchId);

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <Loader />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-8">
        <Users className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Squad data not available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {[data.home, data.away].map((side) => (
        <div key={side.team_id} className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <TeamLogo
              teamName={side.team_name}
              src={getLogoUrl(side.team_name)}
              size={20}
            />
            <h4 className="text-sm font-medium">{side.team_name}</h4>
          </div>
          {side.manager && <ManagerCard manager={side.manager} compact />}
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1.5">
              Absences ({side.injuries.length})
            </p>
            <InjuryList injuries={side.injuries} compact />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Full match detail content with tabs (for use in drawers)
 */
export function MatchDetailContent({
  match,
}: {
  match: MatchSummary;
}) {
  const [activeTab, setActiveTab] = useState("overview");
  const { getLogoUrl } = useTeamLogos();

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
