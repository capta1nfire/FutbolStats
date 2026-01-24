"use client";

import Link from "next/link";
import {
  SotaEnrichmentNormalized,
  SotaEnrichmentNormalizedItem,
  SotaEnrichmentStatus,
  SotaEnrichmentKey,
} from "@/lib/api/ops";
import { cn } from "@/lib/utils";
import {
  Database,
  CloudSun,
  MapPin,
  Users,
  ListChecks,
  ExternalLink,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SotaEnrichmentSectionProps {
  data: SotaEnrichmentNormalized | null;
  isMockFallback?: boolean;
  className?: string;
  /** Optional tile to render as 6th card in the grid (e.g., Movement) */
  movementTile?: React.ReactNode;
}

/**
 * Status pill colors
 */
const statusColors: Record<SotaEnrichmentStatus, string> = {
  ok: "bg-green-500/10 text-green-400 border-green-500/20",
  warn: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  red: "bg-red-500/10 text-red-400 border-red-500/20",
  unavailable: "bg-gray-500/10 text-gray-400 border-gray-500/20",
  pending: "bg-blue-500/10 text-blue-400 border-blue-500/20",
};

/**
 * Status dot colors for indicator
 */
const statusDotColors: Record<SotaEnrichmentStatus, string> = {
  ok: "bg-green-500",
  warn: "bg-yellow-500",
  red: "bg-red-500",
  unavailable: "bg-gray-500",
  pending: "bg-blue-500",
};

/**
 * Icon mapping for each component
 */
const componentIcons: Record<SotaEnrichmentKey, React.ReactNode> = {
  understat: <Database className="h-4 w-4" />,
  weather: <CloudSun className="h-4 w-4" />,
  venue_geo: <MapPin className="h-4 w-4" />,
  team_profiles: <Users className="h-4 w-4" />,
  sofascore_xi: <ListChecks className="h-4 w-4" />,
};

/**
 * Display names for each component
 */
const componentNames: Record<SotaEnrichmentKey, string> = {
  understat: "Understat",
  weather: "Weather",
  venue_geo: "Venue Geo",
  team_profiles: "Team Profiles",
  sofascore_xi: "Sofascore XI",
};

/**
 * Deep-link query params for Data Quality
 */
const componentDqLinks: Record<SotaEnrichmentKey, string> = {
  understat: "/data-quality?category=coverage&q=understat",
  weather: "/data-quality?category=coverage&q=weather",
  venue_geo: "/data-quality?category=coverage&q=geo",
  team_profiles: "/data-quality?category=coverage&q=team",
  sofascore_xi: "/data-quality?category=coverage&q=sofascore",
};

/**
 * Coverage color based on percentage
 */
function getCoverageColor(pct: number): string {
  if (pct >= 80) return "text-green-400";
  if (pct >= 50) return "text-yellow-400";
  return "text-red-400";
}

/**
 * Individual SOTA component card
 */
function SotaCard({ item }: { item: SotaEnrichmentNormalizedItem }) {
  const icon = componentIcons[item.key];
  const name = componentNames[item.key];
  const dqLink = componentDqLinks[item.key];

  const noteText = item.error || item.note || "No data available";

  return (
    <div className="bg-surface border border-border rounded-lg p-3">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">{icon}</span>
          <span className="text-sm font-medium text-foreground">{name}</span>
        </div>
        <span
          className={cn(
            "px-1.5 py-0.5 text-[10px] font-medium rounded border",
            statusColors[item.status]
          )}
        >
          {item.status}
        </span>
      </div>

      {/* Metrics */}
      {item.status !== "unavailable" && item.status !== "pending" ? (
        <div className="space-y-1">
          {/* Coverage */}
          <div className="flex items-baseline gap-1">
            <span
              className={cn(
                "text-xl font-bold tabular-nums",
                getCoverageColor(item.coverage_pct)
              )}
            >
              {item.coverage_pct.toFixed(1)}%
            </span>
            <span className="text-xs text-muted-foreground">
              ({item.with_data}/{item.total})
            </span>
          </div>

          {/* Staleness */}
          {item.staleness_hours !== null && (
            <div className="text-xs text-muted-foreground">
              Staleness: {item.staleness_hours.toFixed(1)}h
            </div>
          )}

          {/* Sofascore XI specific KPIs */}
          {item.key === "sofascore_xi" && item.total_lineups !== undefined && (
            <div className="text-xs text-muted-foreground">
              Lineups: {item.total_lineups}
            </div>
          )}

          {/* Note (if present) */}
          {item.note && (
            <div className="text-[10px] text-muted-foreground/70 truncate" title={item.note}>
              {item.note}
            </div>
          )}
        </div>
      ) : (
        // Unavailable/Pending state
        <div className="text-xs text-muted-foreground">
          {noteText}
        </div>
      )}

      {/* View checks link */}
      <div className="mt-2 pt-2 border-t border-border">
        <Link
          href={dqLink}
          className="text-[10px] text-primary hover:text-primary/80 flex items-center gap-1"
        >
          View checks
          <ExternalLink className="h-3 w-3" />
        </Link>
      </div>
    </div>
  );
}

/**
 * SOTA Enrichment Section
 *
 * Displays 5 cards for SOTA data enrichment metrics:
 * - Understat (xG data)
 * - Weather (forecasts)
 * - Venue Geo (geocoding)
 * - Team Profiles (metadata)
 * - Sofascore XI (lineups)
 */
export function SotaEnrichmentSection({
  data,
  isMockFallback = false,
  className,
  movementTile,
}: SotaEnrichmentSectionProps) {
  const isDegraded = !data || isMockFallback;

  // Generate mock items if no data
  const items: SotaEnrichmentNormalizedItem[] = data?.items ?? [
    { key: "understat", status: "unavailable", coverage_pct: 0, with_data: 0, total: 0, staleness_hours: null, latest_capture_at: null, note: "No data", error: null },
    { key: "weather", status: "unavailable", coverage_pct: 0, with_data: 0, total: 0, staleness_hours: null, latest_capture_at: null, note: "No data", error: null },
    { key: "venue_geo", status: "unavailable", coverage_pct: 0, with_data: 0, total: 0, staleness_hours: null, latest_capture_at: null, note: "No data", error: null },
    { key: "team_profiles", status: "unavailable", coverage_pct: 0, with_data: 0, total: 0, staleness_hours: null, latest_capture_at: null, note: "No data", error: null },
    { key: "sofascore_xi", status: "unavailable", coverage_pct: 0, with_data: 0, total: 0, staleness_hours: null, latest_capture_at: null, note: "No data", error: null },
  ];

  // Overall status dot
  const overallStatus = data?.status ?? "unavailable";

  return (
    <div className={cn("space-y-3", className)}>
      {/* Section header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-foreground">SOTA Enrichment</h2>
          <span
            className={cn("h-2 w-2 rounded-full", statusDotColors[overallStatus])}
            title={`Overall: ${overallStatus}`}
          />
        </div>
        {isDegraded && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-yellow-500/10 border border-yellow-500/20">
                  <Database className="h-3.5 w-3.5 text-yellow-400" />
                  <span className="text-[10px] text-yellow-400 font-medium">
                    mock
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p>Using mock data - backend unavailable</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {/* Cards grid - 6 columns on xl (5 SOTA + Movement), 5 on lg, 3 on md, 2 on sm */}
      <div className={cn(
        "grid gap-3",
        movementTile
          ? "grid-cols-2 md:grid-cols-3 lg:grid-cols-6"
          : "grid-cols-2 md:grid-cols-3 lg:grid-cols-5"
      )}>
        {items.map((item) => (
          <SotaCard key={item.key} item={item} />
        ))}
        {movementTile}
      </div>
    </div>
  );
}
