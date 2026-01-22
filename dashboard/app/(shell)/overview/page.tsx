"use client";

import { useOverviewData } from "@/lib/hooks";
import {
  HealthCard,
  CoverageBar,
  UpcomingMatchesList,
  ActiveIncidentsList,
  ApiBudgetCard,
} from "@/components/overview";
import { RefreshCw, AlertCircle } from "lucide-react";
import { Loader } from "@/components/ui/loader";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import Link from "next/link";

/**
 * Overview Page
 *
 * Dashboard home with:
 * - Health cards (System, Predictions, Jobs, Live)
 * - Coverage bar
 * - Two-column layout: Upcoming Matches + Active Incidents
 */
export default function OverviewPage() {
  const { data, isLoading, error, refetch } = useOverviewData();

  // Loading state
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <Loader size="md" />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <AlertCircle className="h-12 w-12 text-error" />
          <div className="text-center">
            <p className="text-foreground font-medium mb-1">
              Failed to load overview
            </p>
            <p className="text-sm text-muted-foreground mb-4">
              {error.message}
            </p>
          </div>
          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // No data (shouldn't happen but handle gracefully)
  if (!data) {
    return null;
  }

  const { health, upcomingMatches, activeIncidents, apiBudget } = data;

  return (
    <div className="h-full flex overflow-hidden">
      {/* Left Rail: API Budget (aligned with FilterPanel at 277px) */}
      <aside className="w-[277px] shrink-0 border-r border-border bg-sidebar flex flex-col">
        {/* Header - consistent with FilterPanel */}
        <div className="h-12 flex items-center px-3 border-b border-border">
          <span className="text-sm font-medium text-foreground">Dashboard</span>
        </div>
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-3">
          {apiBudget && <ApiBudgetCard budget={apiBudget} />}
        </div>
      </aside>

      {/* Main content */}
      <ScrollArea className="flex-1">
        <div className="p-6 space-y-6">
          {/* Page header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">Overview</h1>
              <p className="text-sm text-muted-foreground">
                System health and real-time status
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetch()}
              aria-label="Refresh data"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>

          {/* Health cards grid */}
          <section aria-labelledby="health-heading">
            <h2 id="health-heading" className="sr-only">
              Health Status
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {health.cards.map((card) => (
                <HealthCard key={card.id} card={card} />
              ))}
            </div>
          </section>

          {/* Coverage bar */}
          <section
            aria-labelledby="coverage-heading"
            className="bg-surface rounded-lg border border-border p-4"
          >
            <h2 id="coverage-heading" className="sr-only">
              Prediction Coverage
            </h2>
            <CoverageBar percentage={health.coveragePct} />
          </section>

          {/* Two-column layout: Upcoming + Incidents */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Upcoming Matches */}
            <section
              aria-labelledby="upcoming-heading"
              className="bg-surface rounded-lg border border-border p-4"
            >
              <div className="flex items-center justify-between mb-4">
                <h2
                  id="upcoming-heading"
                  className="text-sm font-semibold text-foreground"
                >
                  Upcoming Matches
                </h2>
                <span className="text-xs text-muted-foreground">
                  {upcomingMatches.length} matches
                </span>
              </div>
              <UpcomingMatchesList matches={upcomingMatches.slice(0, 6)} />
              {upcomingMatches.length > 6 && (
                <div className="mt-3 pt-3 border-t border-border">
                  <Link
                    href="/matches"
                    className="text-xs text-accent hover:underline"
                  >
                    View all {upcomingMatches.length} matches →
                  </Link>
                </div>
              )}
            </section>

            {/* Active Incidents */}
            <section
              aria-labelledby="incidents-heading"
              className="bg-surface rounded-lg border border-border p-4"
            >
              <div className="flex items-center justify-between mb-4">
                <h2
                  id="incidents-heading"
                  className="text-sm font-semibold text-foreground"
                >
                  Active Incidents
                </h2>
                <span className="text-xs text-muted-foreground">
                  {activeIncidents.length} active
                </span>
              </div>
              <ActiveIncidentsList incidents={activeIncidents.slice(0, 5)} />
              {activeIncidents.length > 5 && (
                <div className="mt-3 pt-3 border-t border-border">
                  <Link
                    href="/incidents"
                    className="text-xs text-accent hover:underline"
                  >
                    View all {activeIncidents.length} incidents →
                  </Link>
                </div>
              )}
            </section>
          </div>

          {/* Last updated */}
          <div className="text-xs text-muted-foreground text-center">
            Last updated:{" "}
            {new Date(health.lastUpdated).toLocaleString("en-US", {
              month: "short",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
