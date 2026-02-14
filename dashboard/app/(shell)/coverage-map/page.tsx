"use client";

import { useState, useCallback } from "react";
import { useCoverageMap } from "@/lib/hooks/use-coverage-map";
import { CoverageWorldMap, CoverageLeagueTable } from "@/components/coverage-map";
import { Loader } from "@/components/ui/loader";
import type { CoverageWindow } from "@/lib/types/coverage-map";
import { Globe } from "lucide-react";

const WINDOW_OPTIONS: { value: CoverageWindow; label: string }[] = [
  { value: "since_2023", label: "Since 2023" },
  { value: "last_365d", label: "Last 365 days" },
  { value: "season_to_date", label: "Season to date" },
];

function CoverageMapContent() {
  const [timeWindow, setTimeWindow] = useState<CoverageWindow>("since_2023");
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const { data, isLoading, error } = useCoverageMap(timeWindow);

  const handleWindowChange = useCallback((w: CoverageWindow) => {
    setTimeWindow(w);
    setSelectedCountry(null);
  }, []);

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="h-12 flex items-center justify-between px-6 border-b border-border">
        <div className="flex items-center gap-2">
          <Globe className="h-4 w-4 text-muted-foreground" />
          <h1 className="text-lg font-semibold text-foreground">
            Coverage Map
          </h1>
        </div>
        <div className="flex items-center gap-3">
          {data?.summary && (
            <span className="text-xs text-muted-foreground">
              {data.summary.countries} countries, {data.summary.leagues} leagues,{" "}
              {data.summary.eligible_matches.toLocaleString()} matches
            </span>
          )}
          <select
            value={timeWindow}
            onChange={(e) =>
              handleWindowChange(e.target.value as CoverageWindow)
            }
            className="bg-[var(--surface-elevated,#232326)] text-foreground border border-border rounded-md px-2.5 py-1.5 text-[13px] outline-none focus:border-[var(--primary)]"
          >
            {WINDOW_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader size="md" />
        </div>
      ) : error ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-[var(--error,#ef4444)]">
          <p className="text-sm">Failed to load coverage data</p>
          <p className="text-xs text-muted-foreground">
            {error instanceof Error ? error.message : "Unknown error"}
          </p>
        </div>
      ) : (
        <>
          {/* Map */}
          <CoverageWorldMap
            countries={data?.countries ?? []}
            onCountryClick={setSelectedCountry}
            selectedCountry={selectedCountry}
          />

          {/* League Table */}
          <CoverageLeagueTable
            leagues={data?.leagues ?? []}
            filterISO3={selectedCountry}
            onClearFilter={() => setSelectedCountry(null)}
          />
        </>
      )}
    </div>
  );
}

export default function CoverageMapPage() {
  return <CoverageMapContent />;
}
