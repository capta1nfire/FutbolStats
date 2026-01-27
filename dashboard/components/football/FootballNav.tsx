"use client";

import { useState } from "react";
import Image from "next/image";
import { useFootballNav, useFootballCountries, useNationalsCountries } from "@/lib/hooks";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Loader } from "@/components/ui/loader";
import { cn } from "@/lib/utils";
import { getCountryIsoCode } from "@/lib/utils/country-flags";
import {
  Search,
  Globe,
  Trophy,
  Flag,
  Users,
  ChevronRight,
  AlertCircle,
} from "lucide-react";

/**
 * Country Flag Component
 * Displays circular flag icon from circle-flags
 */
function CountryFlag({ country, className }: { country: string; className?: string }) {
  const isoCode = getCountryIsoCode(country);

  if (!isoCode) {
    // Fallback: show globe icon for unknown countries
    return <Globe className={cn("h-4 w-4 text-muted-foreground", className)} />;
  }

  return (
    <Image
      src={`/flags/${isoCode}.svg`}
      alt={`${country} flag`}
      width={16}
      height={16}
      className={cn("rounded-full object-cover", className)}
    />
  );
}

interface FootballNavProps {
  selectedCategory: string;
  onCategoryChange: (category: string) => void;
  selectedCountry: string | null;
  onCountrySelect: (country: string) => void;
}

/**
 * Get icon for category
 */
function getCategoryIcon(categoryId: string) {
  switch (categoryId) {
    case "overview":
      return Globe;
    case "leagues_by_country":
      return Trophy;
    case "national_teams":
      return Flag;
    case "clubs":
      return Users;
    default:
      return Globe;
  }
}

/**
 * FootballNav Component (Col 2)
 *
 * Displays:
 * - Category list from nav.json
 * - When category=leagues_by_country: searchable countries list
 */
export function FootballNav({
  selectedCategory,
  onCategoryChange,
  selectedCountry,
  onCountrySelect,
}: FootballNavProps) {
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch navigation categories
  const {
    data: navData,
    isLoading: isNavLoading,
    error: navError,
  } = useFootballNav();

  // Fetch countries list (only when category is leagues_by_country)
  const {
    data: countriesData,
    isLoading: isCountriesLoading,
  } = useFootballCountries();

  // Fetch nationals countries list (only when category is national_teams)
  const {
    data: nationalsData,
    isLoading: isNationalsLoading,
  } = useNationalsCountries();

  // Filter countries by search query (no manual memoization)
  const query = searchQuery.trim().toLowerCase();
  const countries = countriesData?.countries ?? [];
  const nationals = nationalsData?.countries ?? [];
  const filteredCountries = query
    ? countries.filter((c) => c.country.toLowerCase().includes(query))
    : countries;
  const filteredNationals = query
    ? nationals.filter((c) => c.country.toLowerCase().includes(query))
    : nationals;

  // Show countries list when category is leagues_by_country or national_teams
  const showCountriesList = selectedCategory === "leagues_by_country";
  const showNationalsList = selectedCategory === "national_teams";

  return (
    <div className="w-[277px] border-r border-border bg-sidebar flex flex-col">
      {/* Categories Section */}
      <div className="shrink-0">
        <div className="px-4 py-3 border-b border-border">
          <h2 className="text-sm font-semibold text-foreground">Football</h2>
        </div>

        {/* Category List */}
        <div className="p-2">
          {isNavLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader size="sm" />
            </div>
          ) : navError ? (
            <div className="flex items-center gap-2 px-2 py-3 text-sm text-destructive">
              <AlertCircle className="h-4 w-4" />
              <span>Failed to load categories</span>
            </div>
          ) : (
            <nav className="space-y-1">
              {navData?.categories.map((cat) => {
                const Icon = getCategoryIcon(cat.id);
                const isSelected = selectedCategory === cat.id;
                const isDisabled = !cat.enabled;

                return (
                  <button
                    key={cat.id}
                    onClick={() => !isDisabled && onCategoryChange(cat.id)}
                    disabled={isDisabled}
                    className={cn(
                      "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors text-left",
                      isSelected
                        ? "bg-sidebar-accent text-sidebar-primary font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-sidebar-accent/50",
                      isDisabled && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <Icon className="h-4 w-4 shrink-0" strokeWidth={1.5} />
                    <span className="flex-1 truncate">{cat.label}</span>
                    {cat.count !== undefined && cat.count > 0 && (
                      <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                        {cat.count}
                      </span>
                    )}
                    {cat.note && (
                      <span className="text-xs text-muted-foreground italic">
                        {cat.note}
                      </span>
                    )}
                    {isSelected && (showCountriesList || showNationalsList) && (
                      <ChevronRight className="h-4 w-4 shrink-0" />
                    )}
                  </button>
                );
              })}
            </nav>
          )}
        </div>
      </div>

      {/* National Teams Countries List (when national_teams is selected) */}
      {showNationalsList && (
        <>
          <div className="border-t border-border" />

          {/* Search */}
          <div className="px-3 py-2 shrink-0">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search countries..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 h-8 text-sm"
              />
            </div>
          </div>

          {/* Nationals ScrollArea */}
          <ScrollArea className="flex-1 min-h-0">
            <div className="p-2">
              {isNationalsLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader size="sm" />
                </div>
              ) : filteredNationals.length === 0 ? (
                <div className="px-3 py-4 text-sm text-muted-foreground text-center">
                  {searchQuery ? "No countries found" : "No countries available"}
                </div>
              ) : (
                <nav className="space-y-0.5">
                  {filteredNationals.map((c) => {
                    const isSelected = selectedCountry === c.country;

                    return (
                      <button
                        key={c.country}
                        onClick={() => onCountrySelect(c.country)}
                        className={cn(
                          "w-full flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-colors text-left",
                          isSelected
                            ? "bg-sidebar-accent text-sidebar-primary font-medium"
                            : "text-muted-foreground hover:text-foreground hover:bg-sidebar-accent/50"
                        )}
                      >
                        <CountryFlag country={c.country} className="shrink-0" />
                        <span className="flex-1 truncate">{c.country}</span>
                        <span className="text-xs text-muted-foreground">
                          {c.teams_count}
                        </span>
                      </button>
                    );
                  })}
                </nav>
              )}
            </div>
          </ScrollArea>

          {/* Nationals footer */}
          {nationalsData && (
            <div className="px-3 py-2 border-t border-border shrink-0">
              <span className="text-xs text-muted-foreground">
                {nationalsData.totals.countries_count} countries
              </span>
            </div>
          )}
        </>
      )}

      {/* Countries List (when leagues_by_country is selected) */}
      {showCountriesList && (
        <>
          <div className="border-t border-border" />

          {/* Search */}
          <div className="px-3 py-2 shrink-0">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search countries..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 h-8 text-sm"
              />
            </div>
          </div>

          {/* Countries ScrollArea */}
          <ScrollArea className="flex-1 min-h-0">
            <div className="p-2">
              {isCountriesLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader size="sm" />
                </div>
              ) : filteredCountries.length === 0 ? (
                <div className="px-3 py-4 text-sm text-muted-foreground text-center">
                  {searchQuery ? "No countries found" : "No countries available"}
                </div>
              ) : (
                <nav className="space-y-0.5">
                  {filteredCountries.map((c) => {
                    const isSelected = selectedCountry === c.country;

                    return (
                      <button
                        key={c.country}
                        onClick={() => onCountrySelect(c.country)}
                        className={cn(
                          "w-full flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-colors text-left",
                          isSelected
                            ? "bg-sidebar-accent text-sidebar-primary font-medium"
                            : "text-muted-foreground hover:text-foreground hover:bg-sidebar-accent/50"
                        )}
                      >
                        <CountryFlag country={c.country} className="shrink-0" />
                        <span className="flex-1 truncate">{c.country}</span>
                        <span className="text-xs text-muted-foreground">
                          {c.leagues_count}
                        </span>
                      </button>
                    );
                  })}
                </nav>
              )}
            </div>
          </ScrollArea>

          {/* Countries footer */}
          {countriesData && (
            <div className="px-3 py-2 border-t border-border shrink-0">
              <span className="text-xs text-muted-foreground">
                {countriesData.total} countries
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
