"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import {
  type RegionSettings,
  type LocalDate,
  loadRegionSettings,
  saveRegionSettings,
  getDefaultRegionSettings,
  formatDateTime,
  formatDate,
  formatTime,
  formatShortDate,
  formatShortDateTime,
  getTodayLocalDate,
  localDateToUtcStartIso,
  localDateToUtcEndIso,
  dateToLocalDate,
  localDateToDate,
} from "@/lib/region";

// ============================================================================
// Context Types
// ============================================================================

interface RegionContextValue {
  /** Current region settings */
  region: RegionSettings;
  /** Update region settings (persists to localStorage) */
  setRegion: (settings: Partial<RegionSettings>) => void;
  /** Reset to browser-detected defaults */
  resetRegion: () => void;
  /** Whether settings have been loaded from storage */
  isLoaded: boolean;

  // Convenience formatters (bound to current region)
  formatDateTime: (isoUtc: string) => string;
  formatDate: (isoUtc: string) => string;
  formatTime: (isoUtc: string) => string;
  formatShortDate: (isoUtc: string) => string;
  formatShortDateTime: (isoUtc: string) => string;

  // LocalDate helpers (bound to current timezone)
  getTodayLocalDate: () => LocalDate;
  localDateToUtcStartIso: (localDate: LocalDate) => string;
  localDateToUtcEndIso: (localDate: LocalDate) => string;
  dateToLocalDate: (date: Date) => LocalDate;
  localDateToDate: (localDate: LocalDate) => Date;
}

// ============================================================================
// Context
// ============================================================================

const RegionContext = createContext<RegionContextValue | null>(null);

// ============================================================================
// Provider
// ============================================================================

interface RegionProviderProps {
  children: ReactNode;
}

export function RegionProvider({ children }: RegionProviderProps) {
  // Start with defaults, then load from storage
  const [region, setRegionState] = useState<RegionSettings>(getDefaultRegionSettings);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    const stored = loadRegionSettings();
    setRegionState(stored);
    setIsLoaded(true);
  }, []);

  // Update settings (partial update supported)
  const setRegion = useCallback((updates: Partial<RegionSettings>) => {
    setRegionState((prev) => {
      const next = { ...prev, ...updates };
      saveRegionSettings(next);
      return next;
    });
  }, []);

  // Reset to defaults
  const resetRegion = useCallback(() => {
    const defaults = getDefaultRegionSettings();
    setRegionState(defaults);
    saveRegionSettings(defaults);
  }, []);

  // Memoized context value
  const value = useMemo<RegionContextValue>(
    () => ({
      region,
      setRegion,
      resetRegion,
      isLoaded,

      // Bound formatters
      formatDateTime: (isoUtc: string) => formatDateTime(isoUtc, region),
      formatDate: (isoUtc: string) => formatDate(isoUtc, region),
      formatTime: (isoUtc: string) => formatTime(isoUtc, region),
      formatShortDate: (isoUtc: string) => formatShortDate(isoUtc, region),
      formatShortDateTime: (isoUtc: string) => formatShortDateTime(isoUtc, region),

      // Bound LocalDate helpers
      getTodayLocalDate: () => getTodayLocalDate(region.timeZone),
      localDateToUtcStartIso: (localDate: LocalDate) =>
        localDateToUtcStartIso(localDate, region.timeZone),
      localDateToUtcEndIso: (localDate: LocalDate) =>
        localDateToUtcEndIso(localDate, region.timeZone),
      dateToLocalDate: (date: Date) => dateToLocalDate(date, region.timeZone),
      localDateToDate,
    }),
    [region, setRegion, resetRegion, isLoaded]
  );

  return (
    <RegionContext.Provider value={value}>{children}</RegionContext.Provider>
  );
}

// ============================================================================
// Hook
// ============================================================================

/**
 * Hook to access region settings and formatters
 *
 * @example
 * ```tsx
 * const { region, setRegion, formatDateTime } = useRegion();
 *
 * // Format a UTC timestamp
 * const formatted = formatDateTime("2026-01-23T14:30:00Z");
 *
 * // Change timezone
 * setRegion({ timeZone: "America/New_York" });
 * ```
 */
export function useRegion(): RegionContextValue {
  const context = useContext(RegionContext);
  if (!context) {
    throw new Error("useRegion must be used within a RegionProvider");
  }
  return context;
}

// ============================================================================
// Export types
// ============================================================================

export type { RegionSettings, LocalDate };
