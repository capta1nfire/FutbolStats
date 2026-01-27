"use client";

import { useState, useCallback, useEffect, useRef } from "react";

const STORAGE_KEY = "sota:leagueVisibility";

/**
 * Record of league visibility by league_id
 * true = visible, false = hidden
 * Leagues not in the record default to visible
 */
export type LeagueVisibilityState = Record<string, boolean>;

/**
 * Hook for managing league visibility state with localStorage persistence
 *
 * Uses default value during SSR/hydration to avoid hydration mismatch,
 * then syncs from localStorage after mount.
 *
 * @returns Object with visibility state and helpers
 */
export function useLeagueVisibility() {
  // Always initialize with empty to match server render (all leagues visible)
  const [leagueVisibility, setLeagueVisibilityState] = useState<LeagueVisibilityState>({});
  const isInitialized = useRef(false);

  // Sync from localStorage after mount (client-side only)
  useEffect(() => {
    if (isInitialized.current) return;
    isInitialized.current = true;

    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (typeof parsed === "object" && parsed !== null) {
          // Hydration-safe: we intentionally sync client state post-mount.
          // eslint-disable-next-line react-hooks/set-state-in-effect
          setLeagueVisibilityState(parsed);
        }
      }
    } catch {
      // Ignore localStorage/parsing errors
    }
  }, []);

  // Persist to localStorage when visibility changes (skip initial)
  useEffect(() => {
    if (!isInitialized.current) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(leagueVisibility));
    } catch {
      // Ignore storage errors
    }
  }, [leagueVisibility]);

  // Check if a league is visible (default: true)
  const isLeagueVisible = useCallback(
    (leagueId: string | number): boolean => {
      const id = String(leagueId);
      return leagueVisibility[id] !== false;
    },
    [leagueVisibility]
  );

  // Set visibility for a single league
  const setLeagueVisible = useCallback((leagueId: string | number, visible: boolean) => {
    const id = String(leagueId);
    setLeagueVisibilityState((prev) => ({
      ...prev,
      [id]: visible,
    }));
  }, []);

  // Set all leagues visible or hidden
  const setAllLeaguesVisible = useCallback((visible: boolean, leagueIds: (string | number)[]) => {
    const newState: LeagueVisibilityState = {};
    leagueIds.forEach((id) => {
      newState[String(id)] = visible;
    });
    setLeagueVisibilityState(newState);
  }, []);

  // Reset to default (all visible)
  const resetToDefault = useCallback(() => {
    setLeagueVisibilityState({});
  }, []);

  return {
    leagueVisibility,
    isLeagueVisible,
    setLeagueVisible,
    setAllLeaguesVisible,
    resetToDefault,
  };
}
