"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { SotaView } from "@/components/sota";

const STORAGE_KEY = "sota:activeView";
const DEFAULT_VIEW: SotaView = "enrichment";

/**
 * Hook for managing SOTA view state with localStorage persistence
 *
 * Uses default value during SSR/hydration to avoid hydration mismatch,
 * then syncs from localStorage after mount.
 *
 * @returns Object with activeView state and setter
 */
export function useSotaView() {
  // Always initialize with default to match server render
  const [activeView, setActiveViewState] = useState<SotaView>(DEFAULT_VIEW);
  const isInitialized = useRef(false);

  // Sync from localStorage after mount (client-side only)
  useEffect(() => {
    if (isInitialized.current) return;
    isInitialized.current = true;

    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "enrichment" || stored === "features") {
        setActiveViewState(stored);
      }
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  // Persist to localStorage when activeView changes (skip initial default)
  useEffect(() => {
    if (!isInitialized.current) return;
    try {
      localStorage.setItem(STORAGE_KEY, activeView);
    } catch {
      // Ignore storage errors
    }
  }, [activeView]);

  // Setter
  const setActiveView = useCallback((view: SotaView) => {
    setActiveViewState(view);
  }, []);

  return {
    activeView,
    setActiveView,
  };
}
