"use client";

import { useState, useCallback, useEffect } from "react";

const STORAGE_KEY = "compactPredictions";
const DEFAULT_VALUE = false;

/**
 * Hook for managing compact predictions state with localStorage persistence
 *
 * When enabled, probability cells show only the predicted pick (e.g., "1: 45%")
 * instead of all 3 outcomes (1: 45% X: 28% 2: 27%)
 *
 * @returns Object with compactPredictions state and setter
 */
export function useCompactPredictions() {
  // Initialize state from localStorage or defaults
  const [compactPredictions, setCompactPredictionsState] = useState<boolean>(() => {
    if (typeof window === "undefined") {
      return DEFAULT_VALUE;
    }
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored !== null) {
        return stored === "true";
      }
    } catch {
      // Ignore parsing errors
    }
    return DEFAULT_VALUE;
  });

  // Persist to localStorage when value changes
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(STORAGE_KEY, String(compactPredictions));
    } catch {
      // Ignore storage errors
    }
  }, [compactPredictions]);

  // Setter
  const setCompactPredictions = useCallback((enabled: boolean) => {
    setCompactPredictionsState(enabled);
  }, []);

  return {
    compactPredictions,
    setCompactPredictions,
  };
}
