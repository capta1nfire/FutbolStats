"use client";

import { useState, useCallback, useEffect } from "react";

const STORAGE_KEY = "modelBenchmarkSelection";
const ALL_MODELS = ["Market", "Model A", "Shadow", "Sensor B"] as const;
const DEFAULT_VALUE = [...ALL_MODELS];

/**
 * Hook for managing model selection state with localStorage persistence
 *
 * Used in ModelBenchmarkTile to remember which models the user has selected
 * for the cumulative accuracy chart.
 *
 * @returns Object with selectedModels array and management functions
 */
export function useModelSelection() {
  // Initialize state from localStorage or defaults
  const [selectedModels, setSelectedModelsState] = useState<string[]>(() => {
    if (typeof window === "undefined") {
      return DEFAULT_VALUE;
    }
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Validate parsed value is an array of valid model names
        if (
          Array.isArray(parsed) &&
          parsed.length >= 2 &&
          parsed.every((m) => ALL_MODELS.includes(m as typeof ALL_MODELS[number]))
        ) {
          return parsed;
        }
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
      localStorage.setItem(STORAGE_KEY, JSON.stringify(selectedModels));
    } catch {
      // Ignore storage errors
    }
  }, [selectedModels]);

  // Toggle model selection (maintains minimum of 2 models)
  const toggleModel = useCallback((name: string) => {
    setSelectedModelsState((prev) => {
      const isSelected = prev.includes(name);
      if (isSelected) {
        // Don't allow deselecting if it would leave < 2 models
        if (prev.length <= 2) return prev;
        return prev.filter((m) => m !== name);
      } else {
        return [...prev, name];
      }
    });
  }, []);

  // Check if a model can be toggled off
  const canDeselect = useCallback(
    (name: string): boolean => {
      const isSelected = selectedModels.includes(name);
      return !isSelected || selectedModels.length > 2;
    },
    [selectedModels]
  );

  // Reset to all models
  const resetToAll = useCallback(() => {
    setSelectedModelsState(DEFAULT_VALUE);
  }, []);

  return {
    selectedModels,
    toggleModel,
    canDeselect,
    resetToAll,
  };
}
