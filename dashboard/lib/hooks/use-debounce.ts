"use client";

import { useState, useEffect } from "react";

/**
 * Debounce a value by a given delay.
 *
 * @param value - The value to debounce
 * @param delay - Delay in milliseconds (default 200ms)
 * @returns The debounced value
 *
 * @example
 * const [query, setQuery] = useState("");
 * const debouncedQuery = useDebounce(query, 200);
 * // debouncedQuery updates 200ms after query stops changing
 */
export function useDebounce<T>(value: T, delay: number = 200): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}
