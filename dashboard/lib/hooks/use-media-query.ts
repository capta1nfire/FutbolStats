"use client";

import { useSyncExternalStore } from "react";

/**
 * Hook to detect if a media query matches
 * Uses useSyncExternalStore for proper SSR handling and avoiding setState in effects
 * @param query - CSS media query string (e.g., "(min-width: 1280px)")
 * @returns boolean indicating if the query matches
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = (callback: () => void) => {
    const mediaQuery = window.matchMedia(query);
    mediaQuery.addEventListener("change", callback);
    return () => mediaQuery.removeEventListener("change", callback);
  };

  const getSnapshot = () => {
    return window.matchMedia(query).matches;
  };

  const getServerSnapshot = () => {
    // Default to false on server (assumes mobile-first)
    return false;
  };

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

/**
 * Breakpoint constants matching Tailwind
 */
export const BREAKPOINTS = {
  sm: "(min-width: 640px)",
  md: "(min-width: 768px)",
  lg: "(min-width: 1024px)",
  xl: "(min-width: 1280px)",
  "2xl": "(min-width: 1536px)",
} as const;

/**
 * Hook to detect if viewport is desktop (>=1280px)
 * Desktop uses inline drawer, mobile/tablet uses Sheet overlay
 */
export function useIsDesktop(): boolean {
  return useMediaQuery(BREAKPOINTS.xl);
}

/**
 * Hook to detect if component has mounted (client-side)
 * Uses useSyncExternalStore to avoid setState in effects
 * Useful for avoiding Radix UI hydration mismatches
 */
export function useHasMounted(): boolean {
  const subscribe = () => {
    // No-op: mounted state never changes after initial render
    return () => {};
  };

  const getSnapshot = () => true;
  const getServerSnapshot = () => false;

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
