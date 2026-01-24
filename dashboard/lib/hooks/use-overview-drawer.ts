"use client";

import { useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  OverviewPanel,
  OverviewTab,
  parsePanel,
  parseTab,
  getEffectiveTab,
  buildOverviewDrawerParams,
  PANEL_META,
  TAB_META,
  TABS_BY_PANEL,
} from "@/lib/overview-drawer";

/**
 * Hook for managing overview drawer state via URL query params
 *
 * URL format: /overview?panel=<panel>&tab=<tab>
 *
 * Features:
 * - Deep-linkable drawer state
 * - Tab validation per panel
 * - Focus return on close (via callback)
 */
export function useOverviewDrawer() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Parse current state from URL
  const panel = useMemo(
    () => parsePanel(searchParams.get("panel")),
    [searchParams]
  );

  const rawTab = useMemo(
    () => parseTab(searchParams.get("tab")),
    [searchParams]
  );

  // Get effective tab (validated for current panel)
  const tab = useMemo(() => {
    if (!panel) return null;
    return getEffectiveTab(panel, rawTab);
  }, [panel, rawTab]);

  // Drawer is open if panel is set
  const isOpen = panel !== null;

  // Get panel metadata
  const panelMeta = panel ? PANEL_META[panel] : null;

  // Get available tabs for current panel
  const availableTabs = panel ? TABS_BY_PANEL[panel] : [];

  // Get tab metadata for available tabs
  const tabsMeta = availableTabs.map((t) => ({
    key: t,
    label: TAB_META[t].label,
    isActive: t === tab,
  }));

  /**
   * Open drawer with specific panel and optional tab
   */
  const openDrawer = useCallback(
    (opts: { panel: OverviewPanel; tab?: OverviewTab }) => {
      const params = buildOverviewDrawerParams(opts);
      router.push(`/overview?${params.toString()}`, { scroll: false });
    },
    [router]
  );

  /**
   * Change tab within current panel
   */
  const setTab = useCallback(
    (newTab: OverviewTab) => {
      if (!panel) return;
      const params = buildOverviewDrawerParams({ panel, tab: newTab });
      router.replace(`/overview?${params.toString()}`, { scroll: false });
    },
    [router, panel]
  );

  /**
   * Close drawer and clear URL params
   */
  const closeDrawer = useCallback(() => {
    router.replace("/overview", { scroll: false });
  }, [router]);

  return {
    // State
    isOpen,
    panel,
    tab,
    panelMeta,
    availableTabs,
    tabsMeta,

    // Actions
    openDrawer,
    setTab,
    closeDrawer,
  };
}

/**
 * Hook for making a tile clickable to open the drawer
 */
export function useOverviewTileClick(panel: OverviewPanel) {
  const { openDrawer } = useOverviewDrawer();

  const onClick = useCallback(() => {
    openDrawer({ panel });
  }, [openDrawer, panel]);

  return { onClick };
}
