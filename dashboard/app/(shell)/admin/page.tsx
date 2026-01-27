"use client";

import { Suspense, useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader } from "@/components/ui/loader";
import { buildSearchParams, parseNumericId, parseSingleParam } from "@/lib/url-state";
import {
  AdminNav,
  AdminOverview,
  AdminLeaguesTable,
  LeagueDetailDrawer,
  AdminLeagueGroupsTable,
  AdminAuditTable,
  GroupDetailDrawer,
} from "@/components/admin";
import type { AdminSection } from "@/components/admin";
import type { AdminLeagueListItem, AdminLeagueGroupListItem, AdminLeaguesFilters, AdminAuditFilters } from "@/lib/types";

const VALID_SECTIONS = ["overview", "leagues", "groups", "audit"] as const;

function AdminPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // --- Parse URL state ---
  const section = useMemo<AdminSection>(
    () => parseSingleParam(searchParams.get("section"), VALID_SECTIONS) ?? "overview",
    [searchParams]
  );

  const selectedLeagueId = useMemo(
    () => parseNumericId(searchParams.get("leagueId")),
    [searchParams]
  );

  const selectedGroupId = useMemo(
    () => parseNumericId(searchParams.get("groupId")),
    [searchParams]
  );

  // League filters
  const leagueSearch = useMemo(() => searchParams.get("q") ?? "", [searchParams]);
  const leagueKind = useMemo(() => searchParams.get("kind") ?? "", [searchParams]);
  const leagueActive = useMemo(() => searchParams.get("is_active") ?? "", [searchParams]);

  // Audit filters
  const auditEntityType = useMemo(() => searchParams.get("entity_type") ?? "", [searchParams]);
  const auditEntityId = useMemo(() => searchParams.get("entity_id") ?? "", [searchParams]);
  const auditOffset = useMemo(() => parseNumericId(searchParams.get("offset")) ?? 0, [searchParams]);

  // Static kinds list (matches DB values)
  const kinds = useMemo(() => ["league", "cup", "international", "friendly"], []);

  // --- URL builders ---
  const buildUrl = useCallback(
    (overrides: Record<string, string | number | null | undefined>) => {
      const base: Record<string, string | number | null | undefined> = {
        section: overrides.section ?? section,
      };

      // Preserve league filters when staying in leagues section
      const targetSection = overrides.section ?? section;
      if (targetSection === "leagues") {
        base.q = overrides.q !== undefined ? overrides.q : leagueSearch || undefined;
        base.kind = overrides.kind !== undefined ? overrides.kind : leagueKind || undefined;
        base.is_active = overrides.is_active !== undefined ? overrides.is_active : leagueActive || undefined;
        base.leagueId = overrides.leagueId !== undefined ? overrides.leagueId : selectedLeagueId;
      }

      if (targetSection === "groups") {
        base.groupId = overrides.groupId !== undefined ? overrides.groupId : selectedGroupId;
      }

      if (targetSection === "audit") {
        base.entity_type = overrides.entity_type !== undefined ? overrides.entity_type : auditEntityType || undefined;
        base.entity_id = overrides.entity_id !== undefined ? overrides.entity_id : auditEntityId || undefined;
        const off = overrides.offset !== undefined ? Number(overrides.offset) : auditOffset;
        if (off > 0) base.offset = off;
      }

      const params = buildSearchParams(base);
      const qs = params.toString();
      return `/admin${qs ? `?${qs}` : ""}`;
    },
    [section, leagueSearch, leagueKind, leagueActive, selectedLeagueId, selectedGroupId, auditEntityType, auditEntityId, auditOffset]
  );

  // --- Handlers ---
  const handleSectionChange = useCallback(
    (s: AdminSection) => {
      router.replace(buildUrl({ section: s, leagueId: null, groupId: null, q: null, kind: null, is_active: null, entity_type: null, entity_id: null, offset: null }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleLeagueSearchChange = useCallback(
    (v: string) => {
      router.replace(buildUrl({ q: v || null }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleLeagueKindChange = useCallback(
    (v: string) => {
      router.replace(buildUrl({ kind: v || null }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleLeagueActiveChange = useCallback(
    (v: string) => {
      router.replace(buildUrl({ is_active: v || null }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleLeagueRowClick = useCallback(
    (league: AdminLeagueListItem) => {
      router.replace(buildUrl({ leagueId: league.league_id }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleLeagueDrawerClose = useCallback(() => {
    router.replace(buildUrl({ leagueId: null }), { scroll: false });
  }, [router, buildUrl]);

  const handleGroupRowClick = useCallback(
    (group: AdminLeagueGroupListItem) => {
      router.replace(buildUrl({ groupId: group.group_id }), { scroll: false });
    },
    [router, buildUrl]
  );

  const handleGroupDrawerClose = useCallback(() => {
    router.replace(buildUrl({ groupId: null }), { scroll: false });
  }, [router, buildUrl]);

  const handleAuditOffsetChange = useCallback(
    (newOffset: number) => {
      router.replace(buildUrl({ offset: newOffset || null }), { scroll: false });
    },
    [router, buildUrl]
  );

  // --- Build filter objects ---
  const leagueFilters = useMemo<AdminLeaguesFilters>(
    () => ({
      search: leagueSearch || undefined,
      kind: leagueKind || undefined,
      is_active: leagueActive || undefined,
    }),
    [leagueSearch, leagueKind, leagueActive]
  );

  const auditFilters = useMemo<AdminAuditFilters>(
    () => ({
      entity_type: auditEntityType || undefined,
      entity_id: auditEntityId || undefined,
      limit: 50,
      offset: auditOffset || undefined,
    }),
    [auditEntityType, auditEntityId, auditOffset]
  );

  return (
    <div className="h-full flex overflow-hidden relative">
      {/* Col 2: Nav sidebar */}
      <AdminNav
        section={section}
        onSectionChange={handleSectionChange}
        leagueSearch={leagueSearch}
        onLeagueSearchChange={handleLeagueSearchChange}
        leagueKind={leagueKind}
        onLeagueKindChange={handleLeagueKindChange}
        leagueActive={leagueActive}
        onLeagueActiveChange={handleLeagueActiveChange}
        kinds={kinds}
      />

      {/* Col 4: Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {section === "overview" && <AdminOverview />}

        {section === "leagues" && (
          <AdminLeaguesTable
            filters={leagueFilters}
            selectedLeagueId={selectedLeagueId}
            onRowClick={handleLeagueRowClick}
          />
        )}

        {section === "groups" && (
          <AdminLeagueGroupsTable
            selectedGroupId={selectedGroupId}
            onRowClick={handleGroupRowClick}
          />
        )}

        {section === "audit" && (
          <AdminAuditTable filters={auditFilters} onOffsetChange={handleAuditOffsetChange} />
        )}
      </div>

      {/* Col 5: Detail drawers */}
      {section === "leagues" && (
        <LeagueDetailDrawer
          leagueId={selectedLeagueId}
          onClose={handleLeagueDrawerClose}
        />
      )}
      {section === "groups" && (
        <GroupDetailDrawer
          groupId={selectedGroupId}
          onClose={handleGroupDrawerClose}
        />
      )}
    </div>
  );
}

function AdminLoading() {
  return (
    <div className="h-full flex items-center justify-center">
      <Loader size="md" />
    </div>
  );
}

export default function AdminPage() {
  return (
    <Suspense fallback={<AdminLoading />}>
      <AdminPageContent />
    </Suspense>
  );
}
