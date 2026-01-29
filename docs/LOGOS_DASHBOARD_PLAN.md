# Plan: Logos 3D - Dashboard UI

## Para revisión de: ADB (Auditor Dashboard)

**Fecha**: 2026-01-28
**Autor**: Claude (Codificador Dashboard)
**Backend**: Completado y desplegado
**Dependencia**: Endpoints `/dashboard/logos/*` ya disponibles

---

## Resumen Ejecutivo

Implementar la interfaz gráfica del sistema de logos 3D siguiendo los patrones existentes del dashboard:
- TanStack Query para data fetching
- Componentes en `components/settings/`
- Nueva sección en Settings: "3D Logos"
- API client en `lib/api/logos.ts`
- Hooks en `lib/hooks/use-logos-*.ts`

---

## Análisis de Patrones Existentes

### Estructura de Settings Sections
```
components/settings/
├── SettingsNav.tsx          # Sidebar con secciones
├── SettingsSectionHeader.tsx # Header reutilizable
├── IaFeaturesSection.tsx    # Referencia: sección compleja con forms
├── FeatureFlagsSection.tsx  # Referencia: tabla con search
└── index.ts                 # Barrel exports
```

### Patrones de Hooks
```typescript
// lib/hooks/use-settings-api.ts pattern:
export function useXxxApi(): UseXxxResult {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["xxx", params],
    queryFn: () => fetchXxx(params),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  });
  return { data, isDegraded: !!error, isLoading, refetch };
}
```

### Patrones de API Client
```typescript
// lib/api/settings.ts pattern:
export async function fetchXxx(): Promise<XxxResponse | null> {
  const response = await fetch("/api/xxx");
  if (!response.ok) return null;
  return parseXxx(await response.json());
}
```

---

## Archivos a Crear/Modificar

### 1. Types (`lib/types/logos.ts`) - NUEVO

```typescript
/**
 * Logos 3D Types
 */

// Batch job status
export type LogoBatchStatus =
  | "running"
  | "paused"
  | "completed"
  | "cancelled"
  | "error"
  | "pending_review";

// Team logo status
export type TeamLogoStatus =
  | "pending"
  | "queued"
  | "processing"
  | "pending_resize"
  | "ready"
  | "error"
  | "paused";

// Review status
export type ReviewStatus = "pending" | "approved" | "rejected" | "needs_regeneration";

// Generation mode
export type GenerationMode = "full_3d" | "facing_only" | "front_only" | "manual";

// IA Model
export type IAModel = "imagen-3" | "dall-e-3" | "sdxl";

/**
 * League for generation
 */
export interface LeagueForGeneration {
  leagueId: number;
  name: string;
  country: string;
  teamCount: number;
  pendingCount: number;
  readyCount: number;
  errorCount: number;
}

/**
 * Batch job
 */
export interface LogoBatchJob {
  id: string;
  leagueId: number;
  leagueName: string;
  iaModel: IAModel;
  generationMode: GenerationMode;
  status: LogoBatchStatus;
  totalTeams: number;
  processedTeams: number;
  failedTeams: number;
  estimatedCostUsd: number;
  actualCostUsd: number;
  progress: number; // 0-100
  startedAt: string;
  pausedAt?: string;
  completedAt?: string;
  startedBy: string;
}

/**
 * Team logo for review
 */
export interface TeamLogoReview {
  teamId: number;
  teamName: string;
  status: TeamLogoStatus;
  reviewStatus: ReviewStatus;
  urls: {
    original?: string;
    front?: string;
    right?: string;
    left?: string;
  };
  thumbnails?: {
    front?: Record<number, string>;
    right?: Record<number, string>;
    left?: Record<number, string>;
  };
  fallbackUrl?: string;
  errorMessage?: string;
  iaCostUsd?: number;
}

/**
 * Generate request
 */
export interface GenerateBatchRequest {
  generationMode: GenerationMode;
  iaModel: IAModel;
  promptVersion?: string;
}

/**
 * Review request
 */
export interface ReviewTeamRequest {
  action: "approve" | "reject" | "regenerate";
  notes?: string;
}

/**
 * Cost estimate
 */
export interface CostEstimate {
  teamCount: number;
  imagesPerTeam: number;
  costPerImage: number;
  totalCost: number;
  isFree: boolean;
}
```

### 2. API Client (`lib/api/logos.ts`) - NUEVO

```typescript
/**
 * Logos 3D API Client
 *
 * Proxy calls to backend /dashboard/logos/* endpoints
 */

import {
  LeagueForGeneration,
  LogoBatchJob,
  TeamLogoReview,
  GenerateBatchRequest,
  ReviewTeamRequest,
  CostEstimate,
} from "@/lib/types/logos";

const API_BASE = "/api/logos"; // Next.js proxy

// ============================================================================
// Leagues
// ============================================================================

export async function fetchLeaguesForGeneration(): Promise<LeagueForGeneration[]> {
  const res = await fetch(`${API_BASE}/leagues`);
  if (!res.ok) throw new Error("Failed to fetch leagues");
  const data = await res.json();
  return data.leagues.map(parseLeague);
}

// ============================================================================
// Batch Jobs
// ============================================================================

export async function startBatchJob(
  leagueId: number,
  request: GenerateBatchRequest
): Promise<{ batchId: string; estimatedCost: number }> {
  const res = await fetch(`${API_BASE}/generate/league/${leagueId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to start batch");
  }
  return res.json();
}

export async function fetchBatchStatus(batchId: string): Promise<LogoBatchJob> {
  const res = await fetch(`${API_BASE}/batch/${batchId}`);
  if (!res.ok) throw new Error("Failed to fetch batch status");
  return parseBatchJob(await res.json());
}

export async function pauseBatch(batchId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/batch/${batchId}/pause`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to pause batch");
}

export async function resumeBatch(batchId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/batch/${batchId}/resume`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to resume batch");
}

export async function cancelBatch(batchId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/batch/${batchId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to cancel batch");
}

// ============================================================================
// Review
// ============================================================================

export async function fetchLeagueReview(
  leagueId: number,
  statusFilter?: string
): Promise<{ total: number; teams: TeamLogoReview[] }> {
  const params = new URLSearchParams();
  if (statusFilter) params.set("status_filter", statusFilter);

  const url = `${API_BASE}/review/league/${leagueId}${params.toString() ? `?${params}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch review");

  const data = await res.json();
  return {
    total: data.total,
    teams: data.teams.map(parseTeamLogo),
  };
}

export async function reviewTeamLogo(
  teamId: number,
  request: ReviewTeamRequest
): Promise<void> {
  const res = await fetch(`${API_BASE}/review/team/${teamId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error("Failed to review team");
}

export async function approveLeagueLogos(
  leagueId: number,
  action: "approve_all" | "reject_all"
): Promise<{ updatedCount: number }> {
  const res = await fetch(`${API_BASE}/review/league/${leagueId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  if (!res.ok) throw new Error("Failed to approve league");
  return res.json();
}

// ============================================================================
// Cost Estimation
// ============================================================================

export function estimateCost(
  teamCount: number,
  generationMode: GenerationMode,
  iaModel: IAModel
): CostEstimate {
  const imagesPerTeam = {
    full_3d: 3,
    facing_only: 2,
    front_only: 1,
    manual: 0,
  }[generationMode];

  // Free tier (imagen-3) vs paid
  const costPerImage = {
    "imagen-3": 0, // Free tier default
    "dall-e-3": 0.04,
    "sdxl": 0.006,
  }[iaModel];

  return {
    teamCount,
    imagesPerTeam,
    costPerImage,
    totalCost: teamCount * imagesPerTeam * costPerImage,
    isFree: iaModel === "imagen-3", // Using free tier
  };
}

// ============================================================================
// Parsers
// ============================================================================

function parseLeague(raw: any): LeagueForGeneration {
  return {
    leagueId: raw.league_id,
    name: raw.name,
    country: raw.country,
    teamCount: raw.team_count,
    pendingCount: raw.pending_count ?? 0,
    readyCount: raw.ready_count ?? 0,
    errorCount: raw.error_count ?? 0,
  };
}

function parseBatchJob(raw: any): LogoBatchJob {
  return {
    id: raw.batch_id,
    leagueId: raw.league_id,
    leagueName: raw.league_name ?? "",
    iaModel: raw.ia_model,
    generationMode: raw.generation_mode,
    status: raw.status,
    totalTeams: raw.total_teams,
    processedTeams: raw.progress?.processed_teams ?? 0,
    failedTeams: raw.progress?.failed_teams ?? 0,
    estimatedCostUsd: raw.cost?.estimated_usd ?? 0,
    actualCostUsd: raw.cost?.actual_usd ?? 0,
    progress: raw.progress?.percentage ?? 0,
    startedAt: raw.timestamps?.started_at,
    pausedAt: raw.timestamps?.paused_at,
    completedAt: raw.timestamps?.completed_at,
    startedBy: raw.started_by ?? "unknown",
  };
}

function parseTeamLogo(raw: any): TeamLogoReview {
  return {
    teamId: raw.team_id,
    teamName: raw.team_name,
    status: raw.status,
    reviewStatus: raw.review_status,
    urls: raw.urls ?? {},
    thumbnails: raw.thumbnails,
    fallbackUrl: raw.fallback_url,
    errorMessage: raw.error_message,
    iaCostUsd: raw.ia_cost_usd,
  };
}
```

### 3. Hooks (`lib/hooks/use-logos.ts`) - NUEVO

```typescript
/**
 * Logos 3D Hooks
 *
 * TanStack Query hooks for logos management
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchLeaguesForGeneration,
  fetchBatchStatus,
  fetchLeagueReview,
  startBatchJob,
  pauseBatch,
  resumeBatch,
  cancelBatch,
  reviewTeamLogo,
  approveLeagueLogos,
} from "@/lib/api/logos";
import type { GenerateBatchRequest, ReviewTeamRequest } from "@/lib/types/logos";

// ============================================================================
// Leagues Hook
// ============================================================================

export function useLogosLeagues() {
  return useQuery({
    queryKey: ["logos", "leagues"],
    queryFn: fetchLeaguesForGeneration,
    staleTime: 60 * 1000, // 1 minute
  });
}

// ============================================================================
// Batch Status Hook (with polling)
// ============================================================================

export function useLogosBatchStatus(batchId: string | null) {
  return useQuery({
    queryKey: ["logos", "batch", batchId],
    queryFn: () => fetchBatchStatus(batchId!),
    enabled: !!batchId,
    refetchInterval: (query) => {
      // Poll every 5s while running, stop when done
      const status = query.state.data?.status;
      return status === "running" ? 5000 : false;
    },
  });
}

// ============================================================================
// Review Hook
// ============================================================================

export function useLogosReview(leagueId: number | null, statusFilter?: string) {
  return useQuery({
    queryKey: ["logos", "review", leagueId, statusFilter],
    queryFn: () => fetchLeagueReview(leagueId!, statusFilter),
    enabled: !!leagueId,
  });
}

// ============================================================================
// Mutations
// ============================================================================

export function useStartBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ leagueId, request }: { leagueId: number; request: GenerateBatchRequest }) =>
      startBatchJob(leagueId, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["logos", "leagues"] });
    },
  });
}

export function usePauseBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: pauseBatch,
    onSuccess: (_, batchId) => {
      queryClient.invalidateQueries({ queryKey: ["logos", "batch", batchId] });
    },
  });
}

export function useResumeBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: resumeBatch,
    onSuccess: (_, batchId) => {
      queryClient.invalidateQueries({ queryKey: ["logos", "batch", batchId] });
    },
  });
}

export function useCancelBatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: cancelBatch,
    onSuccess: (_, batchId) => {
      queryClient.invalidateQueries({ queryKey: ["logos", "batch", batchId] });
      queryClient.invalidateQueries({ queryKey: ["logos", "leagues"] });
    },
  });
}

export function useReviewTeamLogo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ teamId, request }: { teamId: number; request: ReviewTeamRequest }) =>
      reviewTeamLogo(teamId, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["logos", "review"] });
    },
  });
}

export function useApproveLeague() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ leagueId, action }: { leagueId: number; action: "approve_all" | "reject_all" }) =>
      approveLeagueLogos(leagueId, action),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["logos"] });
    },
  });
}
```

### 4. Next.js API Proxy (`app/api/logos/[...path]/route.ts`) - NUEVO

```typescript
/**
 * Logos API Proxy
 *
 * Proxies requests to backend /dashboard/logos/* endpoints
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "https://web-production-f2de9.up.railway.app";
const DASHBOARD_TOKEN = process.env.DASHBOARD_TOKEN || "";

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join("/");
  const url = `${BACKEND_URL}/dashboard/logos/${path}${request.nextUrl.search}`;

  const res = await fetch(url, {
    headers: {
      "X-Dashboard-Token": DASHBOARD_TOKEN,
      Accept: "application/json",
    },
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join("/");
  const url = `${BACKEND_URL}/dashboard/logos/${path}`;
  const body = await request.json().catch(() => ({}));

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "X-Dashboard-Token": DASHBOARD_TOKEN,
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
```

### 5. Settings Section Component (`components/settings/LogosSection.tsx`) - NUEVO

```typescript
/**
 * Logos 3D Settings Section
 *
 * Main component for logos generation and review
 */

"use client";

import { useState } from "react";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { LogosLeagueSelector } from "./LogosLeagueSelector";
import { LogosGeneratePanel } from "./LogosGeneratePanel";
import { LogosBatchProgress } from "./LogosBatchProgress";
import { LogosReviewGrid } from "./LogosReviewGrid";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Image, ListChecks, Settings2 } from "lucide-react";

export function LogosSection() {
  const [selectedLeagueId, setSelectedLeagueId] = useState<number | null>(null);
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("generate");

  return (
    <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
      <SettingsSectionHeader
        title="3D Logos"
        description="Generate and review 3D logo variants for team shields"
      />

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="generate" className="flex items-center gap-2">
            <Image className="h-4 w-4" />
            Generate
          </TabsTrigger>
          <TabsTrigger value="review" className="flex items-center gap-2">
            <ListChecks className="h-4 w-4" />
            Review
          </TabsTrigger>
          <TabsTrigger value="config" className="flex items-center gap-2">
            <Settings2 className="h-4 w-4" />
            Config
          </TabsTrigger>
        </TabsList>

        <TabsContent value="generate" className="space-y-6 mt-6">
          {/* League Selector */}
          <LogosLeagueSelector
            selectedLeagueId={selectedLeagueId}
            onSelect={setSelectedLeagueId}
          />

          {/* Generate Panel (when league selected) */}
          {selectedLeagueId && !activeBatchId && (
            <LogosGeneratePanel
              leagueId={selectedLeagueId}
              onBatchStarted={setActiveBatchId}
            />
          )}

          {/* Batch Progress (when batch running) */}
          {activeBatchId && (
            <LogosBatchProgress
              batchId={activeBatchId}
              onComplete={() => {
                setActiveBatchId(null);
                setActiveTab("review");
              }}
            />
          )}
        </TabsContent>

        <TabsContent value="review" className="space-y-6 mt-6">
          {/* League Selector */}
          <LogosLeagueSelector
            selectedLeagueId={selectedLeagueId}
            onSelect={setSelectedLeagueId}
          />

          {/* Review Grid */}
          {selectedLeagueId && (
            <LogosReviewGrid leagueId={selectedLeagueId} />
          )}
        </TabsContent>

        <TabsContent value="config" className="space-y-6 mt-6">
          {/* Config panel - placeholder */}
          <div className="bg-surface/50 rounded-lg p-4 border border-border">
            <p className="text-sm text-muted-foreground">
              Configuration is managed via environment variables. Current settings:
            </p>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              <li>• IA Model: imagen-3 (Google AI Studio - Free Tier)</li>
              <li>• Daily Limit: ~50 images/day</li>
              <li>• Max Batch Cost: $50</li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

### 6. Sub-components (`components/settings/Logos*.tsx`) - NUEVOS

| Componente | Descripción | Complejidad |
|------------|-------------|-------------|
| `LogosLeagueSelector.tsx` | Dropdown de ligas con contadores | Baja |
| `LogosGeneratePanel.tsx` | Form para iniciar generación | Media |
| `LogosBatchProgress.tsx` | Progress bar con pause/resume/cancel | Media |
| `LogosReviewGrid.tsx` | Grid de equipos con approve/reject | Alta |
| `LogosTeamCard.tsx` | Card individual de equipo con previews | Media |

### 7. Modificaciones a Archivos Existentes

#### `lib/types/settings.ts`
```diff
 export type SettingsSection =
   | "general"
   | "timezone"
   | "notifications"
   | "api_keys"
   | "model_versions"
   | "feature_flags"
   | "users"
-  | "ia_features";
+  | "ia_features"
+  | "logos";

 export const SETTINGS_SECTION_LABELS: Record<SettingsSection, string> = {
   ...
   ia_features: "IA Features",
+  logos: "3D Logos",
 };

 export const SETTINGS_SECTIONS: SettingsSection[] = [
   ...
   "ia_features",
+  "logos",
 ];
```

#### `components/settings/SettingsNav.tsx`
```diff
+import { Image } from "lucide-react";

 const sectionIcons: Record<SettingsSection, React.ReactNode> = {
   ...
   ia_features: <Brain className="h-4 w-4" strokeWidth={1.5} />,
+  logos: <Image className="h-4 w-4" strokeWidth={1.5} />,
 };
```

#### `components/settings/index.ts`
```diff
 export { IaFeaturesSection } from "./IaFeaturesSection";
+export { LogosSection } from "./LogosSection";
```

#### `app/(shell)/settings/page.tsx`
```diff
+import { LogosSection } from "@/components/settings";

 function SettingsContent({ section, ... }) {
   switch (section) {
     ...
     case "ia_features":
       return <IaFeaturesSection />;
+    case "logos":
+      return <LogosSection />;
     default:
       ...
   }
 }
```

#### `lib/hooks/index.ts`
```diff
+export * from "./use-logos";
```

#### `lib/types/index.ts`
```diff
+export * from "./logos";
```

---

## Wireframes UI

### Tab: Generate

```
┌─────────────────────────────────────────────────────────────┐
│ 3D Logos                                                     │
│ Generate and review 3D logo variants for team shields        │
├─────────────────────────────────────────────────────────────┤
│ [Generate] [Review] [Config]                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Select League                                                │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Colombia - Primera A                              ▼      ││
│ └──────────────────────────────────────────────────────────┘│
│                                                              │
│ Teams: 20 total | 18 pending | 2 ready | 0 errors           │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Generation Settings                                       ││
│ │                                                           ││
│ │ Mode: [Full 3D (3 variants) ▼]                           ││
│ │ Model: [Imagen 3 (FREE) ▼]                               ││
│ │                                                           ││
│ │ Estimated Cost: $0.00 (Free Tier)                        ││
│ │ Images: 54 (18 teams × 3 variants)                       ││
│ │                                                           ││
│ │ ⚠️ Free tier limit: ~50 imgs/day                          ││
│ │                                                           ││
│ │                              [Start Generation]           ││
│ └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Tab: Generate (Batch Running)

```
┌─────────────────────────────────────────────────────────────┐
│ Batch Progress                                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Colombia - Primera A                                         │
│ Model: imagen-3 | Mode: full_3d                             │
│                                                              │
│ ████████████░░░░░░░░░░░░░░░░░░ 40%                          │
│                                                              │
│ Teams: 8/20 processed | 0 failed                            │
│ Images: 24/60 generated                                      │
│ Cost: $0.00 / $0.00                                         │
│ Time: 3m 24s elapsed                                         │
│                                                              │
│ [Pause] [Cancel]                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Tab: Review

```
┌─────────────────────────────────────────────────────────────┐
│ Review Generated Logos                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Colombia - Primera A | 20 teams | Filter: [All ▼]           │
│                                                              │
│ [Approve All Pending] [Reject All Pending]                   │
│                                                              │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│ │ América Cali│ │ Atlético Nal│ │ Dep. Cali   │             │
│ │ ┌───┬───┬───┤ │ ┌───┬───┬───┤ │ ┌───┬───┬───┤             │
│ │ │ F │ R │ L │ │ │ F │ R │ L │ │ │ F │ R │ L │             │
│ │ └───┴───┴───┤ │ └───┴───┴───┤ │ └───┴───┴───┤             │
│ │ ○ Pending   │ │ ✓ Approved  │ │ ✗ Error     │             │
│ │ [✓] [✗] [↻] │ │             │ │ [↻ Retry]   │             │
│ └─────────────┘ └─────────────┘ └─────────────┘             │
│                                                              │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│ │ Millonarios │ │ Santa Fe    │ │ Junior      │             │
│ │ ...         │ │ ...         │ │ ...         │             │
│ └─────────────┘ └─────────────┘ └─────────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Orden de Implementación

### Fase 1: Infraestructura (1 día)
1. `lib/types/logos.ts` - Types
2. `lib/api/logos.ts` - API client
3. `app/api/logos/[...path]/route.ts` - Proxy
4. `lib/hooks/use-logos.ts` - Hooks
5. Modificar `lib/types/settings.ts` - Agregar sección
6. Modificar `lib/hooks/index.ts` - Export hooks

### Fase 2: Componentes Base (1 día)
1. `components/settings/LogosSection.tsx` - Sección principal
2. `components/settings/LogosLeagueSelector.tsx` - Selector de liga
3. Modificar Settings Nav y Page
4. Export en `components/settings/index.ts`

### Fase 3: Generación (1 día)
1. `components/settings/LogosGeneratePanel.tsx` - Panel de generación
2. `components/settings/LogosBatchProgress.tsx` - Progress con polling

### Fase 4: Revisión (1-2 días)
1. `components/settings/LogosReviewGrid.tsx` - Grid de revisión
2. `components/settings/LogosTeamCard.tsx` - Card de equipo
3. Bulk approve/reject functionality

### Fase 5: Polish (0.5 días)
1. Error states
2. Loading skeletons
3. Toast notifications
4. Config tab

---

## Dependencias

- **Backend endpoints**: Ya desplegados y funcionando
- **TanStack Query**: Ya instalado
- **Sonner (toasts)**: Ya instalado
- **Lucide icons**: Ya instalado
- **Tailwind/shadcn**: Ya configurado

---

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Rate limit free tier | Mostrar warning en UI, deshabilitar botón si >50 imgs |
| Batch largo se pierde | Polling con refetchInterval, persist batchId en localStorage |
| Imágenes no cargan | Fallback a URL de API-Football |
| CORS en proxy | Next.js API routes ya manejan esto |

---

## Verificación

### Criterios de Aceptación
1. [ ] Puedo seleccionar una liga del dropdown
2. [ ] Veo el conteo de equipos (pending/ready/error)
3. [ ] Puedo iniciar generación con estimación de costo
4. [ ] Progress bar actualiza en tiempo real
5. [ ] Puedo pausar/resumir/cancelar batch
6. [ ] Puedo ver previews de los 3 variants
7. [ ] Puedo aprobar/rechazar individual y bulk
8. [ ] Errores se muestran claramente

---

## Referencias

- Spec completo: [docs/TEAM_LOGOS_3D_SPEC.md](TEAM_LOGOS_3D_SPEC.md)
- Plan backend: [docs/LOGOS_PLAN.md](LOGOS_PLAN.md) (archivo anterior en `.claude/plans/`)
- Endpoints: [app/logos/routes.py](../app/logos/routes.py)
