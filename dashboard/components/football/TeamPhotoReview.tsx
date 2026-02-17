"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";
import { SurfaceCard } from "@/components/ui/surface-card";
import { CheckCircle, Crop, ImageIcon, XCircle } from "lucide-react";

interface Candidate {
  id: number;
  player_external_id: number;
  player_name: string;
  team_name: string;
  source: string;
  quality_score: number;
  candidate_url: string;
  current_url: string;
  width: number | null;
  height: number | null;
  identity_score: number | null;
}

interface TeamPhotoReviewProps {
  teamId: number;
  teamName: string;
}

type ManualCrop = {
  x: number;
  y: number;
  size: number;
  source_width: number;
  source_height: number;
};

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function buildDefaultCrop(sourceWidth: number, sourceHeight: number): ManualCrop {
  const base = Math.min(sourceWidth, sourceHeight);
  const size = Math.max(64, Math.round(base * 0.62));
  const x = Math.round((sourceWidth - size) / 2);
  const y = Math.round((sourceHeight - size) * 0.12);
  return {
    x: clamp(x, 0, Math.max(0, sourceWidth - size)),
    y: clamp(y, 0, Math.max(0, sourceHeight - size)),
    size,
    source_width: sourceWidth,
    source_height: sourceHeight,
  };
}

function clampCrop(crop: ManualCrop): ManualCrop {
  const minSize = 64;
  const maxSize = Math.max(1, Math.min(crop.source_width, crop.source_height));
  const size = clamp(Math.round(crop.size), minSize, maxSize);
  const x = clamp(Math.round(crop.x), 0, Math.max(0, crop.source_width - size));
  const y = clamp(Math.round(crop.y), 0, Math.max(0, crop.source_height - size));
  return { ...crop, x, y, size };
}

export function TeamPhotoReview({ teamId, teamName }: TeamPhotoReviewProps) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ approved: 0, rejected: 0 });
  const [imgErrors, setImgErrors] = useState<Record<string, boolean>>({});
  const containerRef = useRef<HTMLDivElement>(null);
  const candidateAdjustImgRef = useRef<HTMLImageElement>(null);

  const [adjustMode, setAdjustMode] = useState(false);
  const [manualCrop, setManualCrop] = useState<ManualCrop | null>(null); // committed
  const [draftCrop, setDraftCrop] = useState<ManualCrop | null>(null); // edit-in-progress

  // Fetch candidates for this team
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setCandidates([]);
    setCurrentIndex(0);
    setStats({ approved: 0, rejected: 0 });

    async function load() {
      try {
        const res = await fetch(
          `/api/photos/review?status=pending_review&team_id=${teamId}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) {
          setCandidates(data.candidates || []);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [teamId]);

  const current = candidates[currentIndex];
  const remaining = candidates.length - currentIndex;

  // Reset per-candidate UI state
  useEffect(() => {
    setAdjustMode(false);
    setManualCrop(null);
    setDraftCrop(null);
  }, [current?.id]);

  const handleAction = useCallback(
    async (action: "approve" | "reject") => {
      if (!current) return;
      try {
        const res = await fetch("/api/photos/review", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: current.id,
            action,
            manual_crop: action === "approve" ? manualCrop : undefined,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        setStats((s) => ({
          ...s,
          [action === "approve" ? "approved" : "rejected"]:
            s[action === "approve" ? "approved" : "rejected"] + 1,
        }));
        setCurrentIndex((i) => i + 1);
        setImgErrors({});
        setAdjustMode(false);
        setManualCrop(null);
        setDraftCrop(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Action failed");
      }
    },
    [current, manualCrop]
  );

  // Scoped keyboard handler (only when container has focus)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (adjustMode) return;
      if (e.key === "ArrowRight" || e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        handleAction("approve");
      } else if (e.key === "ArrowLeft" || e.key === "Backspace") {
        e.preventDefault();
        e.stopPropagation();
        handleAction("reject");
      }
    },
    [adjustMode, handleAction]
  );

  const beginAdjust = useCallback(() => {
    if (!current?.candidate_url) return;
    setError(null);
    setImgErrors((e) => ({ ...e, candidate: false }));
    setAdjustMode(true);
    setDraftCrop(manualCrop);
  }, [current?.candidate_url, manualCrop]);

  const cancelAdjust = useCallback(() => {
    setAdjustMode(false);
    setDraftCrop(null);
  }, []);

  const applyAdjust = useCallback(() => {
    if (!draftCrop) return;
    setManualCrop(clampCrop(draftCrop));
    setAdjustMode(false);
    setDraftCrop(null);
  }, [draftCrop]);

  // Interaction: drag/resize overlay in adjust mode
  type Interaction =
    | {
        kind: "move";
        startClientX: number;
        startClientY: number;
        startCrop: ManualCrop;
        scale: number;
      }
    | {
        kind: "resize";
        corner: "nw" | "ne" | "sw" | "se";
        startClientX: number;
        startClientY: number;
        startCrop: ManualCrop;
        scale: number;
      };

  const interactionRef = useRef<Interaction | null>(null);

  const getAdjustScale = useCallback((): number | null => {
    if (!candidateAdjustImgRef.current) return null;
    if (!draftCrop) return null;
    const rect = candidateAdjustImgRef.current.getBoundingClientRect();
    if (!rect.width || !draftCrop.source_width) return null;
    return rect.width / draftCrop.source_width;
  }, [draftCrop]);

  const endInteraction = useCallback(() => {
    interactionRef.current = null;
    window.removeEventListener("mousemove", onWindowMouseMove);
  }, []);

  const onWindowMouseMove = useCallback(
    (e: MouseEvent) => {
      const interaction = interactionRef.current;
      if (!interaction) return;
      setDraftCrop((prev) => {
        const base = prev ?? interaction.startCrop;
        const crop = interaction.startCrop;
        if (!crop) return base;
        const dx = (e.clientX - interaction.startClientX) / interaction.scale;
        const dy = (e.clientY - interaction.startClientY) / interaction.scale;

        const sw = crop.source_width;
        const sh = crop.source_height;

        if (interaction.kind === "move") {
          const next = clampCrop({
            ...crop,
            x: crop.x + dx,
            y: crop.y + dy,
          });
          // Ensure bounds even if source dims are tiny/invalid
          return {
            ...next,
            x: clamp(next.x, 0, Math.max(0, sw - next.size)),
            y: clamp(next.y, 0, Math.max(0, sh - next.size)),
          };
        }

        // Resize
        const corner = interaction.corner;
        const start = crop;
        const minSize = 64;
        const maxSize = Math.max(1, Math.min(sw, sh));

        let nextSize = start.size;
        let nextX = start.x;
        let nextY = start.y;

        const dominant = (a: number, b: number) =>
          Math.abs(a) >= Math.abs(b) ? a : b;

        if (corner === "se") {
          const delta = dominant(dx, dy);
          nextSize = start.size + delta;
        } else if (corner === "nw") {
          const delta = dominant(dx, dy);
          nextSize = start.size - delta;
          const shift = start.size - nextSize;
          nextX = start.x + shift;
          nextY = start.y + shift;
        } else if (corner === "ne") {
          const delta = dominant(dx, -dy);
          nextSize = start.size + delta;
          const shift = start.size - nextSize;
          nextY = start.y + shift;
        } else if (corner === "sw") {
          const delta = dominant(-dx, dy);
          nextSize = start.size + delta;
          const shift = start.size - nextSize;
          nextX = start.x + shift;
        }

        nextSize = clamp(nextSize, minSize, maxSize);
        const next = clampCrop({
          ...start,
          x: nextX,
          y: nextY,
          size: nextSize,
        });
        return next;
      });
    },
    [setDraftCrop]
  );

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => () => endInteraction(), [endInteraction]);

  const startMove = useCallback(
    (e: React.MouseEvent) => {
      if (!draftCrop) return;
      const scale = getAdjustScale();
      if (!scale) return;
      e.preventDefault();
      e.stopPropagation();
      interactionRef.current = {
        kind: "move",
        startClientX: e.clientX,
        startClientY: e.clientY,
        startCrop: draftCrop,
        scale,
      };
      window.addEventListener("mousemove", onWindowMouseMove);
      window.addEventListener("mouseup", endInteraction, { once: true });
    },
    [draftCrop, endInteraction, getAdjustScale, onWindowMouseMove]
  );

  const startResize = useCallback(
    (corner: "nw" | "ne" | "sw" | "se") =>
      (e: React.MouseEvent) => {
        if (!draftCrop) return;
        const scale = getAdjustScale();
        if (!scale) return;
        e.preventDefault();
        e.stopPropagation();
        interactionRef.current = {
          kind: "resize",
          corner,
          startClientX: e.clientX,
          startClientY: e.clientY,
          startCrop: draftCrop,
          scale,
        };
        window.addEventListener("mousemove", onWindowMouseMove);
        window.addEventListener("mouseup", endInteraction, { once: true });
      },
    [draftCrop, endInteraction, getAdjustScale, onWindowMouseMove]
  );

  const onAdjustWheel = useCallback(
    (e: React.WheelEvent) => {
      if (!draftCrop) return;
      const dir = e.deltaY < 0 ? -1 : 1; // scroll up = zoom in (smaller crop)
      e.preventDefault();
      const step = Math.max(8, Math.round(draftCrop.size * 0.05));
      const nextSize = draftCrop.size + dir * step;
      const cx = draftCrop.x + draftCrop.size / 2;
      const cy = draftCrop.y + draftCrop.size / 2;
      const next = clampCrop({
        ...draftCrop,
        size: nextSize,
        x: cx - nextSize / 2,
        y: cy - nextSize / 2,
      });
      setDraftCrop(next);
    },
    [draftCrop]
  );

  const onAdjustImageLoad = useCallback(
    (e: React.SyntheticEvent<HTMLImageElement>) => {
      const img = e.currentTarget;
      const w = img.naturalWidth || current?.width || 0;
      const h = img.naturalHeight || current?.height || 0;
      if (!w || !h) return;

      setDraftCrop((prev) => {
        if (prev) {
          // If dims changed (rare), scale crop to keep relative framing.
          if (
            prev.source_width !== w ||
            prev.source_height !== h
          ) {
            const sx = w / prev.source_width;
            const sy = h / prev.source_height;
            const s = (sx + sy) / 2;
            return clampCrop({
              ...prev,
              x: prev.x * sx,
              y: prev.y * sy,
              size: prev.size * s,
              source_width: w,
              source_height: h,
            });
          }
          return prev;
        }
        return buildDefaultCrop(w, h);
      });
    },
    [current?.height, current?.width]
  );

  if (loading) {
    return (
      <SurfaceCard className="space-y-2">
        <h4 className="text-sm font-medium flex items-center gap-1.5">
          <ImageIcon className="h-4 w-4 text-primary" />
          Player Photos
        </h4>
        <p className="text-xs text-muted-foreground">Loading candidates...</p>
      </SurfaceCard>
    );
  }

  if (error) {
    return (
      <SurfaceCard className="space-y-2">
        <h4 className="text-sm font-medium flex items-center gap-1.5">
          <ImageIcon className="h-4 w-4 text-primary" />
          Player Photos
        </h4>
        <p className="text-xs text-red-500">{error}</p>
      </SurfaceCard>
    );
  }

  if (candidates.length === 0) {
    return (
      <SurfaceCard className="space-y-2">
        <h4 className="text-sm font-medium flex items-center gap-1.5">
          <ImageIcon className="h-4 w-4 text-primary" />
          Player Photos
        </h4>
        <p className="text-xs text-muted-foreground">
          No pending candidates for {teamName}.
        </p>
      </SurfaceCard>
    );
  }

  // All reviewed
  if (!current) {
    return (
      <SurfaceCard className="space-y-3">
        <h4 className="text-sm font-medium flex items-center gap-1.5">
          <ImageIcon className="h-4 w-4 text-primary" />
          Player Photos
        </h4>
        <div className="flex items-center gap-4 text-xs">
          <span className="text-green-500">{stats.approved} approved</span>
          <span className="text-red-500">{stats.rejected} rejected</span>
          <span className="text-muted-foreground">
            of {candidates.length} total
          </span>
        </div>
        <p className="text-xs text-muted-foreground">
          Review complete for {teamName}. Run{" "}
          <code className="bg-muted px-1 rounded">--approved-only</code> to
          process.
        </p>
      </SurfaceCard>
    );
  }

  return (
    <SurfaceCard className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium flex items-center gap-1.5">
          <ImageIcon className="h-4 w-4 text-primary" />
          Player Photos
        </h4>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-green-500 tabular-nums">
            {stats.approved} ok
          </span>
          <span className="text-red-500 tabular-nums">
            {stats.rejected} no
          </span>
          <span className="text-muted-foreground tabular-nums">
            {remaining} left
          </span>
        </div>
      </div>

      {/* Focusable container for scoped keyboard shortcuts */}
      <div
        ref={containerRef}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        className="outline-none focus:ring-1 focus:ring-primary/30 rounded-lg"
      >
        {/* Player info */}
        <div className="mb-2">
          <p className="text-sm font-medium text-foreground">
            {current.player_name}
          </p>
          <p className="text-[10px] text-muted-foreground">
            {current.source} | Q:{current.quality_score} | ID:{current.identity_score} |{" "}
            {current.width}x{current.height}
          </p>
        </div>

        {/* Comparison: current vs candidate */}
        <div className="flex items-center gap-3">
          {/* Current */}
          <div className="flex flex-col items-center gap-1 flex-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Current
            </p>
            <div className="relative w-full aspect-square rounded-lg border border-red-500/20 bg-muted overflow-hidden">
              {!imgErrors["current"] ? (
                <Image
                  src={current.current_url}
                  alt="Current"
                  fill
                  className="object-contain"
                  unoptimized
                  onError={() =>
                    setImgErrors((e) => ({ ...e, current: true }))
                  }
                />
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground text-[10px]">
                  No image
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-col items-center gap-2">
            <button
              onClick={() => handleAction("reject")}
              disabled={adjustMode}
              className="w-10 h-10 rounded-full bg-red-500/10 border border-red-500/40 flex items-center justify-center hover:bg-red-500/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              title="Reject (Left Arrow)"
            >
              <XCircle className="h-5 w-5 text-red-500" />
            </button>
            <button
              onClick={beginAdjust}
              disabled={adjustMode || !current.candidate_url}
              className="w-10 h-10 rounded-full bg-muted border border-border flex items-center justify-center hover:bg-muted/70 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              title="Ajustar crop (manual)"
            >
              <Crop className="h-5 w-5 text-foreground/80" />
            </button>
            <button
              onClick={() => handleAction("approve")}
              disabled={adjustMode}
              className="w-10 h-10 rounded-full bg-green-500/10 border border-green-500/40 flex items-center justify-center hover:bg-green-500/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              title="Approve (Right Arrow)"
            >
              <CheckCircle className="h-5 w-5 text-green-500" />
            </button>
          </div>

          {/* Candidate — face-crop preview from backend */}
          <div className="flex flex-col items-center gap-1 flex-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Candidate
            </p>
            {adjustMode ? (
              <div className="w-full rounded-lg border border-border bg-muted overflow-hidden">
                {!imgErrors["candidate"] ? (
                  <div className="p-2">
                    <div className="max-h-[340px] overflow-auto">
                      <div className="relative">
                        <img
                          ref={candidateAdjustImgRef}
                          src={current.candidate_url}
                          alt="Candidate original"
                          className="block w-full h-auto select-none"
                          draggable={false}
                          onDragStart={(e) => e.preventDefault()}
                          onError={() =>
                            setImgErrors((e) => ({ ...e, candidate: true }))
                          }
                          onLoad={onAdjustImageLoad}
                        />

                        {draftCrop ? (() => {
                          const scale = getAdjustScale() || 1;
                          const left = draftCrop.x * scale;
                          const top = draftCrop.y * scale;
                          const size = draftCrop.size * scale;
                          return (
                            <div
                              className="absolute border-2 border-primary/80 bg-white/5 cursor-move"
                              style={{
                                left,
                                top,
                                width: size,
                                height: size,
                                boxShadow: "0 0 0 9999px rgba(0,0,0,0.35)",
                              }}
                              onMouseDown={startMove}
                              onWheel={onAdjustWheel}
                              title="Drag para mover · Scroll para zoom"
                            >
                              {/* Resize handles */}
                              <div
                                className="absolute -left-1.5 -top-1.5 h-3 w-3 bg-primary border border-white cursor-nwse-resize"
                                onMouseDown={startResize("nw")}
                              />
                              <div
                                className="absolute -right-1.5 -top-1.5 h-3 w-3 bg-primary border border-white cursor-nesw-resize"
                                onMouseDown={startResize("ne")}
                              />
                              <div
                                className="absolute -left-1.5 -bottom-1.5 h-3 w-3 bg-primary border border-white cursor-nesw-resize"
                                onMouseDown={startResize("sw")}
                              />
                              <div
                                className="absolute -right-1.5 -bottom-1.5 h-3 w-3 bg-primary border border-white cursor-nwse-resize"
                                onMouseDown={startResize("se")}
                              />
                            </div>
                          );
                        })() : null}
                      </div>
                    </div>

                    <div className="flex items-center justify-between gap-2 mt-2">
                      <button
                        onClick={cancelAdjust}
                        className="px-2 py-1 rounded border border-border text-[10px] hover:bg-muted/70"
                      >
                        Cancelar
                      </button>
                      <button
                        onClick={applyAdjust}
                        disabled={!draftCrop}
                        className="px-2 py-1 rounded bg-primary text-primary-foreground text-[10px] disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Aplicar
                      </button>
                    </div>

                    {draftCrop ? (
                      <p className="text-[10px] text-muted-foreground mt-1 tabular-nums">
                        x={Math.round(draftCrop.x)} y={Math.round(draftCrop.y)} size={Math.round(draftCrop.size)} ·
                        src={draftCrop.source_width}x{draftCrop.source_height}
                      </p>
                    ) : (
                      <p className="text-[10px] text-muted-foreground mt-1">
                        Cargando…
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-40 text-muted-foreground text-[10px]">
                    No image
                  </div>
                )}
              </div>
            ) : (
              <div
                className="relative w-full aspect-square rounded-lg border border-green-500/20 bg-muted overflow-hidden"
              >
                {!imgErrors["candidate"] ? (
                  <>
                    <Image
                      src={manualCrop
                        ? `/api/photos/preview?id=${current.id}&cx=${Math.round(manualCrop.x)}&cy=${Math.round(manualCrop.y)}&cs=${Math.round(manualCrop.size)}&sw=${manualCrop.source_width}&sh=${manualCrop.source_height}`
                        : `/api/photos/preview?id=${current.id}`
                      }
                      alt="Candidate"
                      fill
                      className="object-cover"
                      unoptimized
                      onError={() =>
                        setImgErrors((e) => ({ ...e, candidate: true }))
                      }
                    />
                    {manualCrop && (
                      <div className="absolute bottom-1 right-1 bg-black/40 text-white text-[9px] px-1 rounded">
                        manual
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-muted-foreground text-[10px]">
                    No image
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Hint */}
        <p className="text-[10px] text-muted-foreground text-center mt-2">
          Click here then use arrow keys: ← reject · → approve
        </p>
      </div>

      {/* Progress */}
      <div>
        <div className="w-full bg-muted rounded-full h-1.5">
          <div
            className="bg-green-500 h-1.5 rounded-full transition-all"
            style={{
              width: `${(currentIndex / candidates.length) * 100}%`,
            }}
          />
        </div>
        <p className="text-[10px] text-muted-foreground mt-0.5 text-center tabular-nums">
          {currentIndex} / {candidates.length}
        </p>
      </div>
    </SurfaceCard>
  );
}
