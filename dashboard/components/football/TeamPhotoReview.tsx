"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";
import { SurfaceCard } from "@/components/ui/surface-card";
import { CheckCircle, Crop, ImageIcon, Link2, Loader2, XCircle } from "lucide-react";

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
  initialFullscreen?: boolean;
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

function clampCrop(crop: ManualCrop): ManualCrop {
  const minSize = 64;
  const maxSize = Math.max(1, Math.min(crop.source_width, crop.source_height));
  const size = clamp(Math.round(crop.size), minSize, maxSize);
  const x = clamp(Math.round(crop.x), 0, Math.max(0, crop.source_width - size));
  const y = clamp(Math.round(crop.y), 0, Math.max(0, crop.source_height - size));
  return { ...crop, x, y, size };
}

export function TeamPhotoReview({ teamId, teamName, initialFullscreen = false }: TeamPhotoReviewProps) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ approved: 0, rejected: 0 });
  const [imgErrors, setImgErrors] = useState<Record<string, boolean>>({});
  const containerRef = useRef<HTMLDivElement>(null);
  const cropFrameRef = useRef<HTMLDivElement>(null);

  const [adjustMode, setAdjustMode] = useState(false);
  const [manualCrop, setManualCrop] = useState<ManualCrop | null>(null);
  const [cleanPreviewUrl, setCleanPreviewUrl] = useState<string | null>(null);
  const [cleanLoading, setCleanLoading] = useState(false);
  // Pan/zoom adjust state (fixed frame, move image)
  const CROP_FRAME = 360;
  const [adjZoom, setAdjZoom] = useState(1.0);
  const [adjPan, setAdjPan] = useState({ x: 0, y: 0 });
  const [adjNat, setAdjNat] = useState({ w: 0, h: 0 });
  // Ref to avoid stale closures in native wheel handler
  const adjRef = useRef({ zoom: 1.0, panX: 0, panY: 0, natW: 0, natH: 0 });
  adjRef.current = { zoom: adjZoom, panX: adjPan.x, panY: adjPan.y, natW: adjNat.w, natH: adjNat.h };
  const [fullscreen, setFullscreen] = useState(initialFullscreen);
  const [faceData, setFaceData] = useState<{
    detected: boolean;
    confidence?: number;
    bbox?: { x: number; y: number; w: number; h: number };
    keypoints?: Record<string, { x: number; y: number }>;
  } | null>(null);

  // Manual URL add
  const [urlInput, setUrlInput] = useState("");
  const [urlPlayerExtId, setUrlPlayerExtId] = useState("");
  const [urlSubmitting, setUrlSubmitting] = useState(false);
  const [urlMessage, setUrlMessage] = useState<{ type: "ok" | "err"; text: string } | null>(null);

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

  // Reset per-candidate UI state + revoke blob + fetch face data
  useEffect(() => {
    // Keep adjust mode active during fullscreen review session
    if (fullscreen) setAdjustMode(true);
    setManualCrop(null);
    setImgErrors({});
    setCleanLoading(false);
    setCleanPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
    setFaceData(null);
    if (!current?.id) return;
    let cancelled = false;
    fetch(`/api/photos/face-detect?id=${current.id}`)
      .then((r) => { console.log("[face-detect] status:", r.status); return r.ok ? r.json() : null; })
      .then((d) => { console.log("[face-detect] data:", d); if (!cancelled && d) setFaceData(d); })
      .catch((e) => { console.error("[face-detect] error:", e); });
    return () => { cancelled = true; };
  }, [current?.id, fullscreen]);

  const handleAction = useCallback(
    async (action: "approve" | "reject") => {
      if (!current || submitting) return;
      setSubmitting(true);
      try {
        const res = await fetch("/api/photos/review", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: current.id,
            action,
            manual_crop: action === "approve" ? manualCrop : undefined,
            face_detect: action === "approve" && faceData?.detected ? faceData : undefined,
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
        setManualCrop(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Action failed");
      } finally {
        setSubmitting(false);
      }
    },
    [current, manualCrop, faceData, submitting]
  );

  // ── Add candidate from URL ──────────────────────────────────────────
  const handleAddUrl = useCallback(async () => {
    const url = urlInput.trim();
    const extId = urlPlayerExtId.trim();
    if (!url || !extId) return;
    if (!url.startsWith("http")) {
      setUrlMessage({ type: "err", text: "URL must start with http" });
      return;
    }
    setUrlSubmitting(true);
    setUrlMessage(null);
    try {
      const res = await fetch("/api/photos/add-candidate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_external_id: Number(extId), image_url: url }),
      });
      const data = await res.json();
      if (!res.ok) {
        setUrlMessage({ type: "err", text: data.error || `HTTP ${res.status}` });
        return;
      }
      setUrlMessage({ type: "ok", text: `Added #${data.id} — ${data.player_name}` });
      setUrlInput("");
      setUrlPlayerExtId("");
      // Reload candidates to include the new one
      const reload = await fetch(`/api/photos/review?status=pending_review&team_id=${teamId}`);
      if (reload.ok) {
        const rData = await reload.json();
        setCandidates(rData.candidates || []);
      }
    } catch (e) {
      setUrlMessage({ type: "err", text: e instanceof Error ? e.message : "Failed" });
    } finally {
      setUrlSubmitting(false);
    }
  }, [urlInput, urlPlayerExtId, teamId]);

  // Scoped keyboard handler (only when container has focus)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (adjustMode) return;
      if (e.key === "ArrowUp") {
        e.preventDefault();
        e.stopPropagation();
        handleAction("approve");
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        e.stopPropagation();
        handleAction("reject");
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        e.stopPropagation();
        setCurrentIndex((i) => Math.min(i + 1, candidates.length - 1));
        setImgErrors({});
        setManualCrop(null);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        e.stopPropagation();
        setCurrentIndex((i) => Math.max(i - 1, 0));
        setImgErrors({});
        setManualCrop(null);
      }
    },
    [adjustMode, handleAction, candidates.length]
  );

  // ── Pan/zoom helpers ──────────────────────────────────────────────
  const clampPanFn = useCallback(
    (px: number, py: number, natW: number, natH: number, zoom: number) => {
      const ds = natW > 0 ? (CROP_FRAME / natW) * zoom : 1;
      const imgW = natW * ds;
      const imgH = natH * ds;
      return {
        x: clamp(px, Math.min(0, CROP_FRAME - imgW), 0),
        y: clamp(py, Math.min(0, CROP_FRAME - imgH), 0),
      };
    },
    [CROP_FRAME]
  );

  const beginAdjust = useCallback(() => {
    if (!current?.candidate_url) return;
    setError(null);
    setImgErrors((e) => ({ ...e, candidate: false }));
    setAdjustMode(true);
  }, [current?.candidate_url]);

  const cancelAdjust = useCallback(() => {
    setAdjustMode(false);
  }, []);

  const applyAdjust = useCallback(() => {
    if (!adjNat.w || !current) return;
    const ds = (CROP_FRAME / adjNat.w) * adjZoom;
    const crop = clampCrop({
      x: -adjPan.x / ds,
      y: -adjPan.y / ds,
      size: CROP_FRAME / ds,
      source_width: adjNat.w,
      source_height: adjNat.h,
    });
    setManualCrop(crop);
    setAdjustMode(false);
    // Fetch clean preview (crop + PhotoRoom bg removal)
    setCleanLoading(true);
    setCleanPreviewUrl(null);
    const qs = `id=${current.id}&cx=${Math.round(crop.x)}&cy=${Math.round(crop.y)}&cs=${Math.round(crop.size)}&sw=${crop.source_width}&sh=${crop.source_height}&clean=1`;
    console.log("[clean-preview] fetching:", `/api/photos/preview?${qs}`);
    fetch(`/api/photos/preview?${qs}`)
      .then((res) => {
        console.log("[clean-preview] status:", res.status);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        console.log("[clean-preview] got blob:", blob.size, "bytes");
        setCleanPreviewUrl(URL.createObjectURL(blob));
      })
      .catch((err) => {
        console.error("[clean-preview] error:", err);
        setCleanPreviewUrl(null);
      })
      .finally(() => setCleanLoading(false));
  }, [CROP_FRAME, adjNat, adjZoom, adjPan, current]);

  const onAdjustImageLoad = useCallback(
    (e: React.SyntheticEvent<HTMLImageElement>) => {
      const w = e.currentTarget.naturalWidth || 0;
      const h = e.currentTarget.naturalHeight || 0;
      if (!w || !h) return;
      setAdjNat({ w, h });
      if (manualCrop && manualCrop.source_width > 0) {
        const bs = CROP_FRAME / manualCrop.source_width;
        const z = CROP_FRAME / (manualCrop.size * bs);
        const ds = bs * z;
        setAdjZoom(z);
        setAdjPan(clampPanFn(-manualCrop.x * ds, -manualCrop.y * ds, w, h, z));
      } else {
        // Start at 230% zoom so face is prominent
        const initZ = 2.3;
        const ds = (CROP_FRAME / w) * initZ;
        const imgH = h * ds;
        // Center horizontally, bias toward face area
        const panX = -(w * ds - CROP_FRAME) / 2;
        const panY = -(imgH - CROP_FRAME) * 0.28;
        setAdjZoom(initZ);
        setAdjPan(clampPanFn(panX, panY, w, h, initZ));
      }
    },
    [CROP_FRAME, manualCrop, clampPanFn]
  );

  // Re-center on detected face when faceData arrives
  useEffect(() => {
    if (!faceData?.detected || !faceData.bbox || !adjustMode || adjNat.w === 0 || manualCrop) return;
    const b = faceData.bbox;
    const faceCX = b.x + b.w / 2;
    const faceCY = b.y + b.h / 2;
    // Shift up slightly so face is in upper third (head framing)
    const targetY = faceCY - b.h * 0.15;
    const z = adjZoom;
    const ds = (CROP_FRAME / adjNat.w) * z;
    const panX = CROP_FRAME / 2 - faceCX * adjNat.w * ds;
    const panY = CROP_FRAME / 2 - targetY * adjNat.h * ds;
    setAdjPan(clampPanFn(panX, panY, adjNat.w, adjNat.h, z));
  }, [faceData, adjustMode, adjNat, adjZoom, manualCrop, clampPanFn, CROP_FRAME]);

  const onAdjustDrag = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const sp = { ...adjPan };
      const sc = { x: e.clientX, y: e.clientY };
      const nat = { ...adjNat };
      const z = adjZoom;
      const onMove = (ev: MouseEvent) => {
        setAdjPan(clampPanFn(sp.x + ev.clientX - sc.x, sp.y + ev.clientY - sc.y, nat.w, nat.h, z));
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [adjPan, adjNat, adjZoom, clampPanFn]
  );

  // Native wheel listener (passive: false) — React onWheel is passive and ignores preventDefault
  useEffect(() => {
    const el = cropFrameRef.current;
    if (!el || !adjustMode) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const s = adjRef.current;
      const dir = e.deltaY < 0 ? 1 : -1;
      const minZ = s.natW > 0 && s.natH > 0 ? Math.max(1.0, s.natW / s.natH) : 1.0;
      const nextZoom = clamp(s.zoom + dir * 0.05, minZ, 6.0);
      if (nextZoom === s.zoom) return;
      const cx = CROP_FRAME / 2;
      const cy = CROP_FRAME / 2;
      const ratio = nextZoom / s.zoom;
      const clamped = clampPanFn(
        cx - (cx - s.panX) * ratio,
        cy - (cy - s.panY) * ratio,
        s.natW, s.natH, nextZoom,
      );
      setAdjZoom(nextZoom);
      setAdjPan(clamped);
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [adjustMode, clampPanFn, CROP_FRAME]);

  // Auto-focus container when fullscreen opens
  useEffect(() => {
    if (fullscreen && containerRef.current) {
      containerRef.current.focus();
    }
  }, [fullscreen, currentIndex]);

  // Escape to close fullscreen
  const handleFullscreenKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setFullscreen(false);
        return;
      }
      handleKeyDown(e);
    },
    [handleKeyDown]
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

  // Summary card (always shown in drawer)
  const summaryCard = (
    <SurfaceCard className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium flex items-center gap-1.5">
          <ImageIcon className="h-4 w-4 text-primary" />
          Player Photos
        </h4>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-green-500 tabular-nums">{stats.approved} ok</span>
          <span className="text-red-500 tabular-nums">{stats.rejected} no</span>
          <span className="text-muted-foreground tabular-nums">{remaining} left</span>
        </div>
      </div>
      <button
        onClick={() => { setFullscreen(true); setAdjustMode(true); }}
        className="w-full py-2 rounded-lg bg-primary/10 border border-primary/30 text-sm text-primary hover:bg-primary/20 transition-colors"
      >
        Review {remaining} candidates
      </button>
      <div className="w-full bg-muted rounded-full h-1.5">
        <div
          className="bg-green-500 h-1.5 rounded-full transition-all"
          style={{ width: `${(currentIndex / candidates.length) * 100}%` }}
        />
      </div>
    </SurfaceCard>
  );

  if (!fullscreen) return summaryCard;

  // ── Fullscreen overlay ──────────────────────────────────────────────
  return (
    <>
      {summaryCard}
      <div className="fixed inset-0 z-[9999] bg-black/90 flex flex-col">
        {/* Slim top bar — URL add + stats + close */}
        <div className="flex items-center justify-between px-6 py-2 border-b border-white/10">
          {/* Left: add candidate by URL */}
          <div className="flex items-center gap-2">
            <Link2 className="h-3.5 w-3.5 text-white/40" />
            <input
              type="text"
              placeholder="Player ext ID"
              value={urlPlayerExtId || (current ? String(current.player_external_id) : "")}
              onChange={(e) => setUrlPlayerExtId(e.target.value)}
              onFocus={() => { if (!urlPlayerExtId && current) setUrlPlayerExtId(String(current.player_external_id)); }}
              className="w-24 bg-white/5 border border-white/15 rounded px-2 py-1 text-xs text-white placeholder:text-white/30 outline-none focus:border-white/30"
            />
            <input
              type="text"
              placeholder="Paste image URL..."
              value={urlInput}
              onChange={(e) => { setUrlInput(e.target.value); setUrlMessage(null); }}
              onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); handleAddUrl(); } }}
              className="w-80 bg-white/5 border border-white/15 rounded px-2 py-1 text-xs text-white placeholder:text-white/30 outline-none focus:border-white/30"
            />
            <button
              onClick={handleAddUrl}
              disabled={urlSubmitting || !urlInput.trim() || !urlPlayerExtId.trim()}
              className="px-2.5 py-1 rounded bg-primary/80 text-primary-foreground text-[11px] hover:bg-primary disabled:opacity-40 flex items-center gap-1"
            >
              {urlSubmitting ? <Loader2 className="h-3 w-3 animate-spin" /> : "Add"}
            </button>
            {urlMessage && (
              <span className={`text-[11px] ${urlMessage.type === "ok" ? "text-green-400" : "text-red-400"}`}>
                {urlMessage.text}
              </span>
            )}
          </div>
          {/* Right: stats + close */}
          <div className="flex items-center gap-5">
            <span className="text-green-400 text-sm tabular-nums">{stats.approved} ok</span>
            <span className="text-red-400 text-sm tabular-nums">{stats.rejected} no</span>
            <span className="text-white/40 text-sm tabular-nums">{remaining} left</span>
            <button
              onClick={() => setFullscreen(false)}
              className="ml-4 text-white/60 hover:text-white text-sm px-3 py-1.5 rounded border border-white/20 hover:border-white/40 transition-colors"
            >
              ESC
            </button>
          </div>
        </div>

        {/* Main content — focusable for keyboard */}
        <div
          ref={containerRef}
          tabIndex={0}
          onKeyDown={handleFullscreenKeyDown}
          className="flex-1 flex flex-col items-center justify-center gap-4 px-8 outline-none"
        >
          {/* Player name — right above photos */}
          <div className="text-center">
            <p className="text-white text-xl font-semibold">{current.player_name}</p>
            <p className="text-white/40 text-xs">
              {current.source} | Q:{current.quality_score} | ID:{current.identity_score} | {current.width}x{current.height}
            </p>
          </div>

          <div className="flex items-center justify-center gap-8">
          {/* Current photo */}
          <div className="flex flex-col items-center gap-3">
            <p className="text-white/50 text-xs uppercase tracking-widest">Current</p>
            <div className="relative w-[384px] h-[384px] rounded-xl border border-white/10 bg-white/5 overflow-hidden">
              {!imgErrors["current"] && current.current_url ? (
                <Image
                  src={current.current_url}
                  alt="Current"
                  fill
                  className="object-contain"
                  unoptimized
                  onError={() => setImgErrors((e) => ({ ...e, current: true }))}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-white/30 text-sm">
                  No image
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-col items-center gap-4">
            <button
              onClick={() => handleAction("reject")}
              disabled={adjustMode || submitting}
              className="w-16 h-16 rounded-full bg-red-500/20 border-2 border-red-500/60 flex items-center justify-center hover:bg-red-500/30 transition-colors disabled:opacity-40"
              title="Reject (↓ Down Arrow)"
            >
              <XCircle className="h-8 w-8 text-red-400" />
            </button>
            <button
              onClick={beginAdjust}
              disabled={adjustMode || submitting || !current.candidate_url}
              className="w-12 h-12 rounded-full bg-white/10 border border-white/20 flex items-center justify-center hover:bg-white/20 transition-colors disabled:opacity-40"
              title="Ajustar crop"
            >
              <Crop className="h-6 w-6 text-white/70" />
            </button>
            <button
              onClick={() => handleAction("approve")}
              disabled={adjustMode || submitting}
              className="w-16 h-16 rounded-full bg-green-500/20 border-2 border-green-500/60 flex items-center justify-center hover:bg-green-500/30 transition-colors disabled:opacity-40"
              title="Approve (↑ Up Arrow)"
            >
              {submitting ? (
                <div className="h-8 w-8 border-2 border-green-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <CheckCircle className="h-8 w-8 text-green-400" />
              )}
            </button>
          </div>

          {/* Candidate photo */}
          <div className="flex flex-col items-center gap-3">
            <p className="text-white/50 text-xs uppercase tracking-widest">Candidate</p>
            {adjustMode ? (
              <>
                <div
                  ref={cropFrameRef}
                  className="relative rounded-xl border-2 border-primary/80 overflow-hidden cursor-grab active:cursor-grabbing bg-black"
                  style={{ width: CROP_FRAME, height: CROP_FRAME }}
                  onMouseDown={onAdjustDrag}
                >
                  {!imgErrors["candidate"] && current.candidate_url ? (
                    <img
                      src={current.candidate_url}
                      alt="Adjust"
                      className="absolute select-none pointer-events-none max-w-none"
                      referrerPolicy="no-referrer"
                      draggable={false}
                      onLoad={onAdjustImageLoad}
                      onError={() => setImgErrors((e) => ({ ...e, candidate: true }))}
                      style={{
                        width: CROP_FRAME * adjZoom,
                        left: adjPan.x,
                        top: adjPan.y,
                      }}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-white/30 text-sm">No image</div>
                  )}
                  {/* Reference grid 6×6 */}
                  <div className="absolute inset-0 pointer-events-none z-[4]">
                    {[1,2,3,4,5].map((i) => (
                      <div key={`v${i}`} className="absolute top-0 bottom-0" style={{ left: `${(i/6)*100}%`, width: 1, background: "rgba(255,255,255,0.10)" }} />
                    ))}
                    {[1,2,3,4,5].map((i) => (
                      <div key={`h${i}`} className="absolute left-0 right-0" style={{ top: `${(i/6)*100}%`, height: 1, background: "rgba(255,255,255,0.10)" }} />
                    ))}
                  </div>
                  {/* Face detection overlay */}
                  {faceData?.detected && faceData.keypoints && adjNat.w > 0 && (() => {
                    const ds = (CROP_FRAME / adjNat.w) * adjZoom;
                    const toX = (nx: number) => nx * adjNat.w * ds + adjPan.x;
                    const toY = (ny: number) => ny * adjNat.h * ds + adjPan.y;
                    const kp = faceData.keypoints!;
                    const re = kp.RIGHT_EYE, le = kp.LEFT_EYE, nose = kp.NOSE_TIP, mouth = kp.MOUTH_CENTER;
                    const eyeMidX = toX((re.x + le.x) / 2);
                    const eyeMidY = toY((re.y + le.y) / 2);
                    return (
                      <div className="absolute inset-0 pointer-events-none z-[5]">
                        {/* Eye line */}
                        <svg className="absolute inset-0 w-full h-full overflow-visible">
                          <line x1={toX(re.x)} y1={toY(re.y)} x2={toX(le.x)} y2={toY(le.y)} stroke="rgba(0,200,255,0.5)" strokeWidth="1" />
                          {/* Vertical center: eye midpoint to mouth */}
                          <line x1={eyeMidX} y1={eyeMidY} x2={toX(mouth.x)} y2={toY(mouth.y)} stroke="rgba(0,200,255,0.35)" strokeWidth="1" strokeDasharray="4,4" />
                        </svg>
                        {/* Keypoint dots */}
                        {[
                          { p: re, c: "rgba(0,200,255,0.7)" },
                          { p: le, c: "rgba(0,200,255,0.7)" },
                          { p: nose, c: "rgba(255,200,0,0.7)" },
                          { p: mouth, c: "rgba(255,100,100,0.7)" },
                        ].map((d, i) => (
                          <div key={i} className="absolute rounded-full" style={{
                            width: 6, height: 6,
                            left: toX(d.p.x) - 3, top: toY(d.p.y) - 3,
                            backgroundColor: d.c,
                            boxShadow: `0 0 4px ${d.c}`,
                          }} />
                        ))}
                        {/* Head bbox (face bbox extended upward for forehead/hair) */}
                        {faceData.bbox && (() => {
                          const b = faceData.bbox;
                          const extraTop = b.h * 0.35;  // extend 35% upward for forehead+hair
                          const extraSide = b.w * 0.08;  // slight horizontal padding
                          const hx = b.x - extraSide;
                          const hy = b.y - extraTop;
                          const hw = b.w + extraSide * 2;
                          const hh = b.h + extraTop;
                          return (
                            <div className="absolute rounded" style={{
                              left: toX(hx),
                              top: toY(hy),
                              width: hw * adjNat.w * ds,
                              height: hh * adjNat.h * ds,
                              border: "1.5px solid rgba(0,200,255,0.35)",
                              borderRadius: "50% 50% 45% 45% / 55% 55% 45% 45%",
                            }} />
                          );
                        })()}
                      </div>
                    );
                  })()}
                  {/* Floating controls inside frame */}
                  <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-3 py-2 bg-gradient-to-t from-black/70 to-transparent pointer-events-auto z-10" onMouseDown={(e) => e.stopPropagation()}>
                    <span className="text-white/50 text-[10px]">Scroll zoom · Drag to move · <span className="text-white">{Math.round(adjZoom * 100)}%</span></span>
                    <div className="flex items-center gap-2">
                      <button onClick={cancelAdjust} className="px-2.5 py-1 rounded border border-white/30 text-[11px] text-white/80 hover:bg-white/10">
                        Cancel
                      </button>
                      <button onClick={applyAdjust} className="px-2.5 py-1 rounded bg-primary text-primary-foreground text-[11px]">
                        Apply
                      </button>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="relative w-[384px] h-[384px] rounded-xl border border-white/10 bg-white/5 overflow-hidden">
                {cleanLoading ? (
                  <div className="flex flex-col items-center justify-center h-full gap-2">
                    <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span className="text-white/40 text-xs">PhotoRoom...</span>
                  </div>
                ) : cleanPreviewUrl ? (
                  <>
                    <Image
                      src={cleanPreviewUrl}
                      alt="Clean preview"
                      fill
                      className="object-cover"
                      unoptimized
                      onError={() => setImgErrors((e) => ({ ...e, candidate: true }))}
                    />
                    <div className="absolute bottom-2 right-2 bg-green-600/70 text-white text-[10px] px-1.5 py-0.5 rounded">clean</div>
                  </>
                ) : !imgErrors["candidate"] ? (
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
                      onError={() => setImgErrors((e) => ({ ...e, candidate: true }))}
                    />
                    {manualCrop && (
                      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-[10px] px-1.5 py-0.5 rounded">manual</div>
                    )}
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-white/30 text-sm">No image</div>
                )}
              </div>
            )}
          </div>

          {/* Source — full original image */}
          <div className="flex flex-col items-center gap-3">
            <p className="text-white/50 text-xs uppercase tracking-widest">Source</p>
            <div className="relative w-[384px] h-[512px] rounded-xl border border-white/10 bg-white/5 overflow-hidden">
              {!imgErrors["source"] && current.candidate_url ? (
                <Image
                  src={current.candidate_url}
                  alt="Source"
                  fill
                  className="object-contain"
                  unoptimized
                  onError={() => setImgErrors((e) => ({ ...e, source: true }))}
                />
              ) : (
                <div className="flex items-center justify-center h-full text-white/30 text-sm">
                  No source
                </div>
              )}
            </div>
          </div>
          </div>

          {/* Progress + hint — below photos */}
          <div className="w-full max-w-2xl">
            <div className="w-full bg-white/10 rounded-full h-1.5 mb-2">
              <div
                className="bg-green-500 h-1.5 rounded-full transition-all"
                style={{ width: `${(currentIndex / candidates.length) * 100}%` }}
              />
            </div>
            <p className="text-white/40 text-xs text-center tabular-nums">
              {currentIndex} / {candidates.length} · ↑ approve · ↓ reject · ←→ nav · ESC close
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
