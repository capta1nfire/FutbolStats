"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";
import { SurfaceCard } from "@/components/ui/surface-card";
import { CheckCircle, ImageIcon, XCircle } from "lucide-react";

interface Candidate {
  id: number;
  player_external_id: number;
  player_name: string;
  team_name: string;
  source: string;
  quality_score: number;
  candidate_url: string;
  current_url: string;
  width: number;
  height: number;
  identity_score: number;
}

interface TeamPhotoReviewProps {
  teamId: number;
  teamName: string;
}

export function TeamPhotoReview({ teamId, teamName }: TeamPhotoReviewProps) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ approved: 0, rejected: 0 });
  const [imgErrors, setImgErrors] = useState<Record<string, boolean>>({});
  const containerRef = useRef<HTMLDivElement>(null);

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

  const handleAction = useCallback(
    async (action: "approve" | "reject") => {
      if (!current) return;
      try {
        const res = await fetch("/api/photos/review", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: current.id, action }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        setStats((s) => ({
          ...s,
          [action === "approve" ? "approved" : "rejected"]:
            s[action === "approve" ? "approved" : "rejected"] + 1,
        }));
        setCurrentIndex((i) => i + 1);
        setImgErrors({});
      } catch (e) {
        setError(e instanceof Error ? e.message : "Action failed");
      }
    },
    [current]
  );

  // Scoped keyboard handler (only when container has focus)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
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
    [handleAction]
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
              className="w-10 h-10 rounded-full bg-red-500/10 border border-red-500/40 flex items-center justify-center hover:bg-red-500/20 transition-colors"
              title="Reject (Left Arrow)"
            >
              <XCircle className="h-5 w-5 text-red-500" />
            </button>
            <button
              onClick={() => handleAction("approve")}
              className="w-10 h-10 rounded-full bg-green-500/10 border border-green-500/40 flex items-center justify-center hover:bg-green-500/20 transition-colors"
              title="Approve (Right Arrow)"
            >
              <CheckCircle className="h-5 w-5 text-green-500" />
            </button>
          </div>

          {/* Candidate */}
          <div className="flex flex-col items-center gap-1 flex-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Candidate
            </p>
            <div className="relative w-full aspect-square rounded-lg border border-green-500/20 bg-muted overflow-hidden">
              {!imgErrors["candidate"] ? (
                <Image
                  src={current.candidate_url}
                  alt="Candidate"
                  fill
                  className="object-contain"
                  unoptimized
                  onError={() =>
                    setImgErrors((e) => ({ ...e, candidate: true }))
                  }
                />
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground text-[10px]">
                  No image
                </div>
              )}
            </div>
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
