"use client";

import Image from "next/image";
import { useCallback, useEffect, useState } from "react";

interface Candidate {
  id: number;
  player_external_id: number;
  player_name: string;
  team_name: string;
  team_external_id: number;
  source: string;
  quality_score: number;
  candidate_url: string;
  current_url: string;
  width: number;
  height: number;
  identity_score: number;
  review_status: string;
}

type ReviewStats = { approved: number; rejected: number; total: number };

export default function PhotoReviewPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<ReviewStats>({ approved: 0, rejected: 0, total: 0 });
  const [imgErrors, setImgErrors] = useState<Record<string, boolean>>({});

  // Fetch candidates
  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/photos/review?status=pending_review");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const items = data.candidates || [];
        setCandidates(items);
        setStats((s) => ({ ...s, total: items.length }));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load candidates");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

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

  // Keyboard shortcuts
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "ArrowRight" || e.key === "Enter") {
        e.preventDefault();
        handleAction("approve");
      } else if (e.key === "ArrowLeft" || e.key === "Backspace") {
        e.preventDefault();
        handleAction("reject");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleAction]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <p className="text-muted-foreground">Loading candidates...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <p className="text-red-500">{error}</p>
      </div>
    );
  }

  if (!current) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-background gap-4">
        <h2 className="text-2xl font-bold text-foreground">Review Complete</h2>
        <div className="flex gap-8 text-lg">
          <span className="text-green-500">Approved: {stats.approved}</span>
          <span className="text-red-500">Rejected: {stats.rejected}</span>
          <span className="text-muted-foreground">Total: {stats.total}</span>
        </div>
        <p className="text-sm text-muted-foreground mt-4">
          Run <code className="bg-muted px-2 py-1 rounded">--approved-only</code> to process approved candidates through PhotoRoom + R2.
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header bar */}
      <div className="border-b border-border px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-semibold text-foreground">Photo Review</h1>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-green-500 tabular-nums">{stats.approved} approved</span>
          <span className="text-red-500 tabular-nums">{stats.rejected} rejected</span>
          <span className="text-muted-foreground tabular-nums">{remaining} remaining</span>
        </div>
      </div>

      {/* Player info */}
      <div className="text-center py-4">
        <h2 className="text-xl font-bold text-foreground">{current.player_name}</h2>
        <p className="text-sm text-muted-foreground">
          {current.team_name} | {current.source} | Quality: {current.quality_score} | Identity: {current.identity_score}
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          {current.width}x{current.height} | ID: {current.player_external_id}
        </p>
      </div>

      {/* Comparison area */}
      <div className="flex-1 flex items-center justify-center gap-8 px-8 pb-8">
        {/* Current (API Football) */}
        <div className="flex flex-col items-center gap-3">
          <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Current</p>
          <div className="relative w-64 h-64 rounded-xl border-2 border-red-500/30 bg-muted overflow-hidden">
            {!imgErrors["current"] ? (
              <Image
                src={current.current_url}
                alt="Current"
                fill
                className="object-contain"
                unoptimized
                onError={() => setImgErrors((e) => ({ ...e, current: true }))}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                No image
              </div>
            )}
          </div>
          <p className="text-xs text-muted-foreground">API Football (150x150)</p>
        </div>

        {/* Action buttons */}
        <div className="flex flex-col items-center gap-4">
          <button
            onClick={() => handleAction("reject")}
            className="w-16 h-16 rounded-full bg-red-500/10 border-2 border-red-500/50 flex items-center justify-center hover:bg-red-500/20 transition-colors"
            title="Reject (Left Arrow)"
          >
            <span className="text-red-500 text-2xl font-bold">✕</span>
          </button>
          <div className="text-xs text-muted-foreground text-center">
            <p>← Reject</p>
            <p className="mt-2">Approve →</p>
          </div>
          <button
            onClick={() => handleAction("approve")}
            className="w-16 h-16 rounded-full bg-green-500/10 border-2 border-green-500/50 flex items-center justify-center hover:bg-green-500/20 transition-colors"
            title="Approve (Right Arrow)"
          >
            <span className="text-green-500 text-2xl font-bold">✓</span>
          </button>
        </div>

        {/* Candidate (Club Site / New) */}
        <div className="flex flex-col items-center gap-3">
          <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Candidate</p>
          <div className="relative w-64 h-64 rounded-xl border-2 border-green-500/30 bg-muted overflow-hidden">
            {!imgErrors["candidate"] ? (
              <Image
                src={current.candidate_url}
                alt="Candidate"
                fill
                className="object-contain"
                unoptimized
                onError={() => setImgErrors((e) => ({ ...e, candidate: true }))}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                No image
              </div>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {current.source} ({current.width}x{current.height})
          </p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="px-6 pb-4">
        <div className="w-full bg-muted rounded-full h-2">
          <div
            className="bg-green-500 h-2 rounded-full transition-all"
            style={{ width: `${((currentIndex) / candidates.length) * 100}%` }}
          />
        </div>
        <p className="text-xs text-muted-foreground mt-1 text-center">
          {currentIndex} / {candidates.length}
        </p>
      </div>
    </div>
  );
}
