"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";
import type { TeamSquadPlayerSeasonStats } from "@/lib/types/squad";
import { IconTabs } from "@/components/ui/icon-tabs";
import { SurfaceCard } from "@/components/ui/surface-card";
import { Info, BarChart3, ImageIcon, Loader2, CheckCircle, XCircle, Upload, FlipHorizontal2 } from "lucide-react";
import { JerseyIcon } from "@/components/ui/jersey-icon";
import { TeamPhotoReview } from "./TeamPhotoReview";

function playerPhotoUrl(externalId: number): string {
  return `https://media.api-sports.io/football/players/${externalId}.png`;
}

const POS_LABELS: Record<string, string> = {
  G: "Goalkeeper",
  D: "Defender",
  M: "Midfielder",
  F: "Forward",
  U: "Unknown",
};

interface StatItem {
  label: string;
  value: string | number;
}

interface StatSection {
  title: string;
  items: StatItem[];
}

function buildSections(p: TeamSquadPlayerSeasonStats): StatSection[] {
  const isGK = (p.position || "U").toUpperCase() === "G";

  return [
    {
      title: "Attack",
      items: [
        { label: "Goals", value: p.goals },
        { label: "Assists", value: p.assists },
        { label: "Shots", value: p.shots_total },
        { label: "On Target", value: p.shots_on_target },
      ],
    },
    {
      title: "Passing",
      items: [
        { label: "Total", value: p.passes_total },
        { label: "Key Passes", value: p.key_passes },
        { label: "Accuracy", value: p.passes_accuracy != null ? `${p.passes_accuracy}%` : "—" },
      ],
    },
    {
      title: "Defense",
      items: [
        { label: "Tackles", value: p.tackles },
        { label: "Interceptions", value: p.interceptions },
        { label: "Blocks", value: p.blocks },
        ...(isGK ? [{ label: "Saves", value: p.saves }] : []),
      ],
    },
    {
      title: "Duels",
      items: [
        { label: "Total", value: p.duels_total },
        { label: "Won", value: p.duels_won },
        { label: "Dribbles", value: p.dribbles_attempts },
        { label: "Drb Success", value: p.dribbles_success },
      ],
    },
    {
      title: "Discipline",
      items: [
        { label: "Yellows", value: p.yellows },
        { label: "Reds", value: p.reds },
        { label: "Fouls Drawn", value: p.fouls_drawn },
        { label: "Fouls Committed", value: p.fouls_committed },
      ],
    },
  ];
}

function computeAge(birthDateStr: string): number {
  const birth = new Date(birthDateStr + "T00:00:00");
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age--;
  }
  return age;
}

interface PlayerDetailProps {
  player: TeamSquadPlayerSeasonStats;
  teamMatchesPlayed?: number;
  teamName?: string;
  teamLogoUrl?: string;
  teamId?: number | null;
}

const PLAYER_TABS = [
  { id: "overview", icon: <Info />, label: "Overview" },
  { id: "stats", icon: <BarChart3 />, label: "Stats" },
  { id: "images", icon: <ImageIcon />, label: "Images" },
];

export function PlayerDetail({ player, teamMatchesPlayed = 0, teamName, teamLogoUrl, teamId }: PlayerDetailProps) {
  const [imgError, setImgError] = useState(false);
  const [photoModal, setPhotoModal] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const pos = (player.position || "U").toUpperCase();
  const isGK = pos === "G";

  // ── Images tab state ──
  const [imgUrl, setImgUrl] = useState("");
  const [imgSubmitting, setImgSubmitting] = useState(false);
  const [imgMsg, setImgMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [pastePreview, setPastePreview] = useState<string | null>(null);
  const [currentPhotoDims, setCurrentPhotoDims] = useState<{ w: number; h: number } | null>(null);
  const [flipping, setFlipping] = useState(false);
  const [photoVersion, setPhotoVersion] = useState(0);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  // Reset images state when player changes
  useEffect(() => {
    setImgUrl("");
    setImgMsg(null);
    setReviewOpen(false);
    setPastePreview(null);
  }, [player.player_external_id]);

  const submitCandidate = useCallback(async (payload: { image_url?: string; image_base64?: string }) => {
    setImgSubmitting(true);
    setImgMsg(null);
    try {
      const res = await fetch("/api/photos/add-candidate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          player_external_id: player.player_external_id,
          ...payload,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setImgMsg({ type: "err", text: data.error || `HTTP ${res.status}` });
        return;
      }
      setImgUrl("");
      setPastePreview(null);
      setReviewOpen(true);
    } catch (e) {
      setImgMsg({ type: "err", text: e instanceof Error ? e.message : "Failed" });
    } finally {
      setImgSubmitting(false);
    }
  }, [player.player_external_id]);

  const handleAddUrl = useCallback(() => {
    const url = imgUrl.trim();
    if (!url || !url.startsWith("http")) {
      setImgMsg({ type: "err", text: "Paste a valid HTTP URL" });
      return;
    }
    submitCandidate({ image_url: url });
  }, [imgUrl, submitCandidate]);

  const handleImageFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setImgMsg({ type: "err", text: "Not an image file" });
      return;
    }
    if (file.size > 10_000_000) {
      setImgMsg({ type: "err", text: "Image too large (max 10MB)" });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setPastePreview(dataUrl);
      // Extract base64 part (after "data:image/...;base64,")
      const b64 = dataUrl.split(",")[1];
      if (b64) submitCandidate({ image_base64: b64 });
    };
    reader.readAsDataURL(file);
  }, [submitCandidate]);

  // Paste handler for images
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith("image/")) {
        e.preventDefault();
        const file = items[i].getAsFile();
        if (file) handleImageFile(file);
        return;
      }
    }
    // If no image in clipboard, let the text paste through to the URL input
  }, [handleImageFile]);

  // Drop handler
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleImageFile(file);
      return;
    }
    // Check for dropped URL
    const url = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
    if (url && url.startsWith("http")) {
      submitCandidate({ image_url: url.trim() });
    }
  }, [handleImageFile, submitCandidate]);
  const sections = buildSections(player);

  const handleFlip = useCallback(async () => {
    setFlipping(true);
    try {
      const res = await fetch("/api/photos/flip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_external_id: player.player_external_id }),
      });
      if (!res.ok) {
        const data = await res.json();
        setImgMsg({ type: "err", text: data.error || `Flip failed (${res.status})` });
        return;
      }
      setPhotoVersion((v) => v + 1);
      setImgMsg({ type: "ok", text: "Photo flipped" });
    } catch (e) {
      setImgMsg({ type: "err", text: e instanceof Error ? e.message : "Flip failed" });
    } finally {
      setFlipping(false);
    }
  }, [player.player_external_id]);

  // Full name from firstname + lastname if available
  const fullName =
    player.firstname && player.lastname
      ? `${player.firstname} ${player.lastname}`
      : player.player_name;

  // Compute age from birth_date
  const age = player.birth_date ? computeAge(player.birth_date) : null;

  // Birthplace: city + country
  const birthLocation = [player.birth_place, player.birth_country]
    .filter(Boolean)
    .join(", ");

  return (
    <>
    <div data-dev-ref="PlayerDetail" className="px-4 py-4 space-y-4">
      {/* Player header — always visible */}
      <div className="flex items-center gap-4">
        {!imgError ? (
          <div
            className="w-24 h-24 rounded-full shrink-0 bg-white/10 overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary/50 transition-all"
            onClick={() => setPhotoModal(true)}
          >
            <Image
              src={player.photo_url_card_hq || player.photo_url || playerPhotoUrl(player.player_external_id)}
              alt={player.player_name}
              width={96}
              height={96}
              className="h-24 w-24 rounded-full object-cover"
              unoptimized={!(player.photo_url_card_hq)}
              onError={() => setImgError(true)}
            />
          </div>
        ) : (
          <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center shrink-0">
            <span className="text-lg font-semibold text-muted-foreground">
              {player.player_name.charAt(0)}
            </span>
          </div>
        )}
        <div className="min-w-0">
          <h3 className="text-lg font-semibold text-foreground truncate">
            {fullName}
          </h3>
          <div className="flex items-center gap-2 text-sm text-muted-foreground mt-0.5 flex-wrap">
            <span>{POS_LABELS[pos] || pos}</span>
            {player.jersey_number != null && (
              <JerseyIcon
                number={player.jersey_number}
                numberColor="rgba(255,255,255,0.5)"
                size={20}
              />
            )}
            {player.ever_captain && (
              <span className="text-xs font-medium bg-muted px-1.5 py-0.5 rounded">Captain</span>
            )}
          </div>
          {teamName && (
            <p className="text-xs text-muted-foreground mt-0.5">{teamName}</p>
          )}
          <p className="text-xs text-muted-foreground/50 mt-0.5">
            ID {player.player_external_id}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <IconTabs
        tabs={PLAYER_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />

      {/* Tab content */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          {/* Bio info */}
          {(player.nationality || player.birth_date || player.height) && (
            <SurfaceCard className="space-y-2.5">
              {player.nationality && (
                <div className="flex items-baseline justify-between">
                  <span className="text-sm text-muted-foreground">Nationality</span>
                  <span className="text-sm text-foreground">{player.nationality}</span>
                </div>
              )}
              {player.birth_date && (
                <div className="flex items-baseline justify-between">
                  <span className="text-sm text-muted-foreground">Born</span>
                  <span className="text-sm text-foreground tabular-nums">
                    {new Date(player.birth_date + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}
                    {age != null && ` (${age})`}
                  </span>
                </div>
              )}
              {birthLocation && (
                <div className="flex items-baseline justify-between">
                  <span className="text-sm text-muted-foreground shrink-0">Birthplace</span>
                  <span className="text-sm text-foreground text-right ml-4">{birthLocation}</span>
                </div>
              )}
              {player.height && (
                <div className="flex items-baseline justify-between">
                  <span className="text-sm text-muted-foreground">Height</span>
                  <span className="text-sm text-foreground tabular-nums">{player.height} cm</span>
                </div>
              )}
              {player.weight && (
                <div className="flex items-baseline justify-between">
                  <span className="text-sm text-muted-foreground">Weight</span>
                  <span className="text-sm text-foreground tabular-nums">{player.weight} kg</span>
                </div>
              )}
            </SurfaceCard>
          )}

          {/* Hero stats */}
          <div className={`grid gap-2 ${isGK ? "grid-cols-5" : "grid-cols-4"}`}>
            <div className="rounded-lg border border-border px-2 py-2 text-center">
              <p className="text-lg font-bold text-foreground tabular-nums">
                {player.avg_rating != null ? player.avg_rating.toFixed(1) : "—"}
              </p>
              <p className="text-[10px] text-muted-foreground">Rating</p>
            </div>
            <div className="rounded-lg border border-border px-2 py-2 text-center">
              <p className="text-lg font-bold text-foreground tabular-nums">
                {player.appearances}
                {teamMatchesPlayed > 0 && (
                  <span className="text-muted-foreground/40">/{teamMatchesPlayed}</span>
                )}
              </p>
              <p className="text-[10px] text-muted-foreground">Apps</p>
            </div>
            <div className="rounded-lg border border-border px-2 py-2 text-center">
              <p className="text-lg font-bold text-foreground tabular-nums">
                {player.total_minutes.toLocaleString()}
              </p>
              <p className="text-[10px] text-muted-foreground">Minutes</p>
            </div>
            {isGK ? (
              <>
                <div className="rounded-lg border border-border px-2 py-2 text-center">
                  <p className="text-lg font-bold text-foreground tabular-nums">{player.saves}</p>
                  <p className="text-[10px] text-muted-foreground">Saves</p>
                </div>
                <div className="rounded-lg border border-border px-2 py-2 text-center">
                  <p className="text-lg font-bold text-foreground tabular-nums">{player.goals_conceded}</p>
                  <p className="text-[10px] text-muted-foreground">Conc.</p>
                </div>
              </>
            ) : (
              <div className="rounded-lg border border-border px-2 py-2 text-center">
                <p className="text-lg font-bold text-foreground tabular-nums">{player.goals}</p>
                <p className="text-[10px] text-muted-foreground">Goals</p>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "stats" && (
        <div className="space-y-4">
          {sections.map((section) => (
            <div key={section.title}>
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-2 text-muted-foreground">
                {section.title}
              </h4>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                {section.items.map((item) => (
                  <div key={item.label} className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">{item.label}</span>
                    <span className="text-sm font-medium text-foreground tabular-nums">
                      {item.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === "images" && (
        <div className="space-y-4" onPaste={handlePaste}>
          {/* Current photo */}
          <SurfaceCard className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Current Photo
              </h4>
              {player.photo_url_card_hq && (
                <button
                  onClick={handleFlip}
                  disabled={flipping}
                  className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
                  title="Flip photo horizontally"
                >
                  {flipping ? <Loader2 className="h-3 w-3 animate-spin" /> : <FlipHorizontal2 className="h-3 w-3" />}
                  Flip
                </button>
              )}
            </div>
            <div className="flex justify-center">
              <div className="relative w-32 h-32 rounded-lg bg-muted overflow-hidden">
                <Image
                  src={`${player.photo_url_card_hq || player.photo_url || playerPhotoUrl(player.player_external_id)}${photoVersion ? `?v=${photoVersion}` : ""}`}
                  alt={player.player_name}
                  fill
                  className="object-cover"
                  unoptimized
                  onLoad={(e) => {
                    const img = e.currentTarget as HTMLImageElement;
                    if (img.naturalWidth && img.naturalHeight) setCurrentPhotoDims({ w: img.naturalWidth, h: img.naturalHeight });
                  }}
                />
              </div>
            </div>
            <p className="text-[10px] text-muted-foreground text-center tabular-nums">
              ext_id: {player.player_external_id}
              {currentPhotoDims && ` · ${currentPhotoDims.w}×${currentPhotoDims.h}`}
              {player.photo_url_card_hq && " · HQ card"}
            </p>
          </SurfaceCard>

          {/* Drop zone + URL input */}
          <SurfaceCard className="space-y-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Add Image
            </h4>
            {/* Drop zone */}
            <div
              ref={dropZoneRef}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors cursor-pointer ${
                dragOver ? "border-primary bg-primary/5" : "border-border hover:border-muted-foreground/40"
              }`}
              onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = "image/*";
                input.onchange = () => { if (input.files?.[0]) handleImageFile(input.files[0]); };
                input.click();
              }}
            >
              {pastePreview ? (
                <div className="flex justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={pastePreview} alt="Preview" className="max-h-32 rounded" />
                </div>
              ) : (
                <>
                  <Upload className="h-5 w-5 mx-auto text-muted-foreground/50 mb-1.5" />
                  <p className="text-xs text-muted-foreground">
                    Drop image, paste (Ctrl+V), or click to browse
                  </p>
                </>
              )}
            </div>
            {/* URL input */}
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Or paste image URL..."
                value={imgUrl}
                onChange={(e) => { setImgUrl(e.target.value); setImgMsg(null); }}
                onKeyDown={(e) => { if (e.key === "Enter") handleAddUrl(); }}
                className="flex-1 min-w-0 bg-muted border border-border rounded px-2.5 py-1.5 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50"
              />
              <button
                onClick={handleAddUrl}
                disabled={imgSubmitting || !imgUrl.trim()}
                className="px-3 py-1.5 rounded bg-primary text-primary-foreground text-sm hover:bg-primary/90 disabled:opacity-40 flex items-center gap-1.5 shrink-0"
              >
                {imgSubmitting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Add"}
              </button>
            </div>
            {imgMsg && (
              <p className={`text-xs ${imgMsg.type === "ok" ? "text-green-500" : "text-red-500"}`}>
                {imgMsg.text}
              </p>
            )}
          </SurfaceCard>

        </div>
      )}

      {/* Team photo review — renders inline summary, opens its own fullscreen */}
      {reviewOpen && teamId && teamName && (
        <div className="px-4 pb-4">
          <TeamPhotoReview teamId={teamId} teamName={teamName} initialFullscreen playerExternalId={player.player_external_id} />
        </div>
      )}
    </div>

      {/* Photo modal */}
      {photoModal && !imgError && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/20"
          onClick={() => setPhotoModal(false)}
        >
          <div className="relative max-w-sm w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="absolute bottom-0 left-0 right-0 h-1/2 bg-black rounded-2xl shadow-tooltip" />
            <div className="relative rounded-2xl overflow-hidden bg-gradient-to-b from-neutral-800 to-neutral-950 shadow-tooltip mb-10">
              <Image
                src={player.photo_url_card_hq || player.photo_url || playerPhotoUrl(player.player_external_id)}
                alt={player.player_name}
                width={512}
                height={512}
                className="w-full h-auto object-contain"
                unoptimized={!(player.photo_url_card_hq)}
              />
            </div>
            <div className="absolute bottom-0 left-0 right-0 px-4 py-2.5 flex items-center justify-center gap-2">
              {teamLogoUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={teamLogoUrl} alt="" className="w-5 h-5 shrink-0 object-contain" />
              )}
              <p className="text-white/80 text-sm font-medium">{fullName}</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
