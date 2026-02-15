"use client";

import Image from "next/image";
import { useState } from "react";
import type { TeamSquadPlayerSeasonStats } from "@/lib/types/squad";

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
  color: string;
  items: StatItem[];
}

function buildSections(p: TeamSquadPlayerSeasonStats): StatSection[] {
  const isGK = (p.position || "U").toUpperCase() === "G";

  return [
    {
      title: "Attack",
      color: "text-emerald-500",
      items: [
        { label: "Goals", value: p.goals },
        { label: "Assists", value: p.assists },
        { label: "Shots", value: p.shots_total },
        { label: "On Target", value: p.shots_on_target },
      ],
    },
    {
      title: "Passing",
      color: "text-blue-500",
      items: [
        { label: "Total", value: p.passes_total },
        { label: "Key Passes", value: p.key_passes },
        { label: "Accuracy", value: p.passes_accuracy != null ? `${p.passes_accuracy}%` : "—" },
      ],
    },
    {
      title: "Defense",
      color: "text-amber-500",
      items: [
        { label: "Tackles", value: p.tackles },
        { label: "Interceptions", value: p.interceptions },
        { label: "Blocks", value: p.blocks },
        ...(isGK ? [{ label: "Saves", value: p.saves }] : []),
      ],
    },
    {
      title: "Duels",
      color: "text-purple-500",
      items: [
        { label: "Total", value: p.duels_total },
        { label: "Won", value: p.duels_won },
        { label: "Dribbles", value: p.dribbles_attempts },
        { label: "Drb Success", value: p.dribbles_success },
      ],
    },
    {
      title: "Discipline",
      color: "text-red-400",
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
  teamName?: string;
}

export function PlayerDetail({ player, teamName }: PlayerDetailProps) {
  const [imgError, setImgError] = useState(false);
  const pos = (player.position || "U").toUpperCase();
  const isGK = pos === "G";
  const sections = buildSections(player);

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
    <div className="px-4 py-4 space-y-4">
      {/* Player header */}
      <div className="flex items-center gap-4">
        {!imgError ? (
          <Image
            src={player.photo_url || playerPhotoUrl(player.player_external_id)}
            alt={player.player_name}
            width={64}
            height={64}
            className="rounded-full object-cover shrink-0"
            unoptimized
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center shrink-0">
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
            {player.jersey_number != null && (
              <span className="text-xs tabular-nums bg-muted px-1.5 py-0.5 rounded">
                #{player.jersey_number}
              </span>
            )}
            <span>{POS_LABELS[pos] || pos}</span>
            {player.ever_captain && (
              <span className="text-xs font-medium bg-muted px-1.5 py-0.5 rounded">Captain</span>
            )}
          </div>
          {teamName && (
            <p className="text-xs text-muted-foreground mt-0.5">{teamName}</p>
          )}
        </div>
      </div>

      {/* Bio info */}
      {(player.nationality || player.birth_date || player.height) && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          {player.nationality && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Nationality</span>
              <span className="text-sm text-foreground">{player.nationality}</span>
            </div>
          )}
          {player.birth_date && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Born</span>
              <span className="text-sm text-foreground tabular-nums">
                {new Date(player.birth_date + "T00:00:00").toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })}
                {age != null && ` (${age})`}
              </span>
            </div>
          )}
          {birthLocation && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Birthplace</span>
              <span className="text-sm text-foreground truncate ml-2">{birthLocation}</span>
            </div>
          )}
          {(player.height || player.weight) && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Physical</span>
              <span className="text-sm text-foreground tabular-nums">
                {[player.height ? `${player.height} cm` : null, player.weight ? `${player.weight} kg` : null].filter(Boolean).join(" / ")}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Hero stats */}
      <div className="grid grid-cols-4 gap-2">
        <div className="rounded-lg border border-border px-3 py-2 text-center">
          <p className="text-lg font-bold text-foreground tabular-nums">
            {player.avg_rating != null ? player.avg_rating.toFixed(1) : "—"}
          </p>
          <p className="text-[10px] text-muted-foreground">Rating</p>
        </div>
        <div className="rounded-lg border border-border px-3 py-2 text-center">
          <p className="text-lg font-bold text-foreground tabular-nums">{player.appearances}</p>
          <p className="text-[10px] text-muted-foreground">Apps</p>
        </div>
        <div className="rounded-lg border border-border px-3 py-2 text-center">
          <p className="text-lg font-bold text-foreground tabular-nums">
            {player.total_minutes.toLocaleString()}
          </p>
          <p className="text-[10px] text-muted-foreground">Minutes</p>
        </div>
        <div className="rounded-lg border border-border px-3 py-2 text-center">
          <p className="text-lg font-bold text-foreground tabular-nums">
            {isGK ? player.saves : player.goals}
          </p>
          <p className="text-[10px] text-muted-foreground">{isGK ? "Saves" : "Goals"}</p>
        </div>
      </div>

      {/* Stat sections */}
      {sections.map((section) => (
        <div key={section.title}>
          <h4 className={`text-xs font-semibold uppercase tracking-wider mb-2 ${section.color}`}>
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
  );
}
