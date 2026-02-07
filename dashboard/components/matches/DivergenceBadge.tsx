"use client";

import type { Gap20Result } from "@/lib/predictions/gap20";
import type { MatchScore } from "@/lib/types";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface DivergenceBadgeProps {
  result: Gap20Result;
  /** Match score — shown as badge text when available (e.g. "2-1") */
  score?: MatchScore | null;
  className?: string;
}

const FAV_LABELS = ["1", "X", "2"] as const;

/**
 * Badge for GAP20 model-vs-market divergence.
 * AGREE = no badge (normal case, no visual noise).
 * DISAGREE = amber chip with score (or "D" if no score).
 * STRONG_FAV_DISAGREE = destructive chip with score (or "SFAV" if no score).
 */
export function DivergenceBadge({ result, score, className }: DivergenceBadgeProps) {
  if (result.category === "AGREE") return null;

  const isSFAV = result.category === "STRONG_FAV_DISAGREE";
  const gapPct = (Math.abs(result.gapOnModelFav) * 100).toFixed(0);
  const mktPct = (result.marketFavProb * 100).toFixed(0);

  const tooltipText = isSFAV
    ? `Model ${FAV_LABELS[result.modelFav]} vs Market ${FAV_LABELS[result.marketFav]} (gap ${gapPct}pp, mkt fav ${mktPct}%)`
    : `Model ${FAV_LABELS[result.modelFav]} vs Market ${FAV_LABELS[result.marketFav]}`;

  // Show score when available, fallback to D/SFAV labels
  const badgeText = score != null
    ? `${score.home}-${score.away}`
    : isSFAV ? "SFAV" : "D";

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              "inline-flex items-center justify-center text-[10px] font-bold rounded shrink-0",
              isSFAV
                ? "px-1.5 py-0.5 bg-destructive/15 text-destructive border border-destructive/25"
                : "px-1 py-0.5 bg-warning/10 text-warning border border-warning/20",
              className,
            )}
          >
            {badgeText}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" sideOffset={8}>
          <p className="text-xs">{tooltipText}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
