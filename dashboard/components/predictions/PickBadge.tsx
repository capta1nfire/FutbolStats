"use client";

import { PickOutcome, PICK_OUTCOME_LABELS } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

interface PickBadgeProps {
  pick: PickOutcome;
  isCorrect?: boolean; // if evaluated, whether pick matched result
}

export function PickBadge({ pick, isCorrect }: PickBadgeProps) {
  let className = "bg-surface text-foreground border-border";

  if (isCorrect === true) {
    className = "bg-success/10 text-success border-success/20";
  } else if (isCorrect === false) {
    className = "bg-error/10 text-error border-error/20";
  }

  return (
    <Badge variant="outline" className={className}>
      {PICK_OUTCOME_LABELS[pick]}
    </Badge>
  );
}
