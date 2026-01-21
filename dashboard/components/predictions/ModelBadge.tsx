"use client";

import { ModelType, MODEL_TYPE_LABELS } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Cpu } from "lucide-react";

interface ModelBadgeProps {
  model: ModelType;
}

export function ModelBadge({ model }: ModelBadgeProps) {
  const isShadow = model === "Shadow";

  return (
    <Badge
      variant="outline"
      className={`gap-1 ${
        isShadow
          ? "bg-warning/10 text-warning border-warning/20"
          : "bg-surface text-foreground border-border"
      }`}
    >
      <Cpu className="h-3 w-3" />
      {MODEL_TYPE_LABELS[model]}
    </Badge>
  );
}
