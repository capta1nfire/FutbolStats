"use client";

import { PredictionStatus, PREDICTION_STATUS_LABELS } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, XCircle, Snowflake, Clock } from "lucide-react";

interface PredictionStatusBadgeProps {
  status: PredictionStatus;
}

const statusConfig: Record<
  PredictionStatus,
  { icon: React.ReactNode; className: string }
> = {
  generated: {
    icon: <CheckCircle className="h-3 w-3" />,
    className: "bg-success/10 text-success border-success/20",
  },
  missing: {
    icon: <XCircle className="h-3 w-3" />,
    className: "bg-error/10 text-error border-error/20",
  },
  frozen: {
    icon: <Snowflake className="h-3 w-3" />,
    className: "bg-info/10 text-info border-info/20",
  },
  evaluated: {
    icon: <Clock className="h-3 w-3" />,
    className: "bg-accent/10 text-accent border-accent/20",
  },
};

export function PredictionStatusBadge({ status }: PredictionStatusBadgeProps) {
  const config = statusConfig[status];

  return (
    <Badge variant="outline" className={`gap-1 ${config.className}`}>
      {config.icon}
      {PREDICTION_STATUS_LABELS[status]}
    </Badge>
  );
}
