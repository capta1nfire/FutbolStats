"use client";

import { JobStatus } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Loader2, CheckCircle2, XCircle, Clock } from "lucide-react";

interface JobStatusBadgeProps {
  status: JobStatus;
  className?: string;
}

const statusConfig: Record<
  JobStatus,
  { label: string; variant: "default" | "secondary" | "destructive" | "outline"; icon: React.ReactNode }
> = {
  running: {
    label: "Running",
    variant: "default",
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
  },
  success: {
    label: "Success",
    variant: "secondary",
    icon: <CheckCircle2 className="h-3 w-3 text-success" />,
  },
  failed: {
    label: "Failed",
    variant: "destructive",
    icon: <XCircle className="h-3 w-3" />,
  },
  pending: {
    label: "Pending",
    variant: "outline",
    icon: <Clock className="h-3 w-3" />,
  },
};

export function JobStatusBadge({ status, className }: JobStatusBadgeProps) {
  const config = statusConfig[status];

  return (
    <Badge
      variant={config.variant}
      className={cn("gap-1 font-normal", className)}
    >
      {config.icon}
      {config.label}
    </Badge>
  );
}
