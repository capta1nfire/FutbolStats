"use client";

import { Badge } from "@/components/ui/badge";
import { AuditEventType, AUDIT_EVENT_TYPE_LABELS } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  Play,
  Brain,
  Lock,
  CheckCircle,
  XCircle,
  Settings,
  Shield,
  Server,
  User,
} from "lucide-react";

interface AuditTypeBadgeProps {
  type: AuditEventType;
  showIcon?: boolean;
  className?: string;
}

const typeConfig: Record<
  AuditEventType,
  { icon: typeof Play; className: string }
> = {
  job_run: {
    icon: Play,
    className: "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)] border-[var(--tag-blue-border)]",
  },
  prediction_generated: {
    icon: Brain,
    className: "bg-[var(--tag-purple-bg)] text-[var(--tag-purple-text)] border-[var(--tag-purple-border)]",
  },
  prediction_frozen: {
    icon: Lock,
    className: "bg-[var(--tag-cyan-bg)] text-[var(--tag-cyan-text)] border-[var(--tag-cyan-border)]",
  },
  incident_ack: {
    icon: CheckCircle,
    className: "bg-[var(--tag-blue-bg)] text-[var(--tag-blue-text)] border-[var(--tag-blue-border)]",
  },
  incident_resolve: {
    icon: XCircle,
    className: "bg-[var(--tag-cyan-bg)] text-[var(--tag-cyan-text)] border-[var(--tag-cyan-border)]",
  },
  config_changed: {
    icon: Settings,
    className: "bg-[var(--tag-orange-bg)] text-[var(--tag-orange-text)] border-[var(--tag-orange-border)]",
  },
  data_quality_check: {
    icon: Shield,
    className: "bg-[var(--tag-indigo-bg)] text-[var(--tag-indigo-text)] border-[var(--tag-indigo-border)]",
  },
  system: {
    icon: Server,
    className: "bg-[var(--tag-gray-bg)] text-[var(--tag-gray-text)] border-[var(--tag-gray-border)]",
  },
  user_action: {
    icon: User,
    className: "bg-[var(--tag-pink-bg)] text-[var(--tag-pink-text)] border-[var(--tag-pink-border)]",
  },
};

export function AuditTypeBadge({
  type,
  showIcon = true,
  className,
}: AuditTypeBadgeProps) {
  const config = typeConfig[type];
  const Icon = config.icon;
  const label = AUDIT_EVENT_TYPE_LABELS[type];

  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1 font-medium text-xs",
        config.className,
        className
      )}
    >
      {showIcon && <Icon className="h-3 w-3" />}
      {label}
    </Badge>
  );
}
