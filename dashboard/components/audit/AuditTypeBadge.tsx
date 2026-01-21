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
    className: "bg-green-500/20 text-green-400 border-green-500/30",
  },
  prediction_generated: {
    icon: Brain,
    className: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  },
  prediction_frozen: {
    icon: Lock,
    className: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  },
  incident_ack: {
    icon: CheckCircle,
    className: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  },
  incident_resolve: {
    icon: XCircle,
    className: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  },
  config_changed: {
    icon: Settings,
    className: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  },
  data_quality_check: {
    icon: Shield,
    className: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
  },
  system: {
    icon: Server,
    className: "bg-gray-500/20 text-gray-400 border-gray-500/30",
  },
  user_action: {
    icon: User,
    className: "bg-pink-500/20 text-pink-400 border-pink-500/30",
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
