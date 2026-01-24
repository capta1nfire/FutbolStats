"use client";

import { SettingsSummary } from "@/lib/types";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { Badge } from "@/components/ui/badge";
import { Cpu, GitBranch, Clock } from "lucide-react";
import { formatDistanceToNow } from "@/lib/utils";

interface ModelVersionsSectionProps {
  settings: SettingsSummary;
}

export function ModelVersionsSection({ settings }: ModelVersionsSectionProps) {
  const { modelVersions } = settings;

  return (
    <div className="bg-background rounded-lg p-6 space-y-6 border border-border">
      <SettingsSectionHeader
        title="Model Versions"
        description="ML model versions currently deployed"
      />

      <div className="space-y-6">
        {/* Active Model (Model A) */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Active Model (Model A)</span>
            </div>
            <Badge variant="outline" className="bg-success/10 text-success border-success/20">
              Production
            </Badge>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <GitBranch className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-sm font-mono text-foreground">
                {modelVersions.modelA}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              XGBoost classifier with 14 features for 1X2 predictions
            </p>
          </div>
        </div>

        {/* Shadow Model */}
        <div className="bg-surface rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Shadow Model</span>
            </div>
            <Badge variant="outline" className="bg-warning/10 text-warning border-warning/20">
              Evaluation
            </Badge>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <GitBranch className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-sm font-mono text-foreground">
                {modelVersions.shadow}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              Two-stage model with form-based features (in shadow evaluation)
            </p>
          </div>
        </div>

        {/* Last Updated */}
        <div className="bg-surface rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Models last updated:</span>
            <span className="text-foreground">
              {formatDistanceToNow(modelVersions.updatedAt)}
            </span>
          </div>
        </div>

        {/* Phase 0 Notice */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <p className="text-sm text-muted-foreground">
            Model versions are managed through the deployment pipeline. The
            shadow model runs in parallel for evaluation but does not affect
            production predictions.
          </p>
        </div>
      </div>
    </div>
  );
}
