"use client";

import { Badge } from "@/components/ui/badge";

interface SettingsSectionHeaderProps {
  title: string;
  description?: string;
  readOnly?: boolean;
}

export function SettingsSectionHeader({
  title,
  description,
  readOnly = true,
}: SettingsSectionHeaderProps) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-foreground">{title}</h2>
        {readOnly && (
          <Badge variant="outline" className="text-xs text-muted-foreground">
            Read-only
          </Badge>
        )}
      </div>
      {description && (
        <p className="text-sm text-muted-foreground mt-1">{description}</p>
      )}
    </div>
  );
}
