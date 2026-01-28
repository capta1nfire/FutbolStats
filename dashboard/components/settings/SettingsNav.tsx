"use client";

import { cn } from "@/lib/utils";
import {
  SettingsSection,
  SETTINGS_SECTIONS,
  SETTINGS_SECTION_LABELS,
} from "@/lib/types";
import {
  Settings,
  Clock,
  Bell,
  Key,
  Cpu,
  ToggleLeft,
  Users,
  Brain,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface SettingsNavProps {
  activeSection: SettingsSection;
  onSectionChange: (section: SettingsSection) => void;
}

const sectionIcons: Record<SettingsSection, React.ReactNode> = {
  general: <Settings className="h-4 w-4" strokeWidth={1.5} />,
  timezone: <Clock className="h-4 w-4" strokeWidth={1.5} />,
  notifications: <Bell className="h-4 w-4" strokeWidth={1.5} />,
  api_keys: <Key className="h-4 w-4" strokeWidth={1.5} />,
  model_versions: <Cpu className="h-4 w-4" strokeWidth={1.5} />,
  feature_flags: <ToggleLeft className="h-4 w-4" strokeWidth={1.5} />,
  users: <Users className="h-4 w-4" strokeWidth={1.5} />,
  ia_features: <Brain className="h-4 w-4" strokeWidth={1.5} />,
};

export function SettingsNav({
  activeSection,
  onSectionChange,
}: SettingsNavProps) {
  return (
    <nav className="w-[200px] bg-background border-r border-border flex flex-col">
      <div className="h-12 flex items-center px-4">
        <span className="text-sm font-medium text-foreground">Settings</span>
      </div>
      <div className="flex-1 py-2">
        {SETTINGS_SECTIONS.map((section) => (
          <Button
            key={section}
            variant="ghost"
            className={cn(
              "w-full justify-start gap-2 px-4 py-2 h-9 text-sm font-normal",
              activeSection === section
                ? "text-primary"
                : "text-muted-foreground hover:text-foreground"
            )}
            onClick={() => onSectionChange(section)}
          >
            {sectionIcons[section]}
            {SETTINGS_SECTION_LABELS[section]}
          </Button>
        ))}
      </div>
    </nav>
  );
}
