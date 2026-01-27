"use client";

import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export type AdminSection = "overview" | "leagues" | "groups" | "audit";

const SECTIONS: { id: AdminSection; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "leagues", label: "Leagues" },
  { id: "groups", label: "Groups" },
  { id: "audit", label: "Audit" },
];

interface AdminNavProps {
  section: AdminSection;
  onSectionChange: (s: AdminSection) => void;
  // League filters (only visible when section === "leagues")
  leagueSearch?: string;
  onLeagueSearchChange?: (v: string) => void;
  leagueKind?: string;
  onLeagueKindChange?: (v: string) => void;
  leagueActive?: string;
  onLeagueActiveChange?: (v: string) => void;
  kinds?: string[];
}

export function AdminNav({
  section,
  onSectionChange,
  leagueSearch = "",
  onLeagueSearchChange,
  leagueKind = "",
  onLeagueKindChange,
  leagueActive = "",
  onLeagueActiveChange,
  kinds = [],
}: AdminNavProps) {
  return (
    <nav
      aria-label="Admin navigation"
      className="w-[220px] shrink-0 border-r border-border bg-sidebar flex flex-col overflow-y-auto"
    >
      {/* Section tabs */}
      <div className="p-3 space-y-1">
        <p className="text-xs font-medium text-muted-foreground px-2 mb-2 uppercase tracking-wider">
          Admin
        </p>
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => onSectionChange(s.id)}
            className={cn(
              "w-full text-left px-3 py-2 rounded-md text-sm transition-colors",
              section === s.id
                ? "bg-accent text-accent-foreground font-medium"
                : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
            )}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* League filters — only when section === "leagues" */}
      {section === "leagues" && (
        <div className="border-t border-border p-3 space-y-3">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Filters
          </p>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search leagues…"
              value={leagueSearch}
              onChange={(e) => onLeagueSearchChange?.(e.target.value)}
              className="pl-8 h-8 text-sm"
            />
          </div>

          {/* Kind */}
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Kind</Label>
            <Select value={leagueKind || "__all__"} onValueChange={(v) => onLeagueKindChange?.(v === "__all__" ? "" : v)}>
              <SelectTrigger className="h-8 text-sm">
                <SelectValue placeholder="All kinds" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All kinds</SelectItem>
                {kinds.map((k) => (
                  <SelectItem key={k} value={k}>
                    {k}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Active */}
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Status</Label>
            <Select value={leagueActive || "__all__"} onValueChange={(v) => onLeagueActiveChange?.(v === "__all__" ? "" : v)}>
              <SelectTrigger className="h-8 text-sm">
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All</SelectItem>
                <SelectItem value="true">Active</SelectItem>
                <SelectItem value="false">Inactive</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      )}
    </nav>
  );
}
