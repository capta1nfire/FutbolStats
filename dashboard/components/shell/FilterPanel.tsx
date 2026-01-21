"use client";

import { ReactNode } from "react";
import { ChevronLeft, ChevronRight, Search, Filter } from "lucide-react";
import { useHasMounted } from "@/lib/hooks";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export interface FilterOption {
  id: string;
  label: string;
  count?: number;
  checked?: boolean;
}

export interface FilterGroup {
  id: string;
  label: string;
  icon?: ReactNode;
  options: FilterOption[];
}

interface FilterPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  groups?: FilterGroup[];
  onFilterChange?: (groupId: string, optionId: string, checked: boolean) => void;
  onSearchChange?: (value: string) => void;
  searchValue?: string;
  /** Callback for "Customize Columns" link click (shows CustomizeColumnsPanel) */
  onCustomizeColumnsClick?: () => void;
  /** Whether to show the Customize Columns link in footer */
  showCustomizeColumns?: boolean;
}

export function FilterPanel({
  collapsed,
  onToggleCollapse,
  groups = [],
  onFilterChange,
  onSearchChange,
  searchValue = "",
  onCustomizeColumnsClick,
  showCustomizeColumns = false,
}: FilterPanelProps) {
  // Track client-side mount to avoid Radix Accordion hydration mismatch
  // Uses useSyncExternalStore instead of useState+useEffect to avoid lint error
  const mounted = useHasMounted();

  // Collapsed state: show rail with icon + tooltips
  if (collapsed) {
    return (
      <aside className="w-12 border-r border-border bg-surface flex flex-col items-center py-3 transition-smooth">
        <TooltipProvider delayDuration={0}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={onToggleCollapse}
                className="mb-2"
                aria-label="Expand filters"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>Expand filters</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="text-muted-foreground"
                aria-label="Filters"
              >
                <Filter className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>Filters</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </aside>
    );
  }

  // Expanded state: full filter panel
  return (
    <aside className="w-[220px] border-r border-border bg-surface flex flex-col transition-smooth">
      {/* Header */}
      <div className="h-12 flex items-center justify-between px-3 border-b border-border">
        <span className="text-sm font-medium text-foreground">Filters</span>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleCollapse}
          className="h-8 w-8"
          aria-label="Collapse filters"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      {/* Search */}
      <div className="p-3 border-b border-border">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search..."
            value={searchValue}
            onChange={(e) => onSearchChange?.(e.target.value)}
            className="pl-8 h-9 bg-background"
          />
        </div>
      </div>

      {/* Filter groups - render only on client to avoid Radix ID hydration mismatch */}
      <ScrollArea className="flex-1">
        {!mounted ? (
          // SSR/initial render: show skeleton placeholders
          <div className="px-3 space-y-4 py-3">
            {groups.map((group) => (
              <div key={group.id} className="space-y-2">
                <Skeleton className="h-5 w-24" />
                <div className="space-y-2 pl-1">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-5/6" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          // Client: render actual Accordion
          <Accordion type="multiple" defaultValue={groups.map((g) => g.id)} className="px-3">
            {groups.map((group) => (
              <AccordionItem key={group.id} value={group.id} className="border-b-0">
                <AccordionTrigger className="py-3 text-sm hover:no-underline">
                  <span className="flex items-center gap-2">
                    {group.icon}
                    {group.label}
                  </span>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-2 pb-2">
                    {group.options.map((option) => (
                      <label
                        key={option.id}
                        className="flex items-center gap-2 cursor-pointer group"
                      >
                        <Checkbox
                          id={`${group.id}-${option.id}`}
                          checked={option.checked}
                          onCheckedChange={(checked) =>
                            onFilterChange?.(group.id, option.id, checked === true)
                          }
                        />
                        <span className="text-sm text-muted-foreground group-hover:text-foreground flex-1">
                          {option.label}
                        </span>
                        {option.count !== undefined && (
                          <span className="text-xs text-muted-foreground">
                            ({option.count})
                          </span>
                        )}
                      </label>
                    ))}
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </ScrollArea>

      {/* Footer with Customize Columns link (UniFi style) */}
      {showCustomizeColumns && (
        <div className="px-3 py-2 border-t border-border">
          <button
            onClick={onCustomizeColumnsClick}
            className="text-sm font-medium text-primary hover:text-blue-400 transition-colors"
          >
            Customize Columns
          </button>
        </div>
      )}
    </aside>
  );
}
