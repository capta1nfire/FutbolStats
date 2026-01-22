"use client";

import { ReactNode } from "react";
import { ChevronLeft, ChevronRight, Filter } from "lucide-react";
import { useHasMounted } from "@/lib/hooks";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { SearchInput } from "@/components/ui/search-input";
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
  /** Whether CustomizeColumnsPanel is currently open (hides collapse button) */
  customizeColumnsOpen?: boolean;
  /** Optional content to render below search (e.g., quick filters) */
  quickFilterContent?: ReactNode;
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
  customizeColumnsOpen = false,
  quickFilterContent,
}: FilterPanelProps) {
  // Track client-side mount to avoid Radix Accordion hydration mismatch
  // Uses useSyncExternalStore instead of useState+useEffect to avoid lint error
  const mounted = useHasMounted();

  // Collapsed state: show rail with icon + tooltips
  if (collapsed) {
    return (
      <aside className="w-12 shrink-0 border-r border-border bg-sidebar flex flex-col items-center py-3 transition-smooth">
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

  // Expanded state: full filter panel (277px matches UniFi Left Rail width)
  return (
    <aside className="w-[277px] shrink-0 border-r border-border bg-sidebar flex flex-col transition-smooth">
      {/* Header - collapse button hidden when CustomizeColumnsPanel is open */}
      <div className="h-12 flex items-center justify-between px-3">
        <span className="text-sm font-medium text-foreground">Filters</span>
        {!customizeColumnsOpen && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleCollapse}
            className="h-8 w-8"
            aria-label="Collapse filters"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Search */}
      <div className="p-3">
        <SearchInput
          placeholder="Search..."
          value={searchValue}
          onChange={(value) => onSearchChange?.(value)}
          className="bg-background"
        />
      </div>

      {/* Quick filter slot (e.g., severity bars) */}
      {quickFilterContent && (
        <div className="px-3 pb-3">
          {quickFilterContent}
        </div>
      )}

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
        <div className="px-3 py-3 flex items-center">
          <button
            onClick={onCustomizeColumnsClick}
            className="text-sm font-medium text-primary hover:text-primary-hover transition-colors"
          >
            Customize Columns
          </button>
        </div>
      )}
    </aside>
  );
}
