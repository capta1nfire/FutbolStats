"use client";

import { ReactNode, useState, useEffect } from "react";
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
  /** If true, only one option can be selected at a time (radio behavior) */
  singleSelect?: boolean;
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
  /** Optional content to render below title, before search (e.g., view tabs) */
  headerContent?: ReactNode;
  /** Optional content to render below search (e.g., quick filters) */
  quickFilterContent?: ReactNode;
  /** Section title to display in header (e.g., "Matches", "Jobs") */
  title?: string;
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
  headerContent,
  quickFilterContent,
  title = "Filters",
}: FilterPanelProps) {
  // Track client-side mount to avoid Radix Accordion hydration mismatch
  // Uses useSyncExternalStore instead of useState+useEffect to avoid lint error
  const mounted = useHasMounted();

  // Controlled accordion state - all groups expanded by default
  const [expandedGroups, setExpandedGroups] = useState<string[]>([]);

  // Update expanded groups when groups change (e.g., switching views)
  useEffect(() => {
    // We intentionally sync derived state when the set of groups changes.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setExpandedGroups(groups.map((g) => g.id));
  }, [groups]);

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
                <ChevronRight className="h-4 w-4" strokeWidth={1.5} />
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
                <Filter className="h-4 w-4" strokeWidth={1.5} />
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
  // overflow-hidden ensures flex-col works and footer stays at bottom
  return (
    <aside className="w-[277px] shrink-0 border-r border-border bg-sidebar flex flex-col overflow-hidden transition-smooth">
      {/* Header - collapse button hidden when CustomizeColumnsPanel is open */}
      <div className="h-12 flex items-center justify-between px-3">
        <span className="text-sm font-semibold text-foreground">{title}</span>
        {!customizeColumnsOpen && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleCollapse}
            className="h-8 w-8"
            aria-label="Collapse filters"
          >
            <ChevronLeft className="h-4 w-4" strokeWidth={1.5} />
          </Button>
        )}
      </div>

      {/* Header content slot (e.g., view tabs) */}
      {headerContent && (
        <div className="px-3 pb-3">
          {headerContent}
        </div>
      )}

      {/* Search */}
      <div className="p-3">
        <SearchInput
          placeholder="Search..."
          value={searchValue}
          onChange={(value) => onSearchChange?.(value)}
        />
      </div>

      {/* Quick filter slot (e.g., severity bars) */}
      {quickFilterContent && (
        <div className="px-3 pb-3">
          {quickFilterContent}
        </div>
      )}

      {/* Filter groups - render only on client to avoid Radix ID hydration mismatch */}
      {/* min-h-0 is the flexbox trick to allow shrinking below content size */}
      <ScrollArea className="flex-1 min-h-0">
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
          // Client: render actual Accordion with controlled state
          <Accordion type="multiple" value={expandedGroups} onValueChange={setExpandedGroups} className="px-3">
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
                        {group.singleSelect ? (
                          <span
                            className={`h-4 w-4 rounded-full border flex items-center justify-center transition-colors ${
                              option.checked
                                ? "border-primary bg-primary"
                                : "border-muted-foreground"
                            }`}
                            onClick={() => onFilterChange?.(group.id, option.id, true)}
                          >
                            {option.checked && (
                              <span className="h-1.5 w-1.5 rounded-full bg-primary-foreground" />
                            )}
                          </span>
                        ) : (
                          <Checkbox
                            id={`${group.id}-${option.id}`}
                            checked={option.checked}
                            onCheckedChange={(checked) =>
                              onFilterChange?.(group.id, option.id, checked === true)
                            }
                          />
                        )}
                        <span
                          className="text-sm text-muted-foreground group-hover:text-foreground flex-1"
                          onClick={() => group.singleSelect && onFilterChange?.(group.id, option.id, true)}
                        >
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

      {/* Footer with Customize Columns link - same height as table pagination (py-4 + h-8 content) */}
      {/* Shadow uses ::before pseudo-element with gradient to render above ScrollArea content */}
      {showCustomizeColumns && (
        <div className="px-4 py-4 flex items-center bg-sidebar shrink-0 relative z-10 before:absolute before:left-0 before:right-0 before:bottom-full before:h-4 before:bg-gradient-to-t before:from-black/30 before:to-transparent before:pointer-events-none">
          <div className="h-8 flex items-center">
            <Button
              variant="actionLink"
              size="sm"
              className="px-0 h-auto"
              onClick={onCustomizeColumnsClick}
            >
              Customize Columns
            </Button>
          </div>
        </div>
      )}
    </aside>
  );
}
