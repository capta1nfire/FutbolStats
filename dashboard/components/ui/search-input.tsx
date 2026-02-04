"use client";

import * as React from "react";
import { Search, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface SearchInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange"> {
  value: string;
  onChange: (value: string) => void;
  onClear?: () => void;
}

/**
 * Search input with clear button (UniFi style)
 * - Search icon on the left
 * - X button appears on the right when there's text
 * - Clicking X clears the input
 */
const SearchInput = React.forwardRef<HTMLInputElement, SearchInputProps>(
  ({ className, value, onChange, onClear, placeholder = "Search...", ...props }, ref) => {
    const handleClear = () => {
      onChange("");
      onClear?.();
    };

    return (
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
        <input
          type="text"
          ref={ref}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={cn(
            // Match global Input styling (UniFi drawer feel)
            "flex h-9 w-full rounded-md border px-8 py-2 text-sm",
            "bg-[color:var(--field-bg)] hover:bg-[color:var(--field-bg-hover)] text-foreground",
            // No border on hover (only focus)
            "border-transparent",
            "placeholder:text-muted-foreground/70",
            "shadow-xs transition-[border-color,box-shadow,background-color,color] outline-none",
            "focus-visible:border-primary focus-visible:ring-0",
            "disabled:cursor-not-allowed disabled:opacity-50",
            className
          )}
          {...props}
        />
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Clear search"
          >
            <X className="h-4 w-4" strokeWidth={1.5} />
          </button>
        )}
      </div>
    );
  }
);

SearchInput.displayName = "SearchInput";

export { SearchInput };
