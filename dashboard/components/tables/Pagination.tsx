"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface PaginationProps {
  currentPage: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (size: number) => void;
  pageSizeOptions?: number[];
  /** Label for items (e.g., "Logs", "Matches"). Defaults to no label. */
  itemLabel?: string;
}

/**
 * UniFi-style table footer pagination
 * Layout: [< >] 1-100 of 17652 Logs     Rows Per Page: [100 v]
 */
export function Pagination({
  currentPage,
  totalItems,
  pageSize,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [50, 100, 200],
  itemLabel,
}: PaginationProps) {
  const totalPages = Math.ceil(totalItems / pageSize);
  const startItem = (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);

  if (totalItems === 0) {
    return null;
  }

  const isPrevDisabled = currentPage === 1;
  const isNextDisabled = currentPage === totalPages;

  return (
    <div className="py-4 px-4 shadow-elevation-up bg-background relative z-10">
      <div className="h-8 flex items-center">
        {/* All content aligned left: Nav buttons + Range text + Rows per page */}
        <div className="flex items-center gap-6">
          {/* Segmented prev/next buttons */}
          <div className="flex items-center">
            <button
              type="button"
              onClick={() => onPageChange(currentPage - 1)}
              disabled={isPrevDisabled}
              aria-label="Previous page"
              className={cn(
                "w-[26px] h-[24px] flex items-center justify-center",
                "border border-[#3f3f3f] rounded-l",
                "transition-colors",
                isPrevDisabled
                  ? "border-[#1e1e1e] cursor-default pointer-events-none"
                  : "hover:bg-[#05254d] focus:shadow-[0_0_0_1px_#0e65d8] focus:outline-none"
              )}
            >
              <ChevronLeft
                className={cn(
                  "w-3.5 h-3.5",
                  isPrevDisabled
                    ? "text-[rgba(255,255,255,0.25)]"
                    : "text-[rgba(255,255,255,0.65)]"
                )}
                strokeWidth={2}
              />
            </button>
            <button
              type="button"
              onClick={() => onPageChange(currentPage + 1)}
              disabled={isNextDisabled}
              aria-label="Next page"
              className={cn(
                "w-[26px] h-[24px] flex items-center justify-center",
                "border border-[#3f3f3f] border-l-0 rounded-r",
                "transition-colors",
                isNextDisabled
                  ? "border-[#1e1e1e] cursor-default pointer-events-none"
                  : "hover:bg-[#05254d] focus:shadow-[0_0_0_1px_#0e65d8] focus:outline-none"
              )}
            >
              <ChevronRight
                className={cn(
                  "w-3.5 h-3.5",
                  isNextDisabled
                    ? "text-[rgba(255,255,255,0.25)]"
                    : "text-[rgba(255,255,255,0.65)]"
                )}
                strokeWidth={2}
              />
            </button>
          </div>

          {/* Range text */}
          <span className="text-[11px] leading-4 font-normal text-[#dee0e3]">
            {startItem}-{endItem} of {totalItems}{itemLabel ? ` ${itemLabel}` : ""}
          </span>

          {/* Rows per page selector */}
          {onPageSizeChange && (
            <div className="flex items-center gap-2">
              <span className="text-[11px] leading-4 font-normal text-[#dee0e3]">
                Rows Per Page:
              </span>
              <Select
                value={String(pageSize)}
                onValueChange={(value) => onPageSizeChange(Number(value))}
              >
                <SelectTrigger
                  size="sm"
                  className="w-[80px] h-8 bg-[rgba(249,250,250,0.04)] border-transparent hover:border-[#b7bcc2] text-[13px] text-[#f9fafa]"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {pageSizeOptions.map((size) => (
                    <SelectItem key={size} value={String(size)}>
                      {size}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
