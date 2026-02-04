import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        ref={ref}
        type={type}
        data-slot="input"
        className={cn(
          // UniFi-style input: subtle fill + visible border on dark surfaces
          "h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm",
          "bg-[color:var(--field-bg)] hover:bg-[color:var(--field-bg-hover)] text-foreground",
          // No border on hover (only focus)
          "border-transparent",
          "placeholder:text-muted-foreground/70",
          "selection:bg-primary selection:text-primary-foreground",
          "shadow-xs transition-[border-color,box-shadow,background-color,color] outline-none",
          // File input styling (keep consistent)
          "file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground",
          // Focus + validation
          "focus-visible:border-primary focus-visible:ring-0",
          "aria-invalid:border-primary",
          // Disabled
          "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        {...props}
      />
    )
  }
)

Input.displayName = "Input"

export { Input }
