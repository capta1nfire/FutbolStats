import * as React from "react"

import { cn } from "@/lib/utils"

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        // UniFi-style textarea: subtle fill + visible border on dark surfaces
        "flex field-sizing-content min-h-16 w-full rounded-md border px-3 py-2 text-sm",
        "bg-[color:var(--field-bg)] hover:bg-[color:var(--field-bg-hover)] text-foreground",
        // No border on hover (only focus)
        "border-transparent",
        "placeholder:text-muted-foreground/70",
        "shadow-xs transition-[border-color,box-shadow,background-color,color] outline-none",
        "focus-visible:border-primary focus-visible:ring-0",
        "aria-invalid:border-primary",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
}

export { Textarea }
