"use client";

interface DevRefProps {
  /** Semantic component path, e.g. "TeamDrawer.tsx:TeamInfoSection:Venue" */
  path: string;
  children: React.ReactNode;
}

/**
 * Semantic override marker for DevRef system.
 * Renders a <span> with data-dev-ref attribute.
 * The DevRefProvider reads this on click and includes it in the clipboard text.
 */
export function DevRef({ path, children }: DevRefProps) {
  return (
    <span data-dev-ref={path}>
      {children}
    </span>
  );
}
