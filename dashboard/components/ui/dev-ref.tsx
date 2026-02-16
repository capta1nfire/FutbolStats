"use client";

import { toast } from "sonner";

interface DevRefProps {
  /** Component path, e.g. "dashboard/components/football/TeamDrawer.tsx:TeamInfoSection" */
  path: string;
  children: React.ReactNode;
}

/**
 * Invisible dev helper — copies component path to clipboard on click.
 * No visual change: same cursor, no underline, no color.
 * Shows a subtle toast on copy.
 */
export function DevRef({ path, children }: DevRefProps) {
  const handleClick = (e: React.MouseEvent) => {
    // Only fire on the element itself, not bubbled from children with their own handlers
    if (e.detail === 1) {
      navigator.clipboard.writeText(path);
      toast.success(path, { duration: 1500 });
    }
  };

  return (
    <span onClick={handleClick} className="select-none">
      {children}
    </span>
  );
}
