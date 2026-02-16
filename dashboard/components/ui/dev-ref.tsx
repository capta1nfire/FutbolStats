"use client";

import { useState, useCallback } from "react";

interface DevRefProps {
  /** Component path, e.g. "dashboard/components/football/TeamDrawer.tsx:TeamInfoSection" */
  path: string;
  children: React.ReactNode;
}

/**
 * Invisible dev helper — copies component path to clipboard on click.
 * Brief opacity flash as feedback, no toast/popup.
 */
export function DevRef({ path, children }: DevRefProps) {
  const [copied, setCopied] = useState(false);

  const handleClick = useCallback(() => {
    navigator.clipboard.writeText(path);
    setCopied(true);
    setTimeout(() => setCopied(false), 600);
  }, [path]);

  return (
    <span
      onClick={handleClick}
      className="select-none transition-opacity duration-300"
      style={{ opacity: copied ? 0.4 : 1 }}
    >
      {children}
    </span>
  );
}
