"use client";

import Image from "next/image";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { Shield } from "lucide-react";

interface TeamLogoProps {
  /** Logo URL from team logos map */
  src: string | null;
  /** Team name for alt text */
  teamName: string;
  /** Size in pixels (default: 16) */
  size?: number;
  /** Additional class names */
  className?: string;
}

/**
 * Team Logo Component
 *
 * Renders a team shield/logo with fallback handling.
 * Uses next/image for optimization and lazy loading.
 *
 * @example
 * <TeamLogo src={getLogoUrl("River Plate")} teamName="River Plate" size={20} />
 */
export function TeamLogo({
  src,
  teamName,
  size = 16,
  className,
}: TeamLogoProps) {
  const [hasError, setHasError] = useState(false);

  // Show placeholder if no src or error loading
  if (!src || hasError) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-sm bg-muted/50",
          className
        )}
        style={{ width: size, height: size }}
        title={teamName}
      >
        <Shield
          className="text-muted-foreground/50"
          style={{ width: size * 0.7, height: size * 0.7 }}
        />
      </div>
    );
  }

  return (
    <Image
      src={src}
      alt={`${teamName} logo`}
      width={size}
      height={size}
      className={cn("object-contain", className)}
      onError={() => setHasError(true)}
      loading="lazy"
      unoptimized // External images from api-sports.io
    />
  );
}
