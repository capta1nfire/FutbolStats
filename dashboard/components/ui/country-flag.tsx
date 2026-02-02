"use client";

import Image from "next/image";
import { Globe } from "lucide-react";
import { cn } from "@/lib/utils";
import { getCountryIsoCode } from "@/lib/utils/country-flags";

interface CountryFlagProps {
  /** Country name (e.g., "England", "Brazil", "Germany") */
  country: string;
  /** Size in pixels (default: 16) */
  size?: number;
  /** Additional class names */
  className?: string;
  /** Use rounded style (default: true) */
  rounded?: boolean;
}

/**
 * Country Flag Component
 *
 * Renders a country flag using circle-flags SVGs.
 * Falls back to a globe icon if the country is not found.
 *
 * @example
 * <CountryFlag country="England" size={20} />
 * <CountryFlag country="Brazil" className="mr-2" />
 */
export function CountryFlag({
  country,
  size = 16,
  className,
  rounded = true,
}: CountryFlagProps) {
  const isoCode = getCountryIsoCode(country);

  if (!isoCode) {
    return (
      <Globe
        className={cn("text-muted-foreground", className)}
        style={{ width: size, height: size }}
      />
    );
  }

  return (
    <Image
      src={`/flags/${isoCode}.svg`}
      alt={`${country} flag`}
      width={size}
      height={size}
      className={cn(
        "object-cover",
        rounded && "rounded-full",
        className
      )}
    />
  );
}
