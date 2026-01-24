/**
 * Regional System - Locale + Timezone
 *
 * Native implementation using Intl APIs.
 * No external dependencies (Moment, Luxon, etc.)
 */

// ============================================================================
// Types
// ============================================================================

export interface RegionSettings {
  /** BCP 47 locale tag (e.g., "en-US", "es-CO") */
  locale: string;
  /** IANA timezone (e.g., "America/Bogota", "UTC") */
  timeZone: string;
  /** Hour cycle preference */
  hourCycle: "h12" | "h23";
}

/** LocalDate string in YYYY-MM-DD format (no time, no timezone) */
export type LocalDate = string;

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = "futbolstats-region";

const DEFAULT_LOCALE = "en-US";
const DEFAULT_TIMEZONE = "UTC";

// ============================================================================
// Defaults Detection
// ============================================================================

/**
 * Detect browser's preferred locale
 */
export function detectLocale(): string {
  if (typeof navigator !== "undefined" && navigator.language) {
    return navigator.language;
  }
  return DEFAULT_LOCALE;
}

/**
 * Detect browser's timezone
 */
export function detectTimeZone(): string {
  try {
    const resolved = Intl.DateTimeFormat().resolvedOptions();
    return resolved.timeZone || DEFAULT_TIMEZONE;
  } catch {
    return DEFAULT_TIMEZONE;
  }
}

/**
 * Infer hour cycle from locale
 * Spanish-speaking countries typically use 24h format
 */
export function inferHourCycle(locale: string): "h12" | "h23" {
  const lang = locale.split("-")[0].toLowerCase();
  // 24h format for Spanish, Portuguese, German, French, Italian, etc.
  const h24Languages = ["es", "pt", "de", "fr", "it", "nl", "pl", "ru", "ja", "ko", "zh"];
  return h24Languages.includes(lang) ? "h23" : "h12";
}

/**
 * Get default region settings based on browser detection
 */
export function getDefaultRegionSettings(): RegionSettings {
  const locale = detectLocale();
  return {
    locale,
    timeZone: detectTimeZone(),
    hourCycle: inferHourCycle(locale),
  };
}

// ============================================================================
// Persistence
// ============================================================================

/**
 * Load region settings from localStorage
 */
export function loadRegionSettings(): RegionSettings {
  if (typeof window === "undefined") {
    return getDefaultRegionSettings();
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Partial<RegionSettings>;
      const defaults = getDefaultRegionSettings();
      return {
        locale: parsed.locale || defaults.locale,
        timeZone: parsed.timeZone || defaults.timeZone,
        hourCycle: parsed.hourCycle || defaults.hourCycle,
      };
    }
  } catch {
    // Ignore parse errors
  }

  return getDefaultRegionSettings();
}

/**
 * Save region settings to localStorage
 */
export function saveRegionSettings(settings: RegionSettings): void {
  if (typeof window === "undefined") return;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Ignore storage errors (e.g., quota exceeded)
  }
}

// ============================================================================
// Timezone List
// ============================================================================

/**
 * Get list of supported IANA timezones
 * Uses Intl.supportedValuesOf if available, otherwise falls back to static list
 */
export function getSupportedTimeZones(): string[] {
  // Modern browsers support this (Chrome 93+, Firefox 93+, Safari 15.4+)
  if (typeof Intl !== "undefined" && "supportedValuesOf" in Intl) {
    try {
      return Intl.supportedValuesOf("timeZone") as string[];
    } catch {
      // Fallback if not supported
    }
  }

  // Static fallback list (common timezones)
  return [
    "UTC",
    // Americas
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Anchorage",
    "America/Bogota",
    "America/Lima",
    "America/Santiago",
    "America/Sao_Paulo",
    "America/Buenos_Aires",
    "America/Mexico_City",
    "America/Toronto",
    "America/Vancouver",
    // Europe
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Europe/Madrid",
    "Europe/Rome",
    "Europe/Amsterdam",
    "Europe/Brussels",
    "Europe/Zurich",
    "Europe/Vienna",
    "Europe/Stockholm",
    "Europe/Oslo",
    "Europe/Copenhagen",
    "Europe/Helsinki",
    "Europe/Warsaw",
    "Europe/Prague",
    "Europe/Budapest",
    "Europe/Athens",
    "Europe/Istanbul",
    "Europe/Moscow",
    "Europe/Kiev",
    // Asia
    "Asia/Tokyo",
    "Asia/Seoul",
    "Asia/Shanghai",
    "Asia/Hong_Kong",
    "Asia/Singapore",
    "Asia/Bangkok",
    "Asia/Jakarta",
    "Asia/Manila",
    "Asia/Kolkata",
    "Asia/Dubai",
    "Asia/Riyadh",
    "Asia/Tehran",
    "Asia/Jerusalem",
    // Oceania
    "Australia/Sydney",
    "Australia/Melbourne",
    "Australia/Perth",
    "Australia/Brisbane",
    "Pacific/Auckland",
    "Pacific/Fiji",
    // Africa
    "Africa/Cairo",
    "Africa/Johannesburg",
    "Africa/Lagos",
    "Africa/Nairobi",
    "Africa/Casablanca",
  ];
}

// ============================================================================
// Formatters
// ============================================================================

/**
 * Format a UTC ISO string as full date and time
 * Example: "Jan 23, 2026, 2:30 PM" or "23 ene 2026, 14:30"
 */
export function formatDateTime(isoUtc: string, region: RegionSettings): string {
  try {
    const date = new Date(isoUtc);
    if (isNaN(date.getTime())) return isoUtc;

    return new Intl.DateTimeFormat(region.locale, {
      timeZone: region.timeZone,
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hourCycle: region.hourCycle,
    }).format(date);
  } catch {
    return isoUtc;
  }
}

/**
 * Format a UTC ISO string as date only
 * Example: "Jan 23, 2026" or "23 ene 2026"
 */
export function formatDate(isoUtc: string, region: RegionSettings): string {
  try {
    const date = new Date(isoUtc);
    if (isNaN(date.getTime())) return isoUtc;

    return new Intl.DateTimeFormat(region.locale, {
      timeZone: region.timeZone,
      year: "numeric",
      month: "short",
      day: "numeric",
    }).format(date);
  } catch {
    return isoUtc;
  }
}

/**
 * Format a UTC ISO string as time only
 * Example: "2:30 PM" or "14:30"
 */
export function formatTime(isoUtc: string, region: RegionSettings): string {
  try {
    const date = new Date(isoUtc);
    if (isNaN(date.getTime())) return isoUtc;

    return new Intl.DateTimeFormat(region.locale, {
      timeZone: region.timeZone,
      hour: "numeric",
      minute: "2-digit",
      hourCycle: region.hourCycle,
    }).format(date);
  } catch {
    return isoUtc;
  }
}

/**
 * Format a UTC ISO string as short date (for tables)
 * Example: "Jan 23" or "23 ene"
 */
export function formatShortDate(isoUtc: string, region: RegionSettings): string {
  try {
    const date = new Date(isoUtc);
    if (isNaN(date.getTime())) return isoUtc;

    return new Intl.DateTimeFormat(region.locale, {
      timeZone: region.timeZone,
      month: "short",
      day: "numeric",
    }).format(date);
  } catch {
    return isoUtc;
  }
}

/**
 * Format a UTC ISO string as short date + time (for tables)
 * Example: "Jan 23, 14:30" or "23 ene, 14:30"
 */
export function formatShortDateTime(isoUtc: string, region: RegionSettings): string {
  try {
    const date = new Date(isoUtc);
    if (isNaN(date.getTime())) return isoUtc;

    return new Intl.DateTimeFormat(region.locale, {
      timeZone: region.timeZone,
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hourCycle: region.hourCycle,
    }).format(date);
  } catch {
    return isoUtc;
  }
}

/**
 * Get current time formatted for display
 */
export function formatCurrentTime(region: RegionSettings): string {
  return new Intl.DateTimeFormat(region.locale, {
    timeZone: region.timeZone,
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: region.hourCycle,
  }).format(new Date());
}

/**
 * Get current date formatted for display
 * Example: "Thu, Jan 23, 2026"
 */
export function formatCurrentDate(region: RegionSettings): string {
  return new Intl.DateTimeFormat(region.locale, {
    timeZone: region.timeZone,
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date());
}

// ============================================================================
// LocalDate Helpers
// ============================================================================

/**
 * Get today's date as LocalDate string (YYYY-MM-DD) in the given timezone
 */
export function getTodayLocalDate(timeZone: string): LocalDate {
  const now = new Date();
  // Format in the target timezone
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(now);

  const year = parts.find((p) => p.type === "year")?.value || "2026";
  const month = parts.find((p) => p.type === "month")?.value || "01";
  const day = parts.find((p) => p.type === "day")?.value || "01";

  return `${year}-${month}-${day}`;
}

/**
 * Convert a LocalDate to UTC ISO string for start of day in the given timezone
 * Example: "2026-01-23" in "America/Bogota" -> "2026-01-23T05:00:00.000Z"
 */
export function localDateToUtcStartIso(localDate: LocalDate, timeZone: string): string {
  // Parse the date parts
  const [year, month, day] = localDate.split("-").map(Number);

  // Create a date string that represents midnight in the target timezone
  // We use a trick: create the date in UTC, then adjust based on timezone offset
  const targetDate = new Date(Date.UTC(year, month - 1, day, 0, 0, 0, 0));

  // Get the offset for this date in the target timezone
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    timeZoneName: "shortOffset",
  });
  const parts = formatter.formatToParts(targetDate);
  const offsetPart = parts.find((p) => p.type === "timeZoneName")?.value || "GMT";

  // Parse offset (e.g., "GMT-5" -> -5, "GMT+1" -> 1)
  const offsetMatch = offsetPart.match(/GMT([+-])?(\d+)?(?::(\d+))?/);
  let offsetMinutes = 0;
  if (offsetMatch) {
    const sign = offsetMatch[1] === "-" ? -1 : 1;
    const hours = parseInt(offsetMatch[2] || "0", 10);
    const minutes = parseInt(offsetMatch[3] || "0", 10);
    offsetMinutes = sign * (hours * 60 + minutes);
  }

  // Adjust: if timezone is GMT-5, midnight local is 05:00 UTC
  const utcTime = targetDate.getTime() - offsetMinutes * 60 * 1000;
  return new Date(utcTime).toISOString();
}

/**
 * Convert a LocalDate to UTC ISO string for end of day in the given timezone
 * Example: "2026-01-23" in "America/Bogota" -> "2026-01-24T04:59:59.999Z"
 */
export function localDateToUtcEndIso(localDate: LocalDate, timeZone: string): string {
  // Parse the date parts
  const [year, month, day] = localDate.split("-").map(Number);

  // Create a date string that represents 23:59:59.999 in the target timezone
  const targetDate = new Date(Date.UTC(year, month - 1, day, 23, 59, 59, 999));

  // Get the offset for this date in the target timezone
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    timeZoneName: "shortOffset",
  });
  const parts = formatter.formatToParts(targetDate);
  const offsetPart = parts.find((p) => p.type === "timeZoneName")?.value || "GMT";

  // Parse offset
  const offsetMatch = offsetPart.match(/GMT([+-])?(\d+)?(?::(\d+))?/);
  let offsetMinutes = 0;
  if (offsetMatch) {
    const sign = offsetMatch[1] === "-" ? -1 : 1;
    const hours = parseInt(offsetMatch[2] || "0", 10);
    const minutes = parseInt(offsetMatch[3] || "0", 10);
    offsetMinutes = sign * (hours * 60 + minutes);
  }

  // Adjust for timezone offset
  const utcTime = targetDate.getTime() - offsetMinutes * 60 * 1000;
  return new Date(utcTime).toISOString();
}

/**
 * Convert a Date object to LocalDate string in the given timezone
 */
export function dateToLocalDate(date: Date, timeZone: string): LocalDate {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);

  const year = parts.find((p) => p.type === "year")?.value || "2026";
  const month = parts.find((p) => p.type === "month")?.value || "01";
  const day = parts.find((p) => p.type === "day")?.value || "01";

  return `${year}-${month}-${day}`;
}

/**
 * Parse a LocalDate string to a Date object (at noon to avoid DST issues)
 * This is useful for calendar components that need Date objects
 */
export function localDateToDate(localDate: LocalDate): Date {
  const [year, month, day] = localDate.split("-").map(Number);
  // Use noon to avoid any DST edge cases
  return new Date(year, month - 1, day, 12, 0, 0, 0);
}

/**
 * Validate a LocalDate string format
 */
export function isValidLocalDate(value: string): value is LocalDate {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const [year, month, day] = value.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  return (
    date.getFullYear() === year &&
    date.getMonth() === month - 1 &&
    date.getDate() === day
  );
}
