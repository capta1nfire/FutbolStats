"use client";

import { useState, useEffect, useCallback } from "react";
import { useTeamEnrichmentPutMutation, useTeamEnrichmentDeleteMutation } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { SurfaceCard } from "@/components/ui/surface-card";
import { Label } from "@/components/ui/label";
import {
  Loader2,
  CheckCircle,
  AlertTriangle,
  ExternalLink,
  Trash2,
  Database,
  Edit3,
} from "lucide-react";
import { toast } from "sonner";
import type { TeamWikidataEnrichment } from "@/lib/types/football";

// =============================================================================
// Validation
// =============================================================================

const TWITTER_HANDLE_REGEX = /^[A-Za-z0-9_]{1,15}$/;
const INSTAGRAM_HANDLE_REGEX = /^[A-Za-z0-9_.]{1,30}$/;
const WEBSITE_URL_REGEX = /^https?:\/\/.+$/;

interface ValidationResult {
  isValid: boolean;
  error?: string;
}

function validateTwitterHandle(handle: string): ValidationResult {
  if (!handle.trim()) return { isValid: true };
  // Remove @ if present
  const cleaned = handle.trim().replace(/^@/, "");
  if (!TWITTER_HANDLE_REGEX.test(cleaned)) {
    return { isValid: false, error: "1-15 chars: letters, numbers, underscore" };
  }
  return { isValid: true };
}

function validateInstagramHandle(handle: string): ValidationResult {
  if (!handle.trim()) return { isValid: true };
  // Remove @ if present
  const cleaned = handle.trim().replace(/^@/, "");
  if (!INSTAGRAM_HANDLE_REGEX.test(cleaned)) {
    return { isValid: false, error: "1-30 chars: letters, numbers, underscore, dot" };
  }
  return { isValid: true };
}

function validateWebsite(url: string): ValidationResult {
  if (!url.trim()) return { isValid: true };
  if (!WEBSITE_URL_REGEX.test(url.trim())) {
    return { isValid: false, error: "Must start with http:// or https://" };
  }
  return { isValid: true };
}

function validateCapacity(value: string): ValidationResult {
  if (!value.trim()) return { isValid: true };
  const num = parseInt(value, 10);
  if (isNaN(num) || num < 0 || num >= 200000) {
    return { isValid: false, error: "0-199,999" };
  }
  return { isValid: true };
}

// =============================================================================
// Component
// =============================================================================

interface TeamEnrichmentSettingsProps {
  teamId: number;
  teamName: string;
  enrichment?: TeamWikidataEnrichment | null;
}

export function TeamEnrichmentSettings({
  teamId,
  teamName,
  enrichment,
}: TeamEnrichmentSettingsProps) {
  // Form state - use override values for edit, fall back to effective values for display
  const [fullName, setFullName] = useState(enrichment?.override?.full_name ?? "");
  const [shortName, setShortName] = useState(enrichment?.override?.short_name ?? "");
  const [stadiumName, setStadiumName] = useState(enrichment?.override?.stadium_name ?? "");
  const [stadiumCapacity, setStadiumCapacity] = useState(
    enrichment?.override?.stadium_capacity?.toString() ?? ""
  );
  const [website, setWebsite] = useState(enrichment?.override?.website ?? "");
  const [twitter, setTwitter] = useState(enrichment?.override?.twitter ?? "");
  const [instagram, setInstagram] = useState(enrichment?.override?.instagram ?? "");
  const [source, setSource] = useState(enrichment?.override?.source ?? "manual");
  const [notes, setNotes] = useState(enrichment?.override?.notes ?? "");

  // Validation errors
  const [errors, setErrors] = useState<Record<string, string | null>>({});

  // Mutations
  const putMutation = useTeamEnrichmentPutMutation();
  const deleteMutation = useTeamEnrichmentDeleteMutation();

  // Sync form state when enrichment prop changes
  useEffect(() => {
    setFullName(enrichment?.override?.full_name ?? "");
    setShortName(enrichment?.override?.short_name ?? "");
    setStadiumName(enrichment?.override?.stadium_name ?? "");
    setStadiumCapacity(enrichment?.override?.stadium_capacity?.toString() ?? "");
    setWebsite(enrichment?.override?.website ?? "");
    setTwitter(enrichment?.override?.twitter ?? "");
    setInstagram(enrichment?.override?.instagram ?? "");
    setSource(enrichment?.override?.source ?? "manual");
    setNotes(enrichment?.override?.notes ?? "");
    setErrors({});
  }, [enrichment]);

  // Check if form has changes from current override
  const isDirty =
    fullName !== (enrichment?.override?.full_name ?? "") ||
    shortName !== (enrichment?.override?.short_name ?? "") ||
    stadiumName !== (enrichment?.override?.stadium_name ?? "") ||
    stadiumCapacity !== (enrichment?.override?.stadium_capacity?.toString() ?? "") ||
    website !== (enrichment?.override?.website ?? "") ||
    twitter !== (enrichment?.override?.twitter ?? "") ||
    instagram !== (enrichment?.override?.instagram ?? "") ||
    notes !== (enrichment?.override?.notes ?? "");

  // Validate all fields
  const validateAll = useCallback(() => {
    const newErrors: Record<string, string | null> = {};

    const twitterResult = validateTwitterHandle(twitter);
    if (!twitterResult.isValid) newErrors.twitter = twitterResult.error ?? null;

    const instagramResult = validateInstagramHandle(instagram);
    if (!instagramResult.isValid) newErrors.instagram = instagramResult.error ?? null;

    const websiteResult = validateWebsite(website);
    if (!websiteResult.isValid) newErrors.website = websiteResult.error ?? null;

    const capacityResult = validateCapacity(stadiumCapacity);
    if (!capacityResult.isValid) newErrors.stadiumCapacity = capacityResult.error ?? null;

    setErrors(newErrors);
    return Object.values(newErrors).every((e) => e === null || e === undefined);
  }, [twitter, instagram, website, stadiumCapacity]);

  // Handle reset
  const handleReset = useCallback(() => {
    setFullName(enrichment?.override?.full_name ?? "");
    setShortName(enrichment?.override?.short_name ?? "");
    setStadiumName(enrichment?.override?.stadium_name ?? "");
    setStadiumCapacity(enrichment?.override?.stadium_capacity?.toString() ?? "");
    setWebsite(enrichment?.override?.website ?? "");
    setTwitter(enrichment?.override?.twitter ?? "");
    setInstagram(enrichment?.override?.instagram ?? "");
    setNotes(enrichment?.override?.notes ?? "");
    setErrors({});
  }, [enrichment]);

  // Handle save
  const handleSave = useCallback(() => {
    if (!validateAll()) return;

    // Clean handles (remove @)
    const cleanTwitter = twitter.trim().replace(/^@/, "") || null;
    const cleanInstagram = instagram.trim().replace(/^@/, "") || null;

    putMutation.mutate(
      {
        teamId,
        data: {
          full_name: fullName.trim() || null,
          short_name: shortName.trim() || null,
          stadium_name: stadiumName.trim() || null,
          stadium_capacity: stadiumCapacity.trim()
            ? parseInt(stadiumCapacity, 10)
            : null,
          website: website.trim() || null,
          twitter_handle: cleanTwitter,
          instagram_handle: cleanInstagram,
          source: source || "manual",
          notes: notes.trim() || null,
        },
      },
      {
        onSuccess: (res) => {
          if (res.action === "deleted") {
            toast.success("Override cleared (all fields empty)");
          } else {
            toast.success("Override saved");
          }
        },
        onError: (error) => {
          toast.error(error.message);
        },
      }
    );
  }, [
    teamId,
    fullName,
    shortName,
    stadiumName,
    stadiumCapacity,
    website,
    twitter,
    instagram,
    source,
    notes,
    validateAll,
    putMutation,
  ]);

  // Handle delete all overrides
  const handleDelete = useCallback(() => {
    deleteMutation.mutate(
      { teamId },
      {
        onSuccess: () => {
          toast.success("Override removed");
        },
        onError: (error) => {
          toast.error(error.message);
        },
      }
    );
  }, [teamId, deleteMutation]);

  const isPending = putMutation.isPending || deleteMutation.isPending;
  const hasValidationErrors = Object.values(errors).some((e) => e);
  const canSave = isDirty && !hasValidationErrors && !isPending;

  // If no enrichment data at all, show placeholder
  if (!enrichment) {
    return (
      <SurfaceCard className="space-y-3">
        <h4 className="text-sm font-medium flex items-center gap-2">
          <Database className="h-4 w-4 text-muted-foreground" />
          Team Enrichment
        </h4>
        <p className="text-xs text-muted-foreground">
          No Wikidata enrichment available for this team.
        </p>
      </SurfaceCard>
    );
  }

  return (
    <SurfaceCard className="space-y-4">
      {/* Header with source badge */}
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium flex items-center gap-2">
          <Database className="h-4 w-4 text-muted-foreground" />
          Team Enrichment
        </h4>
        <Badge
          variant={enrichment.has_override ? "default" : "secondary"}
          className="text-xs"
        >
          {enrichment.has_override ? (
            <Edit3 className="h-3 w-3 mr-1" />
          ) : (
            <CheckCircle className="h-3 w-3 mr-1" />
          )}
          {enrichment.source_badge?.label || "Wikidata"}
        </Badge>
      </div>

      {/* Effective values (read-only display) */}
      <div className="bg-background/20 rounded-lg p-3 border border-border space-y-2">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <CheckCircle className="h-3 w-3" />
          <span>Current values</span>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          {enrichment.full_name && (
            <div className="col-span-2">
              <span className="text-muted-foreground">Full name:</span>{" "}
              <span className="text-foreground">{enrichment.full_name}</span>
            </div>
          )}
          {enrichment.stadium_name && (
            <div>
              <span className="text-muted-foreground">Stadium:</span>{" "}
              <span className="text-foreground">{enrichment.stadium_name}</span>
            </div>
          )}
          {enrichment.stadium_capacity && (
            <div>
              <span className="text-muted-foreground">Capacity:</span>{" "}
              <span className="text-foreground">
                {enrichment.stadium_capacity.toLocaleString()}
              </span>
            </div>
          )}
          {enrichment.twitter && (
            <div>
              <span className="text-muted-foreground">X.com:</span>{" "}
              <a
                href={`https://x.com/${enrichment.twitter}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                @{enrichment.twitter}
              </a>
            </div>
          )}
          {enrichment.instagram && (
            <div>
              <span className="text-muted-foreground">Instagram:</span>{" "}
              <a
                href={`https://instagram.com/${enrichment.instagram}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                @{enrichment.instagram}
              </a>
            </div>
          )}
          {enrichment.website && (
            <div className="col-span-2">
              <span className="text-muted-foreground">Website:</span>{" "}
              <a
                href={enrichment.website}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                {enrichment.website}
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Override form */}
      <div className="space-y-3">
        <div className="text-xs text-muted-foreground">
          Override values (leave empty to use Wikidata)
        </div>

        {/* Social handles row */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label htmlFor={`team-${teamId}-twitter`} className="text-xs">
              X.com handle
            </Label>
            <Input
              id={`team-${teamId}-twitter`}
              type="text"
              placeholder="@handle"
              value={twitter}
              onChange={(e) => setTwitter(e.target.value)}
              disabled={isPending}
              className="h-8 text-sm"
              aria-invalid={!!errors.twitter}
            />
            {errors.twitter && (
              <p className="text-xs text-destructive">{errors.twitter}</p>
            )}
          </div>
          <div className="space-y-1">
            <Label htmlFor={`team-${teamId}-instagram`} className="text-xs">
              Instagram handle
            </Label>
            <Input
              id={`team-${teamId}-instagram`}
              type="text"
              placeholder="@handle"
              value={instagram}
              onChange={(e) => setInstagram(e.target.value)}
              disabled={isPending}
              className="h-8 text-sm"
              aria-invalid={!!errors.instagram}
            />
            {errors.instagram && (
              <p className="text-xs text-destructive">{errors.instagram}</p>
            )}
          </div>
        </div>

        {/* Website */}
        <div className="space-y-1">
          <Label htmlFor={`team-${teamId}-website`} className="text-xs">
            Website
          </Label>
          <Input
            id={`team-${teamId}-website`}
            type="url"
            placeholder="https://..."
            value={website}
            onChange={(e) => setWebsite(e.target.value)}
            disabled={isPending}
            className="h-8 text-sm"
            aria-invalid={!!errors.website}
          />
          {errors.website && (
            <p className="text-xs text-destructive">{errors.website}</p>
          )}
        </div>

        {/* Stadium row */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label htmlFor={`team-${teamId}-stadium`} className="text-xs">
              Stadium name
            </Label>
            <Input
              id={`team-${teamId}-stadium`}
              type="text"
              placeholder="Stadium name"
              value={stadiumName}
              onChange={(e) => setStadiumName(e.target.value)}
              disabled={isPending}
              className="h-8 text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor={`team-${teamId}-capacity`} className="text-xs">
              Capacity
            </Label>
            <Input
              id={`team-${teamId}-capacity`}
              type="number"
              placeholder="0-199999"
              value={stadiumCapacity}
              onChange={(e) => setStadiumCapacity(e.target.value)}
              disabled={isPending}
              className="h-8 text-sm"
              min={0}
              max={199999}
              aria-invalid={!!errors.stadiumCapacity}
            />
            {errors.stadiumCapacity && (
              <p className="text-xs text-destructive">{errors.stadiumCapacity}</p>
            )}
          </div>
        </div>

        {/* Names row */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label htmlFor={`team-${teamId}-fullname`} className="text-xs">
              Full name
            </Label>
            <Input
              id={`team-${teamId}-fullname`}
              type="text"
              placeholder="Full official name"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              disabled={isPending}
              className="h-8 text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor={`team-${teamId}-shortname`} className="text-xs">
              Short name
            </Label>
            <Input
              id={`team-${teamId}-shortname`}
              type="text"
              placeholder="Short name"
              value={shortName}
              onChange={(e) => setShortName(e.target.value)}
              disabled={isPending}
              className="h-8 text-sm"
            />
          </div>
        </div>

        {/* Notes */}
        <div className="space-y-1">
          <Label htmlFor={`team-${teamId}-notes`} className="text-xs">
            Notes <span className="opacity-50">(optional)</span>
          </Label>
          <Input
            id={`team-${teamId}-notes`}
            type="text"
            placeholder="Reason for override..."
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            disabled={isPending}
            className="h-8 text-sm"
          />
        </div>
      </div>

      {/* Action buttons */}
      {(isDirty || enrichment.has_override) && (
        <div className="flex items-center justify-between pt-2 border-t border-border">
          {/* Delete button - only show if there's an existing override */}
          {enrichment.has_override && !isDirty && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDelete}
              disabled={isPending}
              className="text-destructive hover:text-destructive"
            >
              {deleteMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4 mr-1" />
              )}
              Remove override
            </Button>
          )}
          {!enrichment.has_override && !isDirty && <div />}

          {/* Save/Cancel buttons - only show when dirty */}
          {isDirty && (
            <div className="flex items-center gap-3 ml-auto">
              <button
                type="button"
                onClick={handleReset}
                disabled={isPending}
                className="text-sm px-3 py-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-[color:var(--field-bg-hover)] transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <Button
                size="sm"
                onClick={handleSave}
                disabled={!canSave}
              >
                {putMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  "Apply Changes"
                )}
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Override info */}
      {enrichment.has_override && enrichment.override?.updated_at && (
        <div className="text-xs text-muted-foreground">
          Override by {enrichment.override.source || "manual"} ·{" "}
          {new Date(enrichment.override.updated_at).toLocaleDateString()}
        </div>
      )}
    </SurfaceCard>
  );
}
