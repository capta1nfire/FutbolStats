"use client";

import { useState, useEffect, useCallback, useImperativeHandle, forwardRef } from "react";
import type { Ref } from "react";
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

const WIKIDATA_QID_REGEX = /^Q\d{1,10}$/;

function validateStadiumWikidataId(value: string): ValidationResult {
  if (!value.trim()) return { isValid: true };
  if (!WIKIDATA_QID_REGEX.test(value.trim())) {
    return { isValid: false, error: "Q-number format (e.g. Q12345)" };
  }
  return { isValid: true };
}

// =============================================================================
// Component
// =============================================================================

export interface TeamEnrichmentHandle {
  isDirty: boolean;
  canSave: boolean;
  isPending: boolean;
  handleSave: () => void;
  handleReset: () => void;
}

interface TeamEnrichmentSettingsProps {
  teamId: number;
  teamName: string;
  enrichment?: TeamWikidataEnrichment | null;
  notes: string;
  onNotesChange: (value: string) => void;
}

export const TeamEnrichmentSettings = forwardRef(function TeamEnrichmentSettings(
  {
    teamId,
    teamName,
    enrichment,
    notes,
    onNotesChange,
  }: TeamEnrichmentSettingsProps,
  ref: Ref<TeamEnrichmentHandle>,
) {
  // Form state - use override values for edit, fall back to effective values for display
  const [fullName, setFullName] = useState(enrichment?.override?.full_name ?? "");
  const [shortName, setShortName] = useState(enrichment?.override?.short_name ?? "");
  const [stadiumName, setStadiumName] = useState(enrichment?.override?.stadium_name ?? "");
  const [stadiumCapacity, setStadiumCapacity] = useState(
    enrichment?.override?.stadium_capacity?.toString() ?? ""
  );
  const [stadiumWikidataId, setStadiumWikidataId] = useState(
    enrichment?.override?.stadium_wikidata_id ?? ""
  );
  const [website, setWebsite] = useState(enrichment?.override?.website ?? "");
  const [twitter, setTwitter] = useState(enrichment?.override?.twitter ?? "");
  const [instagram, setInstagram] = useState(enrichment?.override?.instagram ?? "");
  const [source, setSource] = useState(enrichment?.override?.source ?? "manual");

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
    setStadiumWikidataId(enrichment?.override?.stadium_wikidata_id ?? "");
    setWebsite(enrichment?.override?.website ?? "");
    setTwitter(enrichment?.override?.twitter ?? "");
    setInstagram(enrichment?.override?.instagram ?? "");
    setSource(enrichment?.override?.source ?? "manual");
    setErrors({});
  }, [enrichment]);

  // Check if form has changes from current override
  const isDirty =
    fullName !== (enrichment?.override?.full_name ?? "") ||
    shortName !== (enrichment?.override?.short_name ?? "") ||
    stadiumName !== (enrichment?.override?.stadium_name ?? "") ||
    stadiumCapacity !== (enrichment?.override?.stadium_capacity?.toString() ?? "") ||
    stadiumWikidataId !== (enrichment?.override?.stadium_wikidata_id ?? "") ||
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

    const wikidataResult = validateStadiumWikidataId(stadiumWikidataId);
    if (!wikidataResult.isValid) newErrors.stadiumWikidataId = wikidataResult.error ?? null;

    setErrors(newErrors);
    return Object.values(newErrors).every((e) => e === null || e === undefined);
  }, [twitter, instagram, website, stadiumCapacity, stadiumWikidataId]);

  // Handle reset
  const handleReset = useCallback(() => {
    setFullName(enrichment?.override?.full_name ?? "");
    setShortName(enrichment?.override?.short_name ?? "");
    setStadiumName(enrichment?.override?.stadium_name ?? "");
    setStadiumCapacity(enrichment?.override?.stadium_capacity?.toString() ?? "");
    setStadiumWikidataId(enrichment?.override?.stadium_wikidata_id ?? "");
    setWebsite(enrichment?.override?.website ?? "");
    setTwitter(enrichment?.override?.twitter ?? "");
    setInstagram(enrichment?.override?.instagram ?? "");
    onNotesChange(enrichment?.override?.notes ?? "");
    setErrors({});
  }, [enrichment, onNotesChange]);

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
          stadium_wikidata_id: stadiumWikidataId.trim() || null,
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
    stadiumWikidataId,
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

  useImperativeHandle(ref, () => ({
    isDirty,
    canSave,
    isPending,
    handleSave,
    handleReset,
  }), [isDirty, canSave, isPending, handleSave, handleReset]);

  // If no enrichment data at all, show placeholder
  if (!enrichment) {
    return (
      <p className="text-xs text-muted-foreground">
        No Wikidata enrichment available for this team.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {/* Override form */}
      <div className="space-y-3">
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

        {/* Stadium Wikidata ID */}
        <div className="space-y-1">
          <Label htmlFor={`team-${teamId}-stadium-wikidata`} className="text-xs">
            Stadium Wikidata ID
          </Label>
          <Input
            id={`team-${teamId}-stadium-wikidata`}
            type="text"
            placeholder="Q00000"
            value={stadiumWikidataId}
            onChange={(e) => setStadiumWikidataId(e.target.value)}
            disabled={isPending}
            className="h-8 text-sm"
            aria-invalid={!!errors.stadiumWikidataId}
          />
          {errors.stadiumWikidataId && (
            <p className="text-xs text-destructive">{errors.stadiumWikidataId}</p>
          )}
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

      </div>

    </div>
  );
});
