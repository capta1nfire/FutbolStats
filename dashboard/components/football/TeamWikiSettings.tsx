"use client";

import { useState, useEffect, useCallback } from "react";
import { useTeamWikiMutation } from "@/lib/hooks";
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
  Copy,
} from "lucide-react";
import { toast } from "sonner";
import type { TeamWikiInfo } from "@/lib/types/football";

// =============================================================================
// Validation
// =============================================================================

// Soporta idiomas como en, es, pt-br, simple, zh-yue, etc.
const WIKI_URL_REGEX = /^https:\/\/[a-z]{2,}(-[a-z]+)?\.wikipedia\.org\/wiki\/.+$/;
const WIKIDATA_URL_REGEX = /^https:\/\/www\.wikidata\.org\/wiki\/(Q\d+)$/;
const WIKIDATA_ID_REGEX = /^Q\d+$/;

interface ValidationResult {
  isValid: boolean;
  error?: string;
  extractedQId?: string; // If wikidata URL was pasted in wiki_url field
}

function validateWikiUrl(url: string): ValidationResult {
  if (!url.trim()) {
    return { isValid: true }; // Optional field
  }

  const trimmed = url.trim();

  // Check if user pasted a wikidata URL - extract Q-id
  const wikidataMatch = trimmed.match(WIKIDATA_URL_REGEX);
  if (wikidataMatch) {
    return {
      isValid: true,
      extractedQId: wikidataMatch[1],
    };
  }

  // Reject mobile URLs
  if (trimmed.includes("m.wikipedia.org")) {
    return { isValid: false, error: "No usar URLs mobile (m.wikipedia.org)" };
  }

  // Reject Special pages
  if (trimmed.includes("/Special:")) {
    return { isValid: false, error: "No usar páginas especiales (Special:)" };
  }

  // Reject querystring/fragment
  if (trimmed.includes("?") || trimmed.includes("#")) {
    return { isValid: false, error: "No incluir parámetros (?...) ni fragmentos (#...)" };
  }

  // Validate Wikipedia URL format
  if (!WIKI_URL_REGEX.test(trimmed)) {
    return { isValid: false, error: "Formato: https://XX.wikipedia.org/wiki/..." };
  }

  return { isValid: true };
}

function validateWikidataId(id: string): ValidationResult {
  if (!id.trim()) {
    return { isValid: true }; // Optional field
  }

  const normalized = id.trim().toUpperCase();

  if (!WIKIDATA_ID_REGEX.test(normalized)) {
    return { isValid: false, error: "Formato: Q seguido de números (ej: Q42)" };
  }

  return { isValid: true };
}

// =============================================================================
// Component
// =============================================================================

interface TeamWikiSettingsProps {
  teamId: number;
  teamName: string;
  wiki?: TeamWikiInfo;
}

export function TeamWikiSettings({ teamId, teamName, wiki }: TeamWikiSettingsProps) {
  // Form state
  const [wikiUrl, setWikiUrl] = useState(wiki?.wiki_url ?? "");
  const [wikidataId, setWikidataId] = useState(wiki?.wikidata_id ?? "");

  // Validation state
  const [urlError, setUrlError] = useState<string | null>(null);
  const [idError, setIdError] = useState<string | null>(null);

  // Mutation
  const mutation = useTeamWikiMutation();

  // Sync form state when wiki prop changes (e.g., after refresh)
  useEffect(() => {
    setWikiUrl(wiki?.wiki_url ?? "");
    setWikidataId(wiki?.wikidata_id ?? "");
  }, [wiki?.wiki_url, wiki?.wikidata_id]);

  // Check if form has changes
  const isDirty =
    wikiUrl.trim() !== (wiki?.wiki_url ?? "") ||
    wikidataId.trim().toUpperCase() !== (wiki?.wikidata_id ?? "").toUpperCase();

  // Handle wiki_url change with wikidata URL detection
  const handleWikiUrlChange = useCallback((value: string) => {
    setWikiUrl(value);

    const result = validateWikiUrl(value);
    setUrlError(result.error ?? null);

    // If user pasted a wikidata URL, extract Q-id and move it
    if (result.extractedQId) {
      setWikidataId(result.extractedQId);
      setWikiUrl(""); // Clear the URL field
      toast.info(`Q-id extraído: ${result.extractedQId}`);
    }
  }, []);

  // Handle wikidata_id change
  const handleWikidataIdChange = useCallback((value: string) => {
    setWikidataId(value.toUpperCase());
    const result = validateWikidataId(value);
    setIdError(result.error ?? null);
  }, []);

  // Handle reset (cancel changes)
  const handleReset = useCallback(() => {
    setWikiUrl(wiki?.wiki_url ?? "");
    setWikidataId(wiki?.wikidata_id ?? "");
    setUrlError(null);
    setIdError(null);
  }, [wiki?.wiki_url, wiki?.wikidata_id]);

  // Handle save
  const handleSave = useCallback(() => {
    // Final validation
    const urlResult = validateWikiUrl(wikiUrl);
    const idResult = validateWikidataId(wikidataId);

    if (!urlResult.isValid || !idResult.isValid) {
      if (!urlResult.isValid) setUrlError(urlResult.error ?? null);
      if (!idResult.isValid) setIdError(idResult.error ?? null);
      return;
    }

    mutation.mutate(
      {
        teamId,
        data: {
          wiki_url: wikiUrl.trim() || null,
          wikidata_id: wikidataId.trim().toUpperCase() || null,
        },
      },
      {
        onSuccess: () => {
          toast.success("Wikipedia actualizado");
        },
        onError: (error) => {
          // Handle backend not supporting wiki fields (ATI condition #1)
          if ((error as Error & { isNotSupported?: boolean }).isNotSupported) {
            toast.info("Backend aún no soporta campos wiki");
          } else {
            toast.error(error.message);
          }
        },
      }
    );
  }, [teamId, wikiUrl, wikidataId, mutation]);

  const hasValidationErrors = !!urlError || !!idError;
  const canSave = isDirty && !hasValidationErrors && !mutation.isPending;
  const wikiUrlInputId = `team-${teamId}-wiki-url`;
  const wikidataIdInputId = `team-${teamId}-wikidata-id`;

  return (
    <SurfaceCard className="space-y-4">
      {/* Header */}
      <h4 className="text-sm font-medium">Wikipedia</h4>

      {/* Form */}
      <div className="space-y-4">
        {/* Wiki URL Input */}
        <div className="space-y-1.5">
          <Label htmlFor={wikiUrlInputId}>Wikipedia URL</Label>
          <div className="relative">
            <Input
              id={wikiUrlInputId}
              type="url"
              placeholder="https://xx.wikipedia.org/wiki/..."
              value={wikiUrl}
              onChange={(e) => handleWikiUrlChange(e.target.value)}
              aria-invalid={!!urlError}
              className="pr-9"
              disabled={mutation.isPending}
            />
            {wikiUrl && !urlError && WIKI_URL_REGEX.test(wikiUrl.trim()) && (
              <Button
                variant="actionLink"
                size="icon-sm"
                className="absolute right-2 top-1/2 -translate-y-1/2"
                asChild
              >
                <a
                  href={wikiUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Abrir en Wikipedia"
                >
                  <ExternalLink className="h-4 w-4" />
                </a>
              </Button>
            )}
          </div>
          {urlError && (
            <p className="text-xs text-destructive">{urlError}</p>
          )}
        </div>

        {/* Wikidata ID Input */}
        <div className="space-y-1.5">
          <Label htmlFor={wikidataIdInputId}>
            Wikidata ID <span className="opacity-50">(optional)</span>
          </Label>
          <div className="relative">
            <Input
              id={wikidataIdInputId}
              type="text"
              placeholder="Q00000"
              value={wikidataId}
              onChange={(e) => handleWikidataIdChange(e.target.value)}
              aria-invalid={!!idError}
              className="pr-9"
              disabled={mutation.isPending}
            />
            {wikidataId && !idError && (
              <Button
                type="button"
                variant="actionLink"
                size="icon-sm"
                className="absolute right-2 top-1/2 -translate-y-1/2"
                onClick={() => {
                  navigator.clipboard.writeText(wikidataId);
                  toast.success("Wikidata ID copied");
                }}
                aria-label="Copy Wikidata ID"
              >
                <Copy className="h-4 w-4" />
              </Button>
            )}
          </div>
          {idError && (
            <p className="text-xs text-destructive">{idError}</p>
          )}
        </div>

        {/* Action Buttons - only show when dirty */}
        {isDirty && (
          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={handleReset}
              disabled={mutation.isPending}
              className="text-sm px-3 py-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-[color:var(--field-bg-hover)] transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!canSave}
            >
              {mutation.isPending ? (
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

      {/* Derived Fields (Read-only) */}
      {wiki && (wiki.wiki_title || wiki.wiki_confidence != null) && (
        <div className="bg-background/20 rounded-lg p-3 border border-border space-y-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle className="h-3 w-3" />
            <span>Derived data</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            {wiki.wiki_title && (
              <div>
                <span className="text-muted-foreground">Title:</span>{" "}
                <span className="text-foreground">{wiki.wiki_title}</span>
              </div>
            )}
            {wiki.wiki_lang && (
              <div>
                <span className="text-muted-foreground">Language:</span>{" "}
                <span className="text-foreground">{wiki.wiki_lang}</span>
              </div>
            )}
            {wiki.wiki_source && (
              <div>
                <span className="text-muted-foreground">Source:</span>{" "}
                <span className="text-foreground">{wiki.wiki_source}</span>
              </div>
            )}
            {wiki.wiki_confidence != null && (
              <div>
                <span className="text-muted-foreground">Confidence:</span>{" "}
                <span className="text-foreground">
                  {(wiki.wiki_confidence * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>

          {wiki.wiki_url_cached && (
            <Button variant="link" size="sm" className="px-0 h-auto text-xs" asChild>
              <a
                href={wiki.wiki_url_cached}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="h-3 w-3" />
                View on Wikipedia
              </a>
            </Button>
          )}
        </div>
      )}

      {/* Low Confidence Warning */}
      {wiki?.wiki_confidence != null && wiki.wiki_confidence < 0.5 && (
        <div className="flex items-start gap-2 p-3 bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)] rounded-lg">
          <AlertTriangle className="h-4 w-4 text-[var(--status-warning-text)] mt-0.5" />
          <div className="text-xs">
            <p className="font-medium text-[var(--status-warning-text)]">Needs manual review</p>
            <p className="text-muted-foreground">
              Match confidence is low ({((wiki.wiki_confidence ?? 0) * 100).toFixed(0)}%).
              Verify that the article corresponds to {teamName}.
            </p>
          </div>
        </div>
      )}
    </SurfaceCard>
  );
}
