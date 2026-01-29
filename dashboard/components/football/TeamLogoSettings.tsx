"use client";

import { useState, useCallback } from "react";
import Image from "next/image";
import { useTeamLogoStatus, useUploadTeamLogo } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Upload,
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ImageIcon,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface TeamLogoSettingsProps {
  teamId: number;
  teamName: string;
  fallbackLogoUrl?: string;
}

const STATUS_LABELS: Record<string, string> = {
  pending: "Pending Upload",
  queued: "Queued",
  processing: "Processing",
  pending_resize: "Resizing",
  ready: "Ready",
  error: "Error",
  paused: "Paused",
};

const REVIEW_STATUS_LABELS: Record<string, string> = {
  pending: "Pending Review",
  approved: "Approved",
  rejected: "Rejected",
  needs_regeneration: "Needs Regen",
};

/**
 * Logo Preview with variants
 */
function LogoVariantPreview({
  label,
  url,
  fallbackUrl,
}: {
  label: string;
  url?: string;
  fallbackUrl?: string;
}) {
  const [hasError, setHasError] = useState(false);
  const displayUrl = hasError ? fallbackUrl : url;

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="w-16 h-16 rounded-lg bg-muted/50 flex items-center justify-center overflow-hidden">
        {displayUrl ? (
          <Image
            src={displayUrl}
            alt={label}
            width={64}
            height={64}
            className="object-contain"
            onError={() => setHasError(true)}
            unoptimized
          />
        ) : (
          <ImageIcon className="h-6 w-6 text-muted-foreground/50" />
        )}
      </div>
      <span className="text-[10px] text-muted-foreground">{label}</span>
    </div>
  );
}

/**
 * Upload Drop Zone
 */
function UploadDropZone({
  onUpload,
  isUploading,
  disabled,
}: {
  onUpload: (file: File) => void;
  isUploading: boolean;
  disabled?: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onUpload(file);
      } else {
        toast.error("Please drop an image file (PNG, WebP, or SVG)");
      }
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onUpload(file);
      }
    },
    [onUpload]
  );

  return (
    <div
      className={cn(
        "border-2 border-dashed rounded-lg p-4 transition-colors",
        isDragging
          ? "border-primary bg-primary/5"
          : "border-border hover:border-muted-foreground/50",
        disabled && "opacity-50 pointer-events-none"
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center gap-2 text-center">
        {isUploading ? (
          <>
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <p className="text-xs text-muted-foreground">Uploading...</p>
          </>
        ) : (
          <>
            <Upload className="h-6 w-6 text-muted-foreground" />
            <p className="text-xs text-muted-foreground">
              Drop high-quality logo here
            </p>
            <p className="text-[10px] text-muted-foreground/70">
              PNG, WebP, SVG (min 512x512, max 5MB)
            </p>
            <label>
              <input
                type="file"
                accept="image/png,image/webp,image/svg+xml"
                onChange={handleFileSelect}
                className="hidden"
                disabled={disabled || isUploading}
              />
              <Button
                variant="outline"
                size="sm"
                className="mt-1 text-xs"
                asChild
              >
                <span>Select File</span>
              </Button>
            </label>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Team Logo Settings Component
 *
 * For TeamDrawer Settings tab - allows:
 * - View current logo status
 * - Upload high-quality original
 * - See 3D variants if generated
 */
export function TeamLogoSettings({
  teamId,
  teamName,
  fallbackLogoUrl,
}: TeamLogoSettingsProps) {
  const { data: logoStatus, isLoading, error, refetch } = useTeamLogoStatus(teamId);
  const uploadMutation = useUploadTeamLogo();

  const handleUpload = useCallback(
    async (file: File) => {
      try {
        const result = await uploadMutation.mutateAsync({ teamId, file });
        toast.success(
          `Logo uploaded: ${result.validation.width}x${result.validation.height}`
        );
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Upload failed");
      }
    },
    [teamId, uploadMutation]
  );

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-6">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // No logo record - show upload prompt
  if (!logoStatus || error) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium">3D Logo</h4>
          <Badge variant="outline" className="text-xs">
            Not Configured
          </Badge>
        </div>

        <p className="text-xs text-muted-foreground">
          Upload a high-quality logo to generate 3D variants for matchup displays.
        </p>

        <UploadDropZone
          onUpload={handleUpload}
          isUploading={uploadMutation.isPending}
        />

        {fallbackLogoUrl && (
          <div className="text-xs text-muted-foreground text-center">
            <span>Current fallback: </span>
            <span className="text-foreground">API-Football</span>
          </div>
        )}
      </div>
    );
  }

  // Has logo record - show status and variants
  const hasOriginal = !!logoStatus.r2Keys.original;
  const hasVariants = logoStatus.status === "ready";
  const isProcessing = ["queued", "processing", "pending_resize"].includes(
    logoStatus.status
  );
  const hasError = logoStatus.status === "error";

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium">3D Logo</h4>
        <Badge
          variant="outline"
          className={cn(
            "text-xs",
            hasVariants && "bg-[var(--status-success-bg)] text-[var(--status-success-text)]",
            isProcessing && "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]",
            hasError && "bg-[var(--status-error-bg)] text-[var(--status-error-text)]"
          )}
        >
          {STATUS_LABELS[logoStatus.status] || logoStatus.status}
        </Badge>
      </div>

      {/* Error message */}
      {hasError && logoStatus.error?.message && (
        <div className="flex items-start gap-2 p-2 rounded-lg bg-[var(--status-error-bg)] border border-[var(--status-error-border)]">
          <AlertTriangle className="h-4 w-4 text-[var(--status-error-text)] shrink-0 mt-0.5" />
          <p className="text-xs text-[var(--status-error-text)]">
            {logoStatus.error.message}
          </p>
        </div>
      )}

      {/* Logo Variants Preview */}
      {(hasOriginal || hasVariants) && (
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="flex items-center justify-center gap-4">
            <LogoVariantPreview
              label="Front"
              url={logoStatus.urls.front}
              fallbackUrl={fallbackLogoUrl}
            />
            <LogoVariantPreview
              label="Right"
              url={logoStatus.urls.right}
              fallbackUrl={fallbackLogoUrl}
            />
            <LogoVariantPreview
              label="Left"
              url={logoStatus.urls.left}
              fallbackUrl={fallbackLogoUrl}
            />
          </div>

          {/* Review Status */}
          {hasVariants && (
            <div className="mt-3 pt-2 border-t border-border/50 flex items-center justify-center gap-2">
              {logoStatus.reviewStatus === "approved" && (
                <CheckCircle className="h-3 w-3 text-[var(--status-success-text)]" />
              )}
              {logoStatus.reviewStatus === "rejected" && (
                <XCircle className="h-3 w-3 text-[var(--status-error-text)]" />
              )}
              <span className="text-xs text-muted-foreground">
                {REVIEW_STATUS_LABELS[logoStatus.reviewStatus] ||
                  logoStatus.reviewStatus}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Processing indicator */}
      {isProcessing && (
        <div className="flex items-center gap-2 p-2 rounded-lg bg-[var(--status-warning-bg)] border border-[var(--status-warning-border)]">
          <Loader2 className="h-4 w-4 animate-spin text-[var(--status-warning-text)]" />
          <p className="text-xs text-[var(--status-warning-text)]">
            Generating 3D variants...
          </p>
        </div>
      )}

      {/* Upload Section */}
      {!isProcessing && (
        <>
          <div className="text-xs text-muted-foreground">
            {hasOriginal
              ? "Upload a new logo to replace the current original."
              : "Upload a high-quality logo to generate 3D variants."}
          </div>
          <UploadDropZone
            onUpload={handleUpload}
            isUploading={uploadMutation.isPending}
            disabled={isProcessing}
          />
        </>
      )}

      {/* Generation info */}
      {logoStatus.generation.iaModel && (
        <div className="text-xs text-muted-foreground text-center space-y-0.5">
          <p>
            Model: {logoStatus.generation.iaModel} | Mode:{" "}
            {logoStatus.generation.mode || "full_3d"}
          </p>
          {logoStatus.generation.costUsd !== undefined && (
            <p>Cost: ${logoStatus.generation.costUsd.toFixed(4)}</p>
          )}
        </div>
      )}

      {/* Refresh button */}
      <Button
        variant="ghost"
        size="sm"
        className="w-full text-xs"
        onClick={() => refetch()}
      >
        <RefreshCw className="h-3 w-3 mr-1" />
        Refresh Status
      </Button>
    </div>
  );
}
