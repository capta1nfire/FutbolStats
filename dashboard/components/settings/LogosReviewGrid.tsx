"use client";

import { useState } from "react";
import { useLogosReview, useReviewTeamLogo, useApproveLeague } from "@/lib/hooks";
import {
  TeamLogoReview,
  REVIEW_STATUS_LABELS,
} from "@/lib/types/logos";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Loader2,
  AlertTriangle,
  Check,
  X,
  RefreshCw,
  CheckCircle,
  XCircle,
  Shield,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import Image from "next/image";

interface LogosReviewGridProps {
  leagueId: number;
}

function LogoPreview({
  src,
  fallbackSrc,
  label,
  size = 64,
}: {
  src?: string;
  fallbackSrc?: string;
  label: string;
  size?: number;
}) {
  const [hasError, setHasError] = useState(false);
  const displaySrc = hasError ? fallbackSrc : src;

  if (!displaySrc) {
    return (
      <div
        className="flex items-center justify-center rounded bg-muted/50"
        style={{ width: size, height: size }}
        title={label}
      >
        <Shield className="h-6 w-6 text-muted-foreground/50" />
      </div>
    );
  }

  return (
    <div className="relative group">
      <Image
        src={displaySrc}
        alt={label}
        width={size}
        height={size}
        className="object-contain rounded"
        onError={() => setHasError(true)}
        unoptimized
      />
      <span className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[10px] text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
        {label}
      </span>
    </div>
  );
}

function TeamCard({
  team,
  onReview,
  isReviewing,
}: {
  team: TeamLogoReview;
  onReview: (action: "approve" | "reject" | "regenerate") => void;
  isReviewing: boolean;
}) {
  const isPending = team.reviewStatus === "pending";
  const isApproved = team.reviewStatus === "approved";
  const isRejected = team.reviewStatus === "rejected";
  const needsRegeneration = team.reviewStatus === "needs_regeneration";
  const hasError = team.status === "error";
  const showRegenerate = hasError || needsRegeneration;

  return (
    <div
      className={cn(
        "bg-surface rounded-lg p-3 border",
        isApproved && "border-[var(--status-success-border)]",
        isRejected && "border-[var(--status-error-border)]",
        (hasError || needsRegeneration) && "border-[var(--status-warning-border)]",
        isPending && "border-border"
      )}
    >
      {/* Team Name */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium truncate" title={team.teamName}>
          {team.teamName}
        </span>
        <Badge
          variant="outline"
          className={cn(
            "text-xs",
            isApproved && "bg-[var(--status-success-bg)] text-[var(--status-success-text)]",
            isRejected && "bg-[var(--status-error-bg)] text-[var(--status-error-text)]",
            hasError && "bg-[var(--status-error-bg)] text-[var(--status-error-text)]",
            needsRegeneration && "bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]"
          )}
        >
          {hasError ? "Error" : REVIEW_STATUS_LABELS[team.reviewStatus]}
        </Badge>
      </div>

      {/* Logo Previews */}
      <div className="flex items-center justify-center gap-3 mb-3 min-h-[80px]">
        <LogoPreview
          src={team.thumbnails?.front?.[64] || team.urls.front}
          fallbackSrc={team.fallbackUrl}
          label="Front"
        />
        <LogoPreview
          src={team.thumbnails?.right?.[64] || team.urls.right}
          fallbackSrc={team.fallbackUrl}
          label="Right"
        />
        <LogoPreview
          src={team.thumbnails?.left?.[64] || team.urls.left}
          fallbackSrc={team.fallbackUrl}
          label="Left"
        />
      </div>

      {/* Error Message */}
      {hasError && team.errorMessage && (
        <p className="text-xs text-[var(--status-error-text)] mb-2 truncate" title={team.errorMessage}>
          {team.errorMessage}
        </p>
      )}

      {/* Actions */}
      {isPending && !hasError && (
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 h-7 text-xs"
            onClick={() => onReview("approve")}
            disabled={isReviewing}
          >
            {isReviewing ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <>
                <Check className="h-3 w-3 mr-1" />
                Approve
              </>
            )}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs text-[var(--status-error-text)]"
            onClick={() => onReview("reject")}
            disabled={isReviewing}
          >
            <X className="h-3 w-3" />
          </Button>
        </div>
      )}

      {/* Regenerate for errors or needs_regeneration */}
      {showRegenerate && (
        <Button
          variant="outline"
          size="sm"
          className="w-full h-7 text-xs"
          onClick={() => onReview("regenerate")}
          disabled={isReviewing}
        >
          {isReviewing ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <>
              <RefreshCw className="h-3 w-3 mr-1" />
              {hasError ? "Retry" : "Regenerate"}
            </>
          )}
        </Button>
      )}
    </div>
  );
}

export function LogosReviewGrid({ leagueId }: LogosReviewGridProps) {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [reviewingTeamId, setReviewingTeamId] = useState<number | null>(null);

  const filterValue = statusFilter === "all" ? undefined : statusFilter;
  const { data, isLoading, error, refetch } = useLogosReview(leagueId, filterValue);
  const reviewTeam = useReviewTeamLogo();
  const approveLeague = useApproveLeague();

  const handleReview = async (
    teamId: number,
    action: "approve" | "reject" | "regenerate"
  ) => {
    setReviewingTeamId(teamId);
    try {
      await reviewTeam.mutateAsync({ teamId, request: { action } });
      toast.success(`Team ${action}d`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : `Failed to ${action}`);
    } finally {
      setReviewingTeamId(null);
    }
  };

  const handleBulkApprove = async () => {
    try {
      const result = await approveLeague.mutateAsync({
        leagueId,
        request: { action: "approve_all" },
      });
      toast.success(`Approved ${result.updated_count} teams`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to approve");
    }
  };

  const handleBulkReject = async () => {
    try {
      const result = await approveLeague.mutateAsync({
        leagueId,
        request: { action: "reject_all" },
      });
      toast.success(`Rejected ${result.updated_count} teams`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to reject");
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center py-8 gap-4">
        <AlertTriangle className="h-8 w-8 text-[var(--status-error-text)]" />
        <p className="text-sm text-muted-foreground">Failed to load teams</p>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  const teams = data.teams;
  const pendingCount = teams.filter((t) => t.reviewStatus === "pending").length;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">
            {data.total} teams
          </span>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-[140px] h-8">
              <SelectValue placeholder="Filter" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="approved">Approved</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
              <SelectItem value="needs_regeneration">Needs Regen</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Bulk Actions */}
        {pendingCount > 0 && (
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleBulkApprove}
              disabled={approveLeague.isPending}
            >
              {approveLeague.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 mr-1" />
                  Approve All ({pendingCount})
                </>
              )}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleBulkReject}
              disabled={approveLeague.isPending}
              className="text-[var(--status-error-text)]"
            >
              <XCircle className="h-4 w-4 mr-1" />
              Reject All
            </Button>
          </div>
        )}
      </div>

      {/* Grid */}
      {teams.length === 0 ? (
        <div className="text-center py-8 text-sm text-muted-foreground">
          No teams found with this filter
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {teams.map((team) => (
            <TeamCard
              key={team.teamId}
              team={team}
              onReview={(action) => handleReview(team.teamId, action)}
              isReviewing={reviewingTeamId === team.teamId}
            />
          ))}
        </div>
      )}
    </div>
  );
}
