"use client";

import { useState } from "react";
import { DetailDrawer } from "@/components/shell/DetailDrawer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader } from "@/components/ui/loader";
import { useAdminLeague } from "@/lib/hooks";
import { useAdminLeagueMutation } from "@/lib/hooks/use-admin-league-mutation";
import type { AdminLeagueDetailCore } from "@/lib/types";

interface LeagueDetailDrawerProps {
  leagueId: number | null;
  onClose: () => void;
}

const PRIORITY_OPTIONS = ["high", "medium", "low"] as const;
const KIND_OPTIONS = ["league", "cup", "international", "friendly"] as const;

export function LeagueDetailDrawer({ leagueId, onClose }: LeagueDetailDrawerProps) {
  const { data, isLoading } = useAdminLeague(leagueId);

  return (
    <DetailDrawer
      open={leagueId !== null}
      onClose={onClose}
      title={data?.name ?? "League Details"}
      variant="overlay"
    >
      {isLoading && (
        <div className="flex items-center justify-center h-48">
          <Loader size="md" />
        </div>
      )}
      {!isLoading && !data && leagueId !== null && (
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          Failed to load league
        </div>
      )}
      {data && <LeagueForm key={data.league_id} league={data} />}
    </DetailDrawer>
  );
}

function LeagueForm({ league }: { league: AdminLeagueDetailCore }) {
  const mutation = useAdminLeagueMutation();

  // Editable local state
  const [isActive, setIsActive] = useState(league.is_active);
  const [kind, setKind] = useState(league.kind);
  const [priority, setPriority] = useState(league.priority);
  const [matchWeight, setMatchWeight] = useState(
    league.match_weight != null ? String(league.match_weight) : ""
  );
  const [lastAuditId, setLastAuditId] = useState<number | null>(null);

  const isDirty =
    isActive !== league.is_active ||
    kind !== league.kind ||
    priority !== league.priority ||
    matchWeight !== (league.match_weight != null ? String(league.match_weight) : "");

  function handleSave() {
    const body: Record<string, unknown> = {};

    if (isActive !== league.is_active) body.is_active = isActive;
    if (kind !== league.kind) body.kind = kind;
    if (priority !== league.priority) body.priority = priority;
    if (matchWeight !== (league.match_weight != null ? String(league.match_weight) : "")) {
      body.match_weight = matchWeight ? parseFloat(matchWeight) : null;
    }

    if (Object.keys(body).length === 0) return;

    mutation.mutate(
      { id: league.league_id, body },
      {
        onSuccess: (data) => {
          if (data?.audit_id) {
            setLastAuditId(data.audit_id);
          }
        },
      }
    );
  }

  return (
    <div className="space-y-5 text-sm">
      {/* Read-only info */}
      <div className="grid grid-cols-2 gap-3">
        <ReadOnlyField label="ID" value={String(league.league_id)} />
        <ReadOnlyField label="Source" value={league.source} />
        <ReadOnlyField label="Country" value={league.country} />
        <ReadOnlyField label="Match Type" value={league.match_type} />
        <ReadOnlyField label="Observed" value={league.observed ? "Yes" : "No"} />
        <ReadOnlyField label="Configured" value={league.configured ? "Yes" : "No"} />
      </div>

      <hr className="border-border" />

      {/* Editable fields */}
      <div className="space-y-4">
        {/* is_active toggle */}
        <div className="flex items-center justify-between">
          <Label htmlFor="is-active" className="text-sm">Active</Label>
          <Switch
            id="is-active"
            checked={isActive}
            onCheckedChange={setIsActive}
          />
        </div>

        {/* kind */}
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Kind</Label>
          <Select value={kind} onValueChange={setKind}>
            <SelectTrigger className="h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {KIND_OPTIONS.map((k) => (
                <SelectItem key={k} value={k}>{k}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* priority */}
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Priority</Label>
          <Select value={priority} onValueChange={setPriority}>
            <SelectTrigger className="h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PRIORITY_OPTIONS.map((p) => (
                <SelectItem key={p} value={p}>{p}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* match_weight */}
        <div className="space-y-1">
          <Label htmlFor="match-weight" className="text-xs text-muted-foreground">
            Match Weight
          </Label>
          <Input
            id="match-weight"
            type="number"
            step="0.1"
            value={matchWeight}
            onChange={(e) => setMatchWeight(e.target.value)}
            placeholder="null"
            className="h-8 text-sm"
          />
        </div>
      </div>

      {/* Save button */}
      <div className="flex items-center gap-3">
        <Button
          onClick={handleSave}
          disabled={!isDirty || mutation.isPending}
          size="sm"
        >
          {mutation.isPending ? "Saving…" : "Save"}
        </Button>
        {mutation.isSuccess && (
          <span className="text-xs text-green-600">
            Saved{lastAuditId ? ` (audit #${lastAuditId})` : ""}
          </span>
        )}
        {mutation.isError && (
          <span className="text-xs text-destructive">
            Error: {(mutation.error as Error)?.message ?? "Save failed"}
          </span>
        )}
      </div>
    </div>
  );
}

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium">{value}</p>
    </div>
  );
}
