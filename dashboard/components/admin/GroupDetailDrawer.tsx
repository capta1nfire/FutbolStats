"use client";

import { DetailDrawer } from "@/components/shell/DetailDrawer";
import { Badge } from "@/components/ui/badge";
import { Loader } from "@/components/ui/loader";
import { useAdminLeagueGroup } from "@/lib/hooks";
import type {
  AdminLeagueGroupDetailFull,
  AdminLeagueGroupMemberFull,
  AdminLeagueGroupSeasonStats,
  AdminLeagueGroupMatch,
} from "@/lib/types";

interface GroupDetailDrawerProps {
  groupId: number | null;
  onClose: () => void;
}

export function GroupDetailDrawer({ groupId, onClose }: GroupDetailDrawerProps) {
  const { data, isLoading } = useAdminLeagueGroup(groupId);

  return (
    <DetailDrawer
      open={groupId !== null}
      onClose={onClose}
      title={data?.group?.name ?? "Group Details"}
      variant="overlay"
    >
      {isLoading && (
        <div className="flex items-center justify-center h-48">
          <Loader size="md" />
        </div>
      )}
      {!isLoading && !data && groupId !== null && (
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          Failed to load group
        </div>
      )}
      {data && <GroupDetailContent data={data} />}
    </DetailDrawer>
  );
}

function GroupDetailContent({ data }: { data: AdminLeagueGroupDetailFull }) {
  const { group, member_leagues, is_active_all, is_active_any, stats_by_season, teams_count, recent_matches } = data;

  return (
    <div className="space-y-5 text-sm">
      {/* Header info */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Badge
            variant={is_active_all ? "default" : is_active_any ? "secondary" : "outline"}
            className="text-xs"
          >
            {is_active_all ? "All Active" : is_active_any ? "Partial" : "Inactive"}
          </Badge>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <InfoField label="Key" value={group.group_key} mono />
          <InfoField label="Country" value={group.country} />
          <InfoField label="Teams" value={String(teams_count)} />
        </div>
      </div>

      <hr className="border-border" />

      {/* Member leagues */}
      {member_leagues.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Member Leagues
          </h3>
          <div className="space-y-1.5">
            {member_leagues.map((m) => (
              <MemberLeagueRow key={m.league_id} member={m} />
            ))}
          </div>
        </div>
      )}

      {/* Stats by season */}
      {stats_by_season.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Stats by Season
          </h3>
          <div className="rounded-md border border-border overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-muted/50">
                  <th className="text-left px-2 py-1.5 font-medium">Season</th>
                  <th className="text-right px-2 py-1.5 font-medium">Matches</th>
                  <th className="text-right px-2 py-1.5 font-medium">Finished</th>
                  <th className="text-right px-2 py-1.5 font-medium">Stats%</th>
                </tr>
              </thead>
              <tbody>
                {stats_by_season.map((s) => (
                  <SeasonRow key={s.season} stats={s} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent matches */}
      {recent_matches.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Recent Matches
          </h3>
          <div className="space-y-1">
            {recent_matches.slice(0, 5).map((m) => (
              <MatchRow key={m.match_id} match={m} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MemberLeagueRow({ member }: { member: AdminLeagueGroupMemberFull }) {
  return (
    <div className="flex items-center justify-between">
      <span>{member.name}</span>
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">{member.kind}</span>
        <Badge
          variant={member.is_active ? "default" : "secondary"}
          className="text-[10px] px-1.5"
        >
          {member.is_active ? "Active" : "Inactive"}
        </Badge>
      </div>
    </div>
  );
}

function SeasonRow({ stats }: { stats: AdminLeagueGroupSeasonStats }) {
  return (
    <tr className="border-t border-border">
      <td className="px-2 py-1.5 font-mono">{stats.season}</td>
      <td className="px-2 py-1.5 text-right font-mono">{stats.total_matches}</td>
      <td className="px-2 py-1.5 text-right font-mono">{stats.finished}</td>
      <td className="px-2 py-1.5 text-right font-mono">{stats.with_stats_pct}%</td>
    </tr>
  );
}

function MatchRow({ match }: { match: AdminLeagueGroupMatch }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="truncate flex-1">
        {match.home_team} vs {match.away_team}
      </span>
      <div className="flex items-center gap-2 shrink-0 ml-2">
        {match.score && <span className="font-mono">{match.score}</span>}
        <span className="text-muted-foreground">
          {new Date(match.date).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
}

function InfoField({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`font-medium ${mono ? "font-mono text-xs" : ""}`}>{value}</p>
    </div>
  );
}
