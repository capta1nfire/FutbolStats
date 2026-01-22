"use client";

import { useState } from "react";
import { useUsers } from "@/lib/hooks";
import { SettingsSectionHeader } from "./SettingsSectionHeader";
import { UsersTable } from "./UsersTable";
import { SearchInput } from "@/components/ui/search-input";
import { Users, Shield, Eye } from "lucide-react";

export function UsersSection() {
  const [searchValue, setSearchValue] = useState("");

  const {
    data: users = [],
    isLoading,
    error,
    refetch,
  } = useUsers(searchValue ? { search: searchValue } : undefined);

  const adminCount = users.filter((u) => u.role === "admin").length;
  const readonlyCount = users.filter((u) => u.role === "readonly").length;

  return (
    <div>
      <SettingsSectionHeader
        title="Users & Permissions"
        description="Manage user access and roles"
      />

      <div className="space-y-4">
        {/* Summary */}
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">
              {users.length} total users
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            <span className="text-muted-foreground">
              {adminCount} admins
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">
              {readonlyCount} read-only
            </span>
          </div>
        </div>

        {/* Search */}
        <div className="max-w-sm">
          <SearchInput
            placeholder="Search users..."
            value={searchValue}
            onChange={setSearchValue}
            className="bg-background"
          />
        </div>

        {/* Table */}
        <UsersTable
          data={users}
          isLoading={isLoading}
          error={error}
          onRetry={() => refetch()}
        />

        {/* Phase 0 Notice */}
        <div className="bg-surface/50 rounded-lg p-4 border border-border">
          <p className="text-sm text-muted-foreground">
            User management is read-only in Phase 0. Users are configured
            through environment variables. Contact an administrator to modify
            access.
          </p>
        </div>
      </div>
    </div>
  );
}
