"use client";

import { useMemo } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { SettingsUser, USER_ROLE_LABELS } from "@/lib/types";
import { DataTable } from "@/components/tables";
import { Badge } from "@/components/ui/badge";
import { formatDistanceToNow } from "@/lib/utils";
import { User, Shield, Eye } from "lucide-react";

interface UsersTableProps {
  data: SettingsUser[];
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
}

export function UsersTable({
  data,
  isLoading,
  error,
  onRetry,
}: UsersTableProps) {
  const columns = useMemo<ColumnDef<SettingsUser>[]>(
    () => [
      {
        accessorKey: "email",
        header: "User",
        cell: ({ row }) => (
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-surface flex items-center justify-center">
              <User className="h-4 w-4 text-muted-foreground" />
            </div>
            <span className="text-sm text-foreground">{row.original.email}</span>
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "role",
        header: "Role",
        cell: ({ row }) => {
          const isAdmin = row.original.role === "admin";
          return (
            <div className="flex items-center gap-1.5">
              {isAdmin ? (
                <Shield className="h-3.5 w-3.5 text-primary" />
              ) : (
                <Eye className="h-3.5 w-3.5 text-muted-foreground" />
              )}
              <Badge
                variant="outline"
                className={
                  isAdmin
                    ? "bg-accent/10 text-accent border-accent/20"
                    : "text-muted-foreground"
                }
              >
                {USER_ROLE_LABELS[row.original.role]}
              </Badge>
            </div>
          );
        },
        enableSorting: true,
      },
      {
        accessorKey: "lastLogin",
        header: "Last Login",
        cell: ({ row }) =>
          row.original.lastLogin ? (
            <span className="text-sm text-muted-foreground">
              {formatDistanceToNow(row.original.lastLogin)}
            </span>
          ) : (
            <span className="text-muted-foreground">Never</span>
          ),
        enableSorting: true,
      },
      {
        accessorKey: "createdAt",
        header: "Created",
        cell: ({ row }) =>
          row.original.createdAt ? (
            <span className="text-sm text-muted-foreground">
              {formatDistanceToNow(row.original.createdAt)}
            </span>
          ) : (
            <span className="text-muted-foreground">-</span>
          ),
        enableSorting: true,
      },
    ],
    []
  );

  return (
    <DataTable
      columns={columns}
      data={data}
      isLoading={isLoading}
      error={error}
      onRetry={onRetry}
      getRowId={(row) => String(row.id)}
      emptyMessage="No users configured"
    />
  );
}
