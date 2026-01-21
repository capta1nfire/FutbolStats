"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Calendar,
  TrendingUp,
  Cog,
  AlertTriangle,
  BarChart3,
  Database,
  FileText,
  Settings,
  LucideIcon,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface NavItem {
  icon: LucideIcon;
  label: string;
  href: string;
}

const navItems: NavItem[] = [
  { icon: LayoutDashboard, label: "Overview", href: "/overview" },
  { icon: Calendar, label: "Matches", href: "/matches" },
  { icon: TrendingUp, label: "Predictions", href: "/predictions" },
  { icon: Cog, label: "Jobs", href: "/jobs" },
  { icon: AlertTriangle, label: "Incidents", href: "/incidents" },
  { icon: BarChart3, label: "Analytics", href: "/analytics" },
  { icon: Database, label: "Data Quality", href: "/data-quality" },
  { icon: FileText, label: "Audit", href: "/audit" },
  { icon: Settings, label: "Settings", href: "/settings" },
];

export function IconSidebar() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Main navigation"
      className="w-14 border-r border-border bg-sidebar flex flex-col items-center py-3 gap-1"
    >
      <TooltipProvider delayDuration={0}>
        {navItems.map((item) => {
          const isActive = pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Tooltip key={item.href}>
              <TooltipTrigger asChild>
                <Link
                  href={item.href}
                  aria-label={item.label}
                  aria-current={isActive ? "page" : undefined}
                  className={cn(
                    "w-10 h-10 flex items-center justify-center rounded-lg transition-smooth focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                    isActive
                      ? "bg-sidebar-accent text-sidebar-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-sidebar-accent"
                  )}
                >
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </Link>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>{item.label}</p>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </TooltipProvider>
    </nav>
  );
}
