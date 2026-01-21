import { LayoutDashboard } from "lucide-react";

export default function OverviewPage() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <div className="text-center">
        <LayoutDashboard className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
        <h1 className="text-xl font-semibold text-foreground mb-2">Overview</h1>
        <p className="text-muted-foreground">
          Health cards, coverage metrics, and system status.
        </p>
        <p className="text-sm text-muted-foreground mt-4">Coming soon</p>
      </div>
    </div>
  );
}
