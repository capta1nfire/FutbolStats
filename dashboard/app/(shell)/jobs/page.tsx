import { Cog } from "lucide-react";

export default function JobsPage() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <div className="text-center">
        <Cog className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
        <h1 className="text-xl font-semibold text-foreground mb-2">Jobs</h1>
        <p className="text-muted-foreground">
          Scheduler status and job run history.
        </p>
        <p className="text-sm text-muted-foreground mt-4">Coming in Phase 0</p>
      </div>
    </div>
  );
}
