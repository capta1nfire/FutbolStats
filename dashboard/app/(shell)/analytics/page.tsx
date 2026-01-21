import { BarChart3 } from "lucide-react";

export default function AnalyticsPage() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <div className="text-center">
        <BarChart3 className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
        <h1 className="text-xl font-semibold text-foreground mb-2">Analytics</h1>
        <p className="text-muted-foreground">
          Model performance and accuracy metrics.
        </p>
        <p className="text-sm text-muted-foreground mt-4">Coming in Phase 1</p>
      </div>
    </div>
  );
}
