import { Database } from "lucide-react";

export default function DataQualityPage() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <div className="text-center">
        <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
        <h1 className="text-xl font-semibold text-foreground mb-2">Data Quality</h1>
        <p className="text-muted-foreground">
          Data quality checks and affected items.
        </p>
        <p className="text-sm text-muted-foreground mt-4">Coming soon</p>
      </div>
    </div>
  );
}
