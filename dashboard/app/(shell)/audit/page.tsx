import { FileText } from "lucide-react";

export default function AuditPage() {
  return (
    <div className="h-full flex items-center justify-center bg-background">
      <div className="text-center">
        <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
        <h1 className="text-xl font-semibold text-foreground mb-2">Audit</h1>
        <p className="text-muted-foreground">
          Audit log and event history.
        </p>
        <p className="text-sm text-muted-foreground mt-4">Coming soon</p>
      </div>
    </div>
  );
}
