"use client";

import { useState } from "react";
import { PredictionDetail, PREDICTION_STATUS_LABELS, MATCH_RESULT_LABELS } from "@/lib/types";
import { useIsDesktop } from "@/lib/hooks";
import { DetailDrawer } from "@/components/shell";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { IconTabs } from "@/components/ui/icon-tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PredictionStatusBadge } from "./PredictionStatusBadge";
import { ModelBadge } from "./ModelBadge";
import { PickBadge } from "./PickBadge";
import {
  Clock,
  TrendingUp,
  Cpu,
  BarChart3,
  History,
  CheckCircle,
  XCircle,
  Info,
} from "lucide-react";

/** Tab definitions for prediction detail drawer */
const PREDICTION_TABS = [
  { id: "details", icon: <Info />, label: "Details" },
  { id: "model", icon: <Cpu />, label: "Model Info" },
  { id: "evaluation", icon: <History />, label: "Evaluation" },
];

interface PredictionDetailDrawerProps {
  prediction: PredictionDetail | null;
  open: boolean;
  onClose: () => void;
  isLoading?: boolean;
}

/**
 * Prediction Tab Content - content only, without tabs (for desktop drawer with fixedContent)
 */
function PredictionTabContent({ prediction, activeTab }: { prediction: PredictionDetail; activeTab: string }) {
  const kickoffDate = new Date(prediction.kickoffISO);
  const formattedKickoff = kickoffDate.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const isCorrect =
    prediction.status === "evaluated" && prediction.result && prediction.pick
      ? prediction.pick === prediction.result
      : undefined;

  return (
    <div className="w-full">
      {/* Details Tab */}
      {activeTab === "details" && (
        <div className="bg-surface rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <PredictionStatusBadge status={prediction.status} />
            <ModelBadge model={prediction.model} />
          </div>

          {/* Match Info */}
          <div>
            <h3 className="text-sm font-medium text-foreground mb-2">
              {prediction.matchLabel}
            </h3>
            <p className="text-xs text-muted-foreground">
              {prediction.leagueName}
            </p>
          </div>

          {/* Metadata */}
          <div className="space-y-3 pt-3 border-t border-border">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Kickoff:</span>
              <span className="text-foreground">{formattedKickoff}</span>
            </div>

            {prediction.generatedAt && (
              <div className="flex items-center gap-2 text-sm">
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Generated:</span>
                <span className="text-foreground">
                  {new Date(prediction.generatedAt).toLocaleString()}
                </span>
              </div>
            )}
          </div>

          {/* Probabilities */}
          {prediction.probs && (
            <div className="pt-3 border-t border-border">
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Probabilities</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-background rounded p-2">
                  <div className="text-lg font-bold text-foreground">
                    {(prediction.probs.home * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Home</div>
                </div>
                <div className="bg-background rounded p-2">
                  <div className="text-lg font-bold text-foreground">
                    {(prediction.probs.draw * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Draw</div>
                </div>
                <div className="bg-background rounded p-2">
                  <div className="text-lg font-bold text-foreground">
                    {(prediction.probs.away * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-muted-foreground">Away</div>
                </div>
              </div>
            </div>
          )}

          {/* Pick */}
          {prediction.pick && (
            <div className="pt-3 border-t border-border">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Prediction Pick:</span>
                <PickBadge pick={prediction.pick} isCorrect={isCorrect} />
              </div>
              {prediction.status === "evaluated" && prediction.result && (
                <div className="flex items-center justify-between mt-2">
                  <span className="text-sm text-muted-foreground">Actual Result:</span>
                  <span className="text-sm text-foreground">
                    {MATCH_RESULT_LABELS[prediction.result]}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Phase 0 notice */}
          <p className="text-xs text-muted-foreground text-center italic pt-4">
            Data from mocks - Phase 0
          </p>
        </div>
      )}

      {/* Model Info Tab */}
      {activeTab === "model" && (
        <div className="bg-surface rounded-lg p-4">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Model Features</span>
          </div>

          {prediction.featuresTop && prediction.featuresTop.length > 0 ? (
            <div className="space-y-2">
              {prediction.featuresTop.map((feature, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-sm py-1 border-b border-border last:border-0"
                >
                  <span className="text-muted-foreground">{feature.name}</span>
                  <span className="text-foreground font-mono">{feature.value}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Cpu className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">No feature data available</p>
            </div>
          )}
        </div>
      )}

      {/* Evaluation Tab */}
      {activeTab === "evaluation" && (
        <div className="bg-surface rounded-lg p-4">
          {prediction.status === "evaluated" && prediction.evaluation ? (
            <div className="space-y-4">
              {/* Evaluation Result */}
              <div className="flex items-center justify-center gap-2">
                {isCorrect ? (
                  <CheckCircle className="h-6 w-6 text-success" />
                ) : (
                  <XCircle className="h-6 w-6 text-error" />
                )}
                <span
                  className={`text-lg font-medium ${
                    isCorrect ? "text-success" : "text-error"
                  }`}
                >
                  {isCorrect ? "Correct Prediction" : "Incorrect Prediction"}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {prediction.evaluation.accuracy !== undefined && (
                  <div className="text-center">
                    <div className="text-2xl font-bold text-foreground">
                      {(prediction.evaluation.accuracy * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-muted-foreground">Accuracy</div>
                  </div>
                )}
                {prediction.evaluation.brier !== undefined && (
                  <div className="text-center">
                    <div className="text-2xl font-bold text-foreground">
                      {prediction.evaluation.brier.toFixed(3)}
                    </div>
                    <div className="text-xs text-muted-foreground">Brier Score</div>
                  </div>
                )}
              </div>

              {prediction.evaluation.notes && (
                <p className="text-sm text-muted-foreground text-center">
                  {prediction.evaluation.notes}
                </p>
              )}

              {/* History */}
              {prediction.history && prediction.history.length > 0 && (
                <div className="pt-3 border-t border-border">
                  <div className="flex items-center gap-2 mb-3">
                    <History className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium">History</span>
                  </div>
                  <div className="space-y-2">
                    {prediction.history.map((entry, idx) => (
                      <div
                        key={idx}
                        className="bg-background rounded-lg p-3 flex items-center justify-between"
                      >
                        <div>
                          <span className="text-sm text-foreground">
                            {PREDICTION_STATUS_LABELS[entry.status]}
                          </span>
                          <p className="text-xs text-muted-foreground">
                            {new Date(entry.ts).toLocaleString()}
                          </p>
                        </div>
                        <ModelBadge model={entry.model} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <Clock className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                {prediction.status === "missing"
                  ? "No prediction generated"
                  : "Evaluation pending - match not finished"}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Prediction Detail Content - used for mobile sheet (tabs + content together)
 */
function PredictionDetailContentMobile({ prediction }: { prediction: PredictionDetail }) {
  const [activeTab, setActiveTab] = useState("details");

  return (
    <div className="w-full space-y-3">
      <IconTabs
        tabs={PREDICTION_TABS}
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      />
      <PredictionTabContent prediction={prediction} activeTab={activeTab} />
    </div>
  );
}

/**
 * Responsive Prediction Detail Drawer
 *
 * Desktop (>=1280px): Overlay drawer (no reflow, ~400px)
 * Mobile/Tablet (<1280px): Sheet overlay
 */
export function PredictionDetailDrawer({
  prediction,
  open,
  onClose,
  isLoading,
}: PredictionDetailDrawerProps) {
  const isDesktop = useIsDesktop();
  const [activeTab, setActiveTab] = useState("details");
  const predictionTitle = prediction
    ? `Prediction #${prediction.id}`
    : "Prediction Details";

  // Loading state content
  const loadingContent = (
    <div className="space-y-4">
      <div className="h-8 bg-surface rounded animate-pulse" />
      <div className="h-24 bg-surface rounded animate-pulse" />
      <div className="h-32 bg-surface rounded animate-pulse" />
    </div>
  );

  // Desktop: overlay drawer with tabs in fixedContent
  if (isDesktop) {
    return (
      <DetailDrawer
        open={open}
        onClose={onClose}
        title={predictionTitle}
        fixedContent={
          prediction && !isLoading && (
            <IconTabs
              tabs={PREDICTION_TABS}
              value={activeTab}
              onValueChange={setActiveTab}
              className="w-full"
            />
          )
        }
      >
        {isLoading ? (
          loadingContent
        ) : prediction ? (
          <PredictionTabContent prediction={prediction} activeTab={activeTab} />
        ) : (
          <p className="text-muted-foreground text-sm">
            Select a prediction to view details
          </p>
        )}
      </DetailDrawer>
    );
  }

  // Mobile/Tablet: Sheet overlay
  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md p-0" data-dev-ref="PredictionDetailDrawer">
        <SheetHeader className="px-4 py-3 border-b border-border">
          <SheetTitle className="text-sm font-semibold truncate">
            {predictionTitle}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-60px)]">
          <div className="p-4">
            {isLoading ? (
              loadingContent
            ) : prediction ? (
              <PredictionDetailContentMobile prediction={prediction} />
            ) : (
              <p className="text-muted-foreground text-sm">
                Select a prediction to view details
              </p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
