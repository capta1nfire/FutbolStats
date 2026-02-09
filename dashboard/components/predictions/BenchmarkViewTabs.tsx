"use client";

import { TrendingUp, Grid3X3 } from "lucide-react";
import { IconTabs } from "@/components/ui/icon-tabs";

export type PredictionsView = "predictions" | "benchmark";

const PREDICTIONS_VIEW_TABS = [
  { id: "predictions", icon: <TrendingUp />, label: "Predictions" },
  { id: "benchmark", icon: <Grid3X3 />, label: "Benchmark" },
];

interface BenchmarkViewTabsProps {
  activeView: PredictionsView;
  onViewChange: (view: PredictionsView) => void;
}

export function BenchmarkViewTabs({ activeView, onViewChange }: BenchmarkViewTabsProps) {
  return (
    <IconTabs
      tabs={PREDICTIONS_VIEW_TABS}
      value={activeView}
      onValueChange={(value) => onViewChange(value as PredictionsView)}
      className="w-full"
    />
  );
}
