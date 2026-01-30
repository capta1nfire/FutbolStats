/**
 * Prediction Pick Utilities
 *
 * Handles model pick calculation with tie-breaking logic.
 * When multiple outcomes have similar max probabilities, they are treated as "co-picks"
 * rather than arbitrarily choosing one based on evaluation order.
 */

/**
 * Epsilon for probability comparison (in the same unit as input probs).
 * - If probs are 0-1: use 0.005 (0.5 percentage points)
 * - If probs are 0-100: use 0.5 (0.5 percentage points)
 *
 * This allows for floating point rounding tolerance.
 */
export const PROB_EPSILON_NORMALIZED = 0.005; // For 0-1 scale
export const PROB_EPSILON_PERCENT = 0.5; // For 0-100 scale

export type Outcome = "home" | "draw" | "away";

export interface ProbabilityInput {
  home: number;
  draw: number;
  away: number;
}

export interface PredictionPickResult {
  /** Outcomes that are tied for maximum probability */
  topOutcomes: Outcome[];
  /** True if multiple outcomes are tied for max */
  isTie: boolean;
  /** Display string for UI: "H", "D", "A", "H/D", "D/A", "H/A", "H/D/A" */
  displayPick: string;
  /** Whether the actual outcome is among the top picks (null if no outcome provided) */
  isCorrect: boolean | null;
  /** CSS class hint for styling */
  matchClass: "correct" | "wrong" | "tie-correct" | "tie-wrong" | "pending";
}

/**
 * Map outcome to display initial
 */
const OUTCOME_INITIALS: Record<Outcome, string> = {
  home: "H",
  draw: "D",
  away: "A",
};

/**
 * Calculate the model's prediction pick(s) considering ties.
 *
 * @param probs - Probability set (home, draw, away)
 * @param actualOutcome - The actual match result (null if not finished)
 * @param epsilon - Tolerance for considering probabilities as equal (default: auto-detect based on scale)
 * @returns PredictionPickResult with pick info and correctness
 *
 * @example
 * // Single pick
 * getPredictionPick({ home: 0.52, draw: 0.28, away: 0.20 }, "home")
 * // => { topOutcomes: ["home"], isTie: false, displayPick: "H", isCorrect: true, matchClass: "correct" }
 *
 * @example
 * // Tied pick - correct
 * getPredictionPick({ home: 0.41, draw: 0.41, away: 0.18 }, "draw")
 * // => { topOutcomes: ["home", "draw"], isTie: true, displayPick: "H/D", isCorrect: true, matchClass: "tie-correct" }
 *
 * @example
 * // Tied pick - wrong
 * getPredictionPick({ home: 0.41, draw: 0.41, away: 0.18 }, "away")
 * // => { topOutcomes: ["home", "draw"], isTie: true, displayPick: "H/D", isCorrect: false, matchClass: "tie-wrong" }
 */
export function getPredictionPick(
  probs: ProbabilityInput,
  actualOutcome: Outcome | null = null,
  epsilon?: number
): PredictionPickResult {
  // Auto-detect epsilon based on probability scale
  const maxVal = Math.max(probs.home, probs.draw, probs.away);
  const detectedEpsilon =
    epsilon ?? (maxVal > 1 ? PROB_EPSILON_PERCENT : PROB_EPSILON_NORMALIZED);

  // Find max probability
  const maxProb = Math.max(probs.home, probs.draw, probs.away);

  // Find all outcomes within epsilon of max (tied for first)
  const topOutcomes: Outcome[] = [];

  // Check in consistent order: home, draw, away
  if (Math.abs(probs.home - maxProb) <= detectedEpsilon) {
    topOutcomes.push("home");
  }
  if (Math.abs(probs.draw - maxProb) <= detectedEpsilon) {
    topOutcomes.push("draw");
  }
  if (Math.abs(probs.away - maxProb) <= detectedEpsilon) {
    topOutcomes.push("away");
  }

  const isTie = topOutcomes.length > 1;

  // Build display string
  const displayPick = topOutcomes.map((o) => OUTCOME_INITIALS[o]).join("/");

  // Determine correctness
  let isCorrect: boolean | null = null;
  let matchClass: PredictionPickResult["matchClass"] = "pending";

  if (actualOutcome !== null) {
    isCorrect = topOutcomes.includes(actualOutcome);

    if (isTie) {
      matchClass = isCorrect ? "tie-correct" : "tie-wrong";
    } else {
      matchClass = isCorrect ? "correct" : "wrong";
    }
  }

  return {
    topOutcomes,
    isTie,
    displayPick,
    isCorrect,
    matchClass,
  };
}

/**
 * Get CSS classes for probability cell styling based on match result.
 *
 * @param probType - The outcome this cell represents (home, draw, away)
 * @param pickResult - Result from getPredictionPick
 * @returns Tailwind CSS classes for the cell
 */
export function getProbabilityCellClasses(
  probType: Outcome,
  pickResult: PredictionPickResult
): string {
  const isTopPick = pickResult.topOutcomes.includes(probType);

  // Match not finished - highlight picks with medium weight
  if (pickResult.matchClass === "pending") {
    return isTopPick ? "text-foreground font-medium" : "text-muted-foreground";
  }

  // Match finished
  if (!isTopPick) {
    // Non-pick outcomes stay muted
    return "text-muted-foreground";
  }

  // This is a top pick - color based on correctness
  switch (pickResult.matchClass) {
    case "correct":
      return "text-success font-medium"; // Green: single pick was correct
    case "tie-correct":
      return "text-amber-500 font-medium"; // Amber: tied pick, one of them correct
    case "wrong":
      return "text-error font-medium"; // Red: single pick was wrong
    case "tie-wrong":
      return "text-error font-medium"; // Red: tied picks, none correct
    default:
      return "text-muted-foreground";
  }
}
