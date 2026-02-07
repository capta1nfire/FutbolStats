/**
 * GAP20 Divergence Classification
 *
 * Client-side replica of predictions_gap20_diag_v1 VIEW logic.
 * Classifies model-vs-market divergence into AGREE / DISAGREE / STRONG_FAV_DISAGREE.
 *
 * Caveat: market probs from backend are round(1/odds, 3) — some edge cases
 * near gap=0.20 may classify differently vs the VIEW (full-precision odds).
 */

import type { ProbabilitySet } from "@/lib/types";

/** Matches VIEW column `divergence_category` exactly */
export type DivergenceCategory = "AGREE" | "DISAGREE" | "STRONG_FAV_DISAGREE";

export interface Gap20Result {
  category: DivergenceCategory;
  /** 0=home, 1=draw, 2=away */
  modelFav: number;
  /** 0=home, 1=draw, 2=away */
  marketFav: number;
  /** Gap on model's favored outcome (model_prob - market_prob, both renormalized) */
  gapOnModelFav: number;
  /** Market's max probability (renormalized) */
  marketFavProb: number;
}

/**
 * Replicate predictions_gap20_diag_v1 VIEW logic client-side.
 *
 * P0: Renormalizes BOTH modelA and market (backend rounds to 3 decimals,
 * neither is guaranteed to sum to exactly 1.0).
 *
 * Returns null if either set has sum <= 0 or contains non-finite values.
 */
export function computeGap20(
  modelA: ProbabilitySet,
  market: ProbabilitySet,
): Gap20Result | null {
  // Guard: non-finite values
  const vals = [modelA.home, modelA.draw, modelA.away, market.home, market.draw, market.away];
  if (vals.some((v) => !Number.isFinite(v))) return null;

  // Renormalize model (P0: backend round(...,3) may not sum to 1)
  const mSum = modelA.home + modelA.draw + modelA.away;
  if (mSum <= 0) return null;
  const mH = modelA.home / mSum;
  const mD = modelA.draw / mSum;
  const mA = modelA.away / mSum;

  // Renormalize market (raw implied probs, sum > 1 due to overround)
  const mktSum = market.home + market.draw + market.away;
  if (mktSum <= 0) return null;
  const mktH = market.home / mktSum;
  const mktD = market.draw / mktSum;
  const mktA = market.away / mktSum;

  // Argmax — same tie-breaking as VIEW (CASE WHEN >= chains)
  const modelFav = mH >= mD && mH >= mA ? 0 : mD >= mH && mD >= mA ? 1 : 2;
  const marketFav = mktH >= mktD && mktH >= mktA ? 0 : mktD >= mktH && mktD >= mktA ? 1 : 2;

  const marketFavProb = Math.max(mktH, mktD, mktA);

  // Gap on model's favored outcome
  const modelProbs = [mH, mD, mA];
  const mktProbs = [mktH, mktD, mktA];
  const gapOnModelFav = modelProbs[modelFav] - mktProbs[modelFav];

  // Classify (mirrors VIEW CASE expression)
  let category: DivergenceCategory;
  if (modelFav === marketFav) {
    category = "AGREE";
  } else if (gapOnModelFav >= 0.20 && marketFavProb >= 0.45) {
    category = "STRONG_FAV_DISAGREE";
  } else {
    category = "DISAGREE";
  }

  return { category, modelFav, marketFav, gapOnModelFav, marketFavProb };
}
