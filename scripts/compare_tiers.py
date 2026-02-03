#!/usr/bin/env python3
"""
Compare evaluation results across TITAN tiers.

Usage:
    python scripts/compare_tiers.py logs/pit_evaluation_v3_*experiments*.json
    python scripts/compare_tiers.py logs/*.json --pattern "*experiments*"

Output:
    Summary table with skill_vs_market delta by tier and recommendation
"""

import argparse
import json
import sys
from pathlib import Path


def load_evaluation(filepath: Path) -> dict:
    """Load evaluation JSON and extract key metrics."""
    with open(filepath) as f:
        data = json.load(f)

    # Extract key metrics
    brier = data.get('brier', {})
    betting = data.get('betting', {})
    filters = data.get('filters', {})

    return {
        'file': filepath.name,
        'model_version': filters.get('model_version') or data.get('prediction_integrity', {}).get('model_version_filter', 'unknown'),
        'source': data.get('prediction_integrity', {}).get('source', 'predictions'),
        'n': brier.get('n_with_predictions', 0),
        'brier_model': brier.get('brier_model'),
        'brier_market': brier.get('brier_market'),
        'skill_vs_market': brier.get('skill_vs_market'),
        'skill_vs_uniform': brier.get('skill_vs_uniform'),
        'logloss_model': brier.get('logloss_model'),
        'logloss_skill_vs_market': brier.get('logloss_skill_vs_market'),
        'ece': brier.get('ece'),
        'brier_diff_mean': brier.get('paired_differential', {}).get('brier_diff_mean'),
        'brier_diff_ci95': brier.get('paired_differential', {}).get('brier_diff_ci95'),
        'n_bets': betting.get('n_bets'),
        'roi': betting.get('roi'),
        'roi_ci95_low': betting.get('roi_ci95_low'),
        'roi_ci95_high': betting.get('roi_ci95_high'),
        'clv_mean': betting.get('clv_mean'),
        'phase': data.get('phase'),
        'verdict': data.get('interpretation', {}).get('verdict'),
    }


def print_comparison(results: list[dict]):
    """Print comparison table."""
    print("=" * 100)
    print("TITAN TIER COMPARISON RESULTS")
    print("=" * 100)

    # Header
    print(f"{'Model Version':<25} {'N':>6} {'skill_vs_mkt':>12} {'logloss':>10} {'ROI':>8} {'CI95':<25}")
    print("-" * 100)

    for r in results:
        ci = r['brier_diff_ci95']
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A"
        skill = r['skill_vs_market']
        skill_str = f"{skill*100:+.2f}%" if skill is not None else "N/A"
        roi = r['roi']
        roi_str = f"{roi*100:+.2f}%" if roi is not None else "N/A"
        logloss = r['logloss_model']
        logloss_str = f"{logloss:.4f}" if logloss is not None else "N/A"

        print(f"{r['model_version']:<25} {r['n']:>6} {skill_str:>12} {logloss_str:>10} {roi_str:>8} {ci_str:<25}")

    print("=" * 100)

    # Detailed comparison
    print("\nDETAILED METRICS:")
    print("-" * 100)
    print(f"{'Model Version':<25} {'ECE':>8} {'CLV_mean':>10} {'n_bets':>8} {'Phase':>12} {'Verdict':>10}")
    print("-" * 100)

    for r in results:
        ece = r['ece']
        ece_str = f"{ece:.4f}" if ece is not None else "N/A"
        clv = r['clv_mean']
        clv_str = f"{clv:.4f}" if clv is not None else "N/A"
        n_bets = r['n_bets'] or 0
        phase = r['phase'] or "N/A"
        verdict = r['verdict'] or "N/A"

        print(f"{r['model_version']:<25} {ece_str:>8} {clv_str:>10} {n_bets:>8} {phase:>12} {verdict:>10}")


def generate_recommendation(results: list[dict]) -> str:
    """Generate recommendation based on results."""
    if not results:
        return "NO DATA: No evaluation files found"

    # Find best and baseline
    baseline = next((r for r in results if 'baseline' in r['model_version'].lower() or r['model_version'] == 'v1.0.0'), None)
    best = results[0]  # Already sorted by skill_vs_market

    recommendations = []

    # Check if any tier beats market
    market_beaters = [r for r in results if r['skill_vs_market'] is not None and r['skill_vs_market'] >= 0]
    if market_beaters:
        best_market_beater = market_beaters[0]
        recommendations.append(f"GO: {best_market_beater['model_version']} beats market (skill={best_market_beater['skill_vs_market']*100:.2f}%)")
    else:
        recommendations.append("NO tier beats market yet")

    # Check for improvement over baseline
    if baseline and best['model_version'] != baseline['model_version']:
        baseline_skill = baseline['skill_vs_market'] or -1
        best_skill = best['skill_vs_market'] or -1
        improvement = best_skill - baseline_skill
        if improvement > 0:
            recommendations.append(f"IMPROVEMENT: {best['model_version']} is {improvement*100:.2f}pp better than baseline")
        else:
            recommendations.append(f"NO IMPROVEMENT over baseline")

    # Check N
    if best['n'] < 200:
        recommendations.append(f"WARNING: N={best['n']} is below 200 threshold for preliminary phase")
    elif best['n'] < 500:
        recommendations.append(f"NOTE: N={best['n']} is in preliminary phase (need 500+ for formal)")

    # Final verdict
    if market_beaters:
        final = f"GO: Continue to N>=500 with {market_beaters[0]['model_version']}"
    elif best['skill_vs_market'] and best['skill_vs_market'] > -0.05:
        final = f"HOLD: {best['model_version']} shows promise but needs more data"
    else:
        final = "NO-GO: No tier shows improvement, review architecture"

    return "\n".join([
        "=" * 100,
        "RECOMMENDATION",
        "=" * 100,
        *recommendations,
        "",
        f"FINAL: {final}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Compare TITAN tier evaluation results")
    parser.add_argument('files', nargs='+', help='JSON evaluation files to compare')
    parser.add_argument('--sort', type=str, default='skill_vs_market',
                        choices=['skill_vs_market', 'roi', 'n', 'logloss'],
                        help='Sort by metric (default: skill_vs_market)')
    args = parser.parse_args()

    # Load all evaluations
    results = []
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        try:
            results.append(load_evaluation(path))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    if not results:
        print("ERROR: No valid evaluation files found")
        return 1

    # Sort results
    reverse = args.sort != 'logloss'  # Higher is better for most metrics
    results.sort(key=lambda x: x.get(args.sort) or -999, reverse=reverse)

    # Print comparison
    print_comparison(results)

    # Generate recommendation
    print("\n" + generate_recommendation(results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
