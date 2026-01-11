#!/usr/bin/env python3
"""
Validate LLM narrative compliance with prompt v4 rules.

Usage:
    python scripts/validate_narrative.py <match_id>
    python scripts/validate_narrative.py --recent  # Check last 5 matches with narratives
"""

import argparse
import json
import sys
from typing import Optional
import requests

BASE_URL = "https://web-production-f2de9.up.railway.app"

# Words that should be digits, not written out
WRITTEN_NUMBERS = [
    "uno", "dos", "tres", "cuatro", "cinco",
    "seis", "siete", "ocho", "nueve", "diez",
    "once", "doce", "trece", "catorce", "quince",
]


def get_narrative(match_id: int) -> Optional[dict]:
    """Fetch narrative from API."""
    try:
        resp = requests.get(f"{BASE_URL}/matches/{match_id}/insights", timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("llm_narrative")
    except Exception as e:
        print(f"Error fetching match {match_id}: {e}")
        return None


def validate_narrative(narrative: dict, match_id: int) -> dict:
    """
    Validate narrative against v4 rules.

    Returns dict with validation results.
    """
    results = {
        "match_id": match_id,
        "issues": [],
        "passed": True,
    }

    narr = narrative.get("narrative", {})
    body = narr.get("body", "")
    key_factors = narr.get("key_factors", [])
    ft_score = narrative.get("result", {}).get("ft_score", "")

    # 1. Check for written numbers
    body_lower = body.lower()
    found_written = [w for w in WRITTEN_NUMBERS if f" {w} " in f" {body_lower} " or body_lower.startswith(f"{w} ") or body_lower.endswith(f" {w}")]
    if found_written:
        results["issues"].append({
            "rule": "DIGITS_ONLY",
            "severity": "warning",
            "detail": f"Found written numbers: {found_written}",
        })

    # 2. Check for score repetition in body
    if ft_score and ft_score in body:
        results["issues"].append({
            "rule": "NO_SCORE_IN_BODY",
            "severity": "warning",
            "detail": f"Score '{ft_score}' found in body",
        })

    # 3. Check Events wording
    events_factor = next((kf for kf in key_factors if kf.get("label") == "Events"), None)
    if events_factor:
        evidence = events_factor.get("evidence", "")
        if "no hubo" in evidence.lower():
            results["issues"].append({
                "rule": "EVENTS_WORDING",
                "severity": "error",
                "detail": f"Events uses 'No hubo' instead of 'Eventos no disponibles': {evidence}",
            })

    # 4. Check key_factors length (max 120 chars)
    for kf in key_factors:
        evidence = kf.get("evidence", "")
        if len(evidence) > 120:
            results["issues"].append({
                "rule": "KEY_FACTORS_LENGTH",
                "severity": "warning",
                "detail": f"{kf.get('label')} evidence too long ({len(evidence)} chars): {evidence[:50]}...",
            })

    # 5. Check body length (140-240 words target)
    word_count = len(body.split())
    if word_count > 280:  # Allow some buffer
        results["issues"].append({
            "rule": "BODY_LENGTH",
            "severity": "warning",
            "detail": f"Body too long: {word_count} words (target: 140-240)",
        })

    results["passed"] = len([i for i in results["issues"] if i["severity"] == "error"]) == 0
    results["word_count"] = word_count
    results["key_factors_count"] = len(key_factors)

    return results


def print_results(results: dict):
    """Print validation results."""
    match_id = results["match_id"]
    passed = results["passed"]
    issues = results["issues"]

    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\nMatch {match_id}: {status}")
    print(f"  Word count: {results.get('word_count', 'N/A')}")
    print(f"  Key factors: {results.get('key_factors_count', 'N/A')}")

    if issues:
        print("  Issues:")
        for issue in issues:
            severity_icon = "❌" if issue["severity"] == "error" else "⚠️"
            print(f"    {severity_icon} [{issue['rule']}] {issue['detail']}")
    else:
        print("  No issues found")


def main():
    parser = argparse.ArgumentParser(description="Validate LLM narrative compliance")
    parser.add_argument("match_id", nargs="?", type=int, help="Match ID to validate")
    parser.add_argument("--recent", action="store_true", help="Check recent matches")
    args = parser.parse_args()

    if args.match_id:
        narrative = get_narrative(args.match_id)
        if not narrative:
            print(f"No narrative found for match {args.match_id}")
            sys.exit(1)

        results = validate_narrative(narrative, args.match_id)
        print_results(results)
        sys.exit(0 if results["passed"] else 1)

    elif args.recent:
        # Would need an endpoint to list recent matches with narratives
        print("--recent not implemented yet. Please provide a match_id.")
        sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
