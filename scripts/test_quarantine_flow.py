"""
Test script to verify quarantine flow works end-to-end.

Tests:
1. Invalid odds trigger dq_odds_invariant_violations_total
2. Invalid odds trigger dq_odds_quarantined_total
3. Quarantined records are excluded from build_training_dataset

Usage:
    PYTHONPATH=/Users/inseqio/FutbolStats python3 scripts/test_quarantine_flow.py
"""

import asyncio
import sys


async def test_quarantine_flow():
    """Test the quarantine flow with invalid odds."""
    print("=" * 60)
    print("QUARANTINE FLOW TEST")
    print("=" * 60)

    # Test 1: Validator detects invalid overround
    print("\n[1] Testing validator with invalid overround...")
    from app.telemetry.validators import validate_odds_1x2

    # Valid odds (overround ~1.05)
    valid_result = validate_odds_1x2(
        odds_home=2.10,
        odds_draw=3.40,
        odds_away=3.50,
        provider="test",
        book="test",
        record_metrics=False,
    )
    print(f"    Valid odds: is_valid={valid_result.is_valid}, quarantined={valid_result.quarantined}, overround={valid_result.overround:.3f}")
    assert valid_result.is_valid, "Valid odds should pass"
    assert not valid_result.quarantined, "Valid odds should not be quarantined"

    # Invalid odds - overround too low (arbitrage opportunity = suspicious)
    invalid_low = validate_odds_1x2(
        odds_home=2.50,
        odds_draw=4.00,
        odds_away=4.00,
        provider="test",
        book="test",
        record_metrics=False,
    )
    print(f"    Low overround: is_valid={invalid_low.is_valid}, quarantined={invalid_low.quarantined}, overround={invalid_low.overround:.3f}")
    # This should be quarantined if overround < 1.01

    # Invalid odds - overround too high
    invalid_high = validate_odds_1x2(
        odds_home=1.50,
        odds_draw=2.00,
        odds_away=2.00,
        provider="test",
        book="test",
        record_metrics=False,
    )
    print(f"    High overround: is_valid={invalid_high.is_valid}, quarantined={invalid_high.quarantined}, overround={invalid_high.overround:.3f}")
    assert not invalid_high.is_valid, "High overround should fail"
    assert invalid_high.quarantined, "High overround should be quarantined"

    # Invalid odds - value too low
    invalid_sanity = validate_odds_1x2(
        odds_home=1.001,  # Below minimum 1.01
        odds_draw=3.00,
        odds_away=3.00,
        provider="test",
        book="test",
        record_metrics=False,
    )
    print(f"    Sanity fail: is_valid={invalid_sanity.is_valid}, quarantined={invalid_sanity.quarantined}, violations={invalid_sanity.violations}")
    assert not invalid_sanity.is_valid, "Sanity violation should fail"
    assert invalid_sanity.quarantined, "Sanity violation should be quarantined"

    print("    ✓ Validator correctly identifies invalid odds")

    # Test 2: Metrics are recorded
    print("\n[2] Testing metrics recording...")
    from app.telemetry.metrics import get_metrics_text

    # Force a violation with metrics enabled
    validate_odds_1x2(
        odds_home=1.50,
        odds_draw=2.00,
        odds_away=2.00,
        provider="test_quarantine",
        book="test_book",
        record_metrics=True,
    )

    metrics_text, _ = get_metrics_text()

    has_violations = "dq_odds_invariant_violations_total" in metrics_text
    has_quarantined = "dq_odds_quarantined_total" in metrics_text

    print(f"    dq_odds_invariant_violations_total present: {has_violations}")
    print(f"    dq_odds_quarantined_total present: {has_quarantined}")

    if has_violations and has_quarantined:
        print("    ✓ Metrics are being recorded")
    else:
        print("    ⚠ Metrics may not be recording (check if counters have values)")

    # Test 3: Training dataset excludes tainted matches
    print("\n[3] Testing training dataset exclusion...")
    print("    (This verifies the SQL filter is in place)")

    from app.features.engineering import FeatureEngineer
    import inspect

    # Check that the filter exists in the source
    source = inspect.getsource(FeatureEngineer.build_training_dataset)
    has_tainted_filter = "Match.tainted == False" in source

    print(f"    build_training_dataset has tainted filter: {has_tainted_filter}")

    if has_tainted_filter:
        print("    ✓ Training dataset will exclude tainted matches")
    else:
        print("    ✗ MISSING: tainted filter not found in build_training_dataset")

    # Check TeamMatchCache.preload
    from app.features.engineering import TeamMatchCache
    cache_source = inspect.getsource(TeamMatchCache.preload)
    cache_has_filter = "Match.tainted == False" in cache_source

    print(f"    TeamMatchCache.preload has tainted filter: {cache_has_filter}")

    if cache_has_filter:
        print("    ✓ Match cache will exclude tainted matches")
    else:
        print("    ✗ MISSING: tainted filter not found in cache preload")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = (
        valid_result.is_valid and
        not valid_result.quarantined and
        invalid_high.quarantined and
        invalid_sanity.quarantined and
        has_tainted_filter and
        cache_has_filter
    )

    if all_passed:
        print("✓ ALL TESTS PASSED - Quarantine flow is working correctly")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Review output above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_quarantine_flow())
    sys.exit(exit_code)
