"""
Closing Line Value (CLV) scoring — Phase 2 (P2-04).

Computes CLV per outcome in log-odds for each prediction, using
the canonical bookmaker's odds at asof_timestamp vs closing line.

Definition:
    CLV_k = ln(odds_asof_k / odds_close_k)
    Positive = we got a better price than the close (timing edge)
    Negative = market moved against us

GDT #5: 3-way formalized, canonical bookmaker, same source for both timestamps.

Data reality: odds_history.recorded_at = our fetch time (not bookmaker set time).
Predictions are made days ahead; odds are first fetched ~48h before kickoff.
Strategy:
  - "asof" odds = latest recorded with recorded_at <= asof_timestamp (PIT-aligned)
  - "close" odds = is_closing=true if available, otherwise latest recorded <= kickoff
  - CLV measures line movement from our prediction time to close (timing edge)
"""

import logging
import math
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.devig import devig_proportional, select_canonical_bookmaker

logger = logging.getLogger(__name__)


async def compute_clv_for_match(
    session: AsyncSession,
    prediction_id: int,
    match_id: int,
    asof_timestamp: datetime,
    match_date: datetime = None,
) -> Optional[dict]:
    """
    Compute CLV for a single prediction.

    Steps:
    1. "asof" = latest odds with recorded_at <= asof_timestamp (PIT-aligned);
       falls back to earliest recorded (opening_proxy) if no odds before asof
    2. "close" = is_closing or latest pre-kickoff (recorded_at <= match_date)
    3. Select canonical bookmaker present in BOTH snapshots
    4. De-vig both sets of odds (proportional method)
    5. Compute CLV_k = ln(odds_asof_k / odds_close_k) per outcome

    Returns dict ready for insertion into prediction_clv, or None if insufficient data.
    """
    # Step 1: Get "asof" odds per bookmaker
    # Primary: latest recorded <= asof_timestamp (PIT-aligned, correct by construction)
    asof_source = "pit_aligned"
    opening_odds = await session.execute(text("""
        SELECT DISTINCT ON (source)
            source, odds_home, odds_draw, odds_away, recorded_at
        FROM odds_history
        WHERE match_id = :match_id
          AND recorded_at <= :asof_ts
          AND NOT quarantined AND NOT tainted
          AND odds_home > 1.0 AND odds_draw > 1.0 AND odds_away > 1.0
        ORDER BY source, recorded_at DESC
    """), {"match_id": match_id, "asof_ts": asof_timestamp})
    opening_rows = opening_odds.fetchall()

    # Fallback: earliest recorded (opening proxy) for historical predictions
    # where odds were fetched AFTER the prediction was made
    if not opening_rows:
        asof_source = "opening_proxy"
        opening_odds = await session.execute(text("""
            SELECT DISTINCT ON (source)
                source, odds_home, odds_draw, odds_away, recorded_at
            FROM odds_history
            WHERE match_id = :match_id
              AND NOT quarantined AND NOT tainted
              AND odds_home > 1.0 AND odds_draw > 1.0 AND odds_away > 1.0
            ORDER BY source, recorded_at ASC
        """), {"match_id": match_id})
        opening_rows = opening_odds.fetchall()

    if not opening_rows:
        return None

    # Step 2: Get CLOSING odds per bookmaker
    # Priority: is_closing=true, then latest recorded before kickoff
    # match_date filter ensures we don't use post-kickoff odds as "close"
    closing_odds = await session.execute(text("""
        SELECT DISTINCT ON (source)
            source, odds_home, odds_draw, odds_away, recorded_at,
            CASE WHEN is_closing THEN 'is_closing' ELSE 'latest_pre_kickoff' END as close_source
        FROM odds_history
        WHERE match_id = :match_id
          AND recorded_at <= COALESCE(:match_date, NOW())
          AND NOT quarantined AND NOT tainted
          AND odds_home > 1.0 AND odds_draw > 1.0 AND odds_away > 1.0
        ORDER BY source, is_closing DESC, recorded_at DESC
    """), {"match_id": match_id, "match_date": match_date})
    closing_rows = closing_odds.fetchall()

    if not closing_rows:
        return None

    # Build lookup dicts
    open_by_source = {}
    for r in opening_rows:
        open_by_source[r[0]] = (r[1], r[2], r[3])

    close_by_source = {}
    close_source_type = {}
    for r in closing_rows:
        close_by_source[r[0]] = (r[1], r[2], r[3])
        close_source_type[r[0]] = r[5]

    # Step 3: Select canonical bookmaker (must have BOTH opening and closing)
    sources_with_both = set(open_by_source.keys()) & set(close_by_source.keys())
    if not sources_with_both:
        return None

    canonical = select_canonical_bookmaker(sources_with_both)
    if not canonical:
        return None

    odds_open = open_by_source[canonical]
    odds_close = close_by_source[canonical]
    source_type = close_source_type[canonical]

    # Skip if opening == closing (single snapshot, no movement to measure)
    if odds_open == odds_close:
        source_type = "single_snapshot"
    # Encode asof_source into close_source for auditability
    # Format: "asof_source|close_source" (e.g., "pit_aligned|is_closing")
    source_type = f"{asof_source}|{source_type}"

    # Step 4: De-vig both
    prob_open = devig_proportional(*odds_open)
    prob_close = devig_proportional(*odds_close)

    # Step 5: Compute CLV per outcome in log-odds
    # CLV_k = ln(odds_open_k / odds_close_k) — positive means we had better price
    def safe_log_ratio(odds_a, odds_c):
        if odds_a <= 1.0 or odds_c <= 1.0:
            return None
        return round(math.log(odds_a / odds_c), 6)

    clv_home = safe_log_ratio(odds_open[0], odds_close[0])
    clv_draw = safe_log_ratio(odds_open[1], odds_close[1])
    clv_away = safe_log_ratio(odds_open[2], odds_close[2])

    return {
        "prediction_id": prediction_id,
        "match_id": match_id,
        "asof_timestamp": asof_timestamp,
        "canonical_bookmaker": canonical,
        "odds_asof_home": float(odds_open[0]),
        "odds_asof_draw": float(odds_open[1]),
        "odds_asof_away": float(odds_open[2]),
        "prob_asof_home": round(prob_open[0], 6),
        "prob_asof_draw": round(prob_open[1], 6),
        "prob_asof_away": round(prob_open[2], 6),
        "prob_close_home": round(prob_close[0], 6),
        "prob_close_draw": round(prob_close[1], 6),
        "prob_close_away": round(prob_close[2], 6),
        "clv_home": clv_home,
        "clv_draw": clv_draw,
        "clv_away": clv_away,
        "selected_outcome": None,
        "clv_selected": None,
        "close_source": source_type,
    }


async def score_clv_batch(
    session: AsyncSession,
    lookback_hours: int = 72,
    limit: int = 200,
) -> dict:
    """
    Score CLV for predictions of recently finished matches.

    Finds predictions where:
    - Match is FT/AET/PEN
    - Prediction has asof_timestamp
    - No CLV score exists yet

    Returns metrics dict.
    """
    # Find unscored predictions for finished matches
    result = await session.execute(text("""
        SELECT p.id, p.match_id, p.asof_timestamp, m.date
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        WHERE m.status IN ('FT', 'AET', 'PEN')
          AND m.date >= NOW() - CAST(:hours AS INT) * INTERVAL '1 hour'
          AND p.asof_timestamp IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM prediction_clv pc WHERE pc.prediction_id = p.id
          )
        ORDER BY m.date DESC
        LIMIT :lim
    """), {"hours": lookback_hours, "lim": limit})
    predictions = result.fetchall()

    metrics = {"scanned": len(predictions), "scored": 0, "skipped_no_odds": 0, "errors": 0}
    logger.info(f"CLV batch: {len(predictions)} predictions to process")

    for idx, (pred_id, match_id, asof_ts, match_date) in enumerate(predictions):
        try:
            clv = await compute_clv_for_match(
                session, pred_id, match_id, asof_ts, match_date
            )
            if clv is None:
                metrics["skipped_no_odds"] += 1
                continue

            await session.execute(text("""
                INSERT INTO prediction_clv
                    (prediction_id, match_id, asof_timestamp, canonical_bookmaker,
                     odds_asof_home, odds_asof_draw, odds_asof_away,
                     prob_asof_home, prob_asof_draw, prob_asof_away,
                     prob_close_home, prob_close_draw, prob_close_away,
                     clv_home, clv_draw, clv_away,
                     selected_outcome, clv_selected, close_source)
                VALUES
                    (:prediction_id, :match_id, :asof_timestamp, :canonical_bookmaker,
                     :odds_asof_home, :odds_asof_draw, :odds_asof_away,
                     :prob_asof_home, :prob_asof_draw, :prob_asof_away,
                     :prob_close_home, :prob_close_draw, :prob_close_away,
                     :clv_home, :clv_draw, :clv_away,
                     :selected_outcome, :clv_selected, :close_source)
                ON CONFLICT (prediction_id, canonical_bookmaker) DO NOTHING
            """), clv)
            metrics["scored"] += 1
        except Exception as e:
            logger.error(f"CLV error for prediction {pred_id}: {e}")
            metrics["errors"] += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"  CLV progress: {idx+1}/{len(predictions)} "
                        f"(scored={metrics['scored']}, skip={metrics['skipped_no_odds']})")

    await session.commit()
    return metrics
