"""
Cascade handler for LINEUP_CONFIRMED events.

Phase 2, P2-10. Asymmetric League-Router (GDT M3).

Flow:
1. Validate match NS + prediction not frozen
2. Get features for match via FeatureEngineer
3. Classify league tier (T1/T2/T3) via league_router
4. For T3 + MTV enabled: compute talent_delta BEFORE prediction, inject features
5. Re-predict with XGBoost (Phase 1 or Family S model)
6. For T1/T2: compute talent_delta AFTER prediction (SteamChaser logging)
7. Apply Market Anchor with fresh odds
8. UPSERT prediction with new asof_timestamp

Steel degradation (ATI #3):
- talent_delta timeout/error: skip, proceed with baseline (no MTV features)
- Feature fetch / model fail: ABORT entirely, daily prediction stays active
- Total crash: Sweeper Queue finds match in next 2min cycle

Tier routing (GDT M3):
- T1 (Big 5): Baseline + high market anchor. talent_delta = SteamChaser log only.
- T2 (Peripheral neutral): Baseline + moderate market anchor. talent_delta = SteamChaser log.
- T3 (MTV winners): Family S model + MTV injection (when enabled). talent_delta = injection.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import text, select

from app.events.bus import Event

logger = logging.getLogger("futbolstats.events")

# ATI #3: Hard timeout for talent delta computation
TALENT_DELTA_TIMEOUT_S = 5.0


async def cascade_handler(event: Event):
    """
    Handle LINEUP_CONFIRMED: re-predict with fresh odds + attempt talent delta.

    Idempotent: safe to call multiple times for the same match.
    The UPSERT on (match_id, model_version) ensures atomic write.
    """
    match_id = event.payload.get("match_id")
    source = event.payload.get("source", "unknown")
    lineup_detected_at = event.payload.get("lineup_detected_at")

    if not match_id:
        logger.warning("[CASCADE] Event missing match_id, skipping")
        return

    logger.info(f"[CASCADE] Processing match_id={match_id} (source={source})")
    start_time = time.time()

    try:
        from app.database import get_session_with_retry
        from app.db_utils import upsert
        from app.ml import XGBoostEngine
        from app.models import Prediction, Match
        from app.features.engineering import (
            FeatureEngineer,
            compute_match_talent_delta_features,
            compute_line_movement_features,
        )
        from app.ml.policy import apply_market_anchor, get_policy_config
        from app.ml.league_router import get_prediction_strategy
        from app.config import get_settings

        # ── Step 1: Validate match is still NS and prediction is not frozen ──
        async with get_session_with_retry(max_retries=2, retry_delay=0.5) as session:
            result = await session.execute(text("""
                SELECT
                    m.id,
                    m.status,
                    m.date,
                    m.league_id,
                    COALESCE(BOOL_OR(p.is_frozen), false) AS is_frozen,
                    MAX(p.asof_timestamp) AS pred_asof
                FROM matches m
                LEFT JOIN predictions p
                    ON p.match_id = m.id
                WHERE m.id = :match_id
                GROUP BY m.id, m.status, m.date, m.league_id
            """), {"match_id": match_id})
            row = result.fetchone()

        if not row:
            logger.warning(f"[CASCADE] Match {match_id} not found, aborting")
            return

        if row.status != "NS":
            logger.info(f"[CASCADE] Match {match_id} status={row.status}, skipping (not NS)")
            return

        if row.is_frozen:
            logger.info(f"[CASCADE] Match {match_id} prediction frozen, skipping")
            return

        # PIT/idempotency: if we already have a prediction saved AFTER lineup_detected_at,
        # skip. This prevents duplicate processing from sweeper + monitoring + multi-worker.
        # Normalize both to naive UTC to avoid offset-naive vs offset-aware comparison.
        if isinstance(lineup_detected_at, datetime) and row.pred_asof:
            _pred_asof = row.pred_asof.replace(tzinfo=None) if row.pred_asof.tzinfo else row.pred_asof
            _lda = lineup_detected_at.replace(tzinfo=None) if lineup_detected_at.tzinfo else lineup_detected_at
            if _pred_asof >= _lda:
                logger.info(
                    f"[CASCADE] Match {match_id} already has post-lineup prediction "
                    f"(pred_asof={row.pred_asof}, lineup_detected_at={lineup_detected_at}), skipping"
                )
                return

        # Canonical asof timestamp for this cascade run (PIT anchor for auditing).
        # Prefer lineup_detected_at if provided; fallback to now.
        asof = lineup_detected_at if isinstance(lineup_detected_at, datetime) else datetime.utcnow()

        # ── Step 2: Load ML engine ───────────────────────────────────────────
        engine = XGBoostEngine()
        if not engine.load_model():
            logger.error(f"[CASCADE] Could not load ML model for match {match_id}")
            return

        # ── Step 3: Get features for this match ─────────────────────────────
        async with get_session_with_retry(max_retries=2, retry_delay=0.5) as session:
            feature_engineer = FeatureEngineer(session=session)
            df = await feature_engineer.get_upcoming_matches_features(league_only=True)

        if len(df) == 0 or match_id not in df["match_id"].values:
            logger.warning(f"[CASCADE] No features for match {match_id}, aborting (daily prediction stays)")
            return

        df_match = df[df["match_id"] == match_id].reset_index(drop=True)

        # ── Step 3b: League Router — classify tier ─────────────────────────
        league_id = row.league_id
        _settings = get_settings()
        strategy = get_prediction_strategy(
            league_id, mtv_enabled=_settings.LEAGUE_ROUTER_MTV_ENABLED
        )
        logger.info(
            f"[CASCADE] Router match_id={match_id} league={league_id}: "
            f"tier={strategy['tier']} strategy={strategy['label']} "
            f"mtv_purpose={strategy['talent_delta_purpose']}"
        )

        # ── Step 4a: For T3 + MTV enabled, compute talent_delta BEFORE prediction
        talent_delta_result = None
        if strategy["inject_mtv"]:
            try:
                async with get_session_with_retry(max_retries=1, retry_delay=0.5) as session:
                    match_obj = (await session.execute(
                        select(Match).where(Match.id == match_id)
                    )).scalar_one_or_none()

                    if match_obj:
                        talent_delta_result = await asyncio.wait_for(
                            compute_match_talent_delta_features(session, match_obj),
                            timeout=TALENT_DELTA_TIMEOUT_S,
                        )
                        # Inject MTV features into df_match for Family S model
                        for mtv_col in ["home_talent_delta", "away_talent_delta",
                                        "talent_delta_diff", "shock_magnitude"]:
                            df_match[mtv_col] = talent_delta_result.get(mtv_col)
                        logger.info(
                            f"[CASCADE] T3 MTV INJECTED match_id={match_id}: "
                            f"home={talent_delta_result.get('home_talent_delta')}, "
                            f"away={talent_delta_result.get('away_talent_delta')}, "
                            f"shock={talent_delta_result.get('shock_magnitude')}"
                        )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[CASCADE] T3 talent_delta TIMEOUT ({TALENT_DELTA_TIMEOUT_S}s) "
                    f"for match {match_id}, falling back to baseline"
                )
            except Exception as e:
                logger.warning(
                    f"[CASCADE] T3 talent_delta failed for match {match_id}: {e}, "
                    f"falling back to baseline"
                )

        # ── Step 4b: Predict ──────────────────────────────────────────────
        predictions = engine.predict(df_match)
        if not predictions or not predictions[0].get("probabilities"):
            logger.error(f"[CASCADE] Model returned no prediction for match {match_id}")
            return

        # ── Step 5: For T1/T2, compute talent_delta AFTER prediction (SteamChaser)
        if not strategy["inject_mtv"]:
            try:
                async with get_session_with_retry(max_retries=1, retry_delay=0.5) as session:
                    match_obj = (await session.execute(
                        select(Match).where(Match.id == match_id)
                    )).scalar_one_or_none()

                    if match_obj:
                        talent_delta_result = await asyncio.wait_for(
                            compute_match_talent_delta_features(session, match_obj),
                            timeout=TALENT_DELTA_TIMEOUT_S,
                        )
                        logger.info(
                            f"[CASCADE] SteamChaser talent_delta match_id={match_id} "
                            f"(tier={strategy['tier']}): "
                            f"home={talent_delta_result.get('home_talent_delta')}, "
                            f"away={talent_delta_result.get('away_talent_delta')}, "
                            f"shock={talent_delta_result.get('shock_magnitude')}"
                        )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[CASCADE] talent_delta TIMEOUT ({TALENT_DELTA_TIMEOUT_S}s) "
                    f"for match {match_id}, proceeding without MTV"
                )
            except Exception as e:
                logger.warning(
                    f"[CASCADE] talent_delta failed for match {match_id}: {e}, "
                    f"proceeding without MTV"
                )

        # ── Step 5b: Line movement features (PIT-safe — ATI Sprint 3) ────────
        # Forward data collection only — not yet in model.
        line_movement = None
        try:
            async with get_session_with_retry(max_retries=1, retry_delay=0.5) as session:
                line_movement = await compute_line_movement_features(
                    session, match_id, asof_timestamp=asof,
                )
                if line_movement.get("line_drift_magnitude") is not None:
                    logger.info(
                        f"[CASCADE] line_movement match_id={match_id}: "
                        f"drift_mag={line_movement['line_drift_magnitude']}, "
                        f"overround_T60={line_movement.get('overround_T60')}"
                    )
        except Exception as e:
            logger.warning(f"[CASCADE] line_movement failed for match {match_id}: {e}")

        # ── Step 6: Apply Market Anchor with fresh odds ──────────────────────
        policy_cfg = get_policy_config()
        anchored, anchor_meta = apply_market_anchor(
            predictions,
            alpha_default=policy_cfg["market_anchor_alpha_default"],
            league_overrides=policy_cfg["market_anchor_league_overrides"],
            enabled=policy_cfg["market_anchor_enabled"],
        )
        pred = anchored[0]
        probs = pred["probabilities"]

        # ── Step 7: UPSERT prediction with new asof_timestamp ────────────────
        async with get_session_with_retry(max_retries=2, retry_delay=0.5) as session:
            await upsert(
                session,
                Prediction,
                values={
                    "match_id": match_id,
                    "model_version": engine.model_version,
                    "home_prob": probs["home"],
                    "draw_prob": probs["draw"],
                    "away_prob": probs["away"],
                    "asof_timestamp": asof,
                },
                conflict_columns=["match_id", "model_version"],
                update_columns=["home_prob", "draw_prob", "away_prob", "asof_timestamp"],
            )
            await session.commit()

        elapsed_ms = (time.time() - start_time) * 1000
        anchor_applied = anchor_meta.get("anchored", 0) > 0
        mtv_status = "INJECTED" if strategy["inject_mtv"] and talent_delta_result else (
            "OK" if talent_delta_result else "SKIP"
        )
        logger.info(
            f"[CASCADE] Complete match_id={match_id} "
            f"tier={strategy['tier']} strategy={strategy['label']}: "
            f"H={probs['home']:.4f} D={probs['draw']:.4f} A={probs['away']:.4f} "
            f"asof={asof.isoformat()} "
            f"mtv={mtv_status} "
            f"lm={'OK' if line_movement and line_movement.get('line_drift_magnitude') is not None else 'SKIP'} "
            f"anchor={'YES' if anchor_applied else 'NO'} "
            f"elapsed={elapsed_ms:.0f}ms"
        )

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[CASCADE] FAILED match_id={match_id}: {e} "
            f"(daily prediction remains active, elapsed={elapsed_ms:.0f}ms)",
            exc_info=True,
        )
