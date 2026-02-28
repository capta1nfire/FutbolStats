"""Dashboard Settings API — Read-only operational settings, no secrets.

9 endpoints for settings dashboard: summary, feature flags, model versions,
IA features (GET/PATCH), prompt template, preview, call history, playground.
All protected by dashboard token.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, get_async_session
from app.models import Match, OddsHistory, PostMatchAudit, Prediction, PredictionOutcome, Team
from app.security import verify_dashboard_token_bool
from app.utils.cache import SimpleCache
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/dashboard/settings", tags=["settings"])

# Cache for settings endpoints
_settings_summary_cache = SimpleCache(ttl=300)
_settings_flags_cache = SimpleCache(ttl=60)
_settings_models_cache = SimpleCache(ttl=300)

# SECURITY: Secrets that MUST NEVER appear in settings responses
_SETTINGS_SECRET_KEYS = frozenset({
    "DATABASE_URL", "RAPIDAPI_KEY", "API_KEY", "DASHBOARD_TOKEN",
    "RUNPOD_API_KEY", "GEMINI_API_KEY", "METRICS_BEARER_TOKEN",
    "SMTP_PASSWORD", "OPS_ADMIN_PASSWORD", "OPS_SESSION_SECRET",
    "SENTRY_AUTH_TOKEN", "FUTBOLSTATS_API_KEY", "X_API_KEY",
})


def _check_token(request: Request):
    """Verify dashboard token (header + session + query param dev)."""
    if not verify_dashboard_token_bool(request):
        raise HTTPException(status_code=401, detail="Dashboard access requires valid token.")


def _is_env_configured(key: str) -> bool:
    """Check if an environment variable is configured (non-empty)."""
    val = os.environ.get(key, "")
    return bool(val and val.strip())


def _get_known_feature_flags() -> list[dict]:
    """
    Return list of known feature flags with metadata.

    NOTE: Values are bool or None, never secret strings.
    """
    flags = [
        # LLM/Narratives
        {"key": "FASTPATH_ENABLED", "scope": "llm", "description": "Enable FastPath LLM narrative generation"},
        {"key": "FASTPATH_DRY_RUN", "scope": "llm", "description": "FastPath dry-run mode (no writes)"},
        {"key": "GEMINI_ENABLED", "scope": "llm", "description": "Use Gemini as LLM provider"},
        {"key": "RUNPOD_ENABLED", "scope": "llm", "description": "Use RunPod as LLM provider"},
        # SOTA
        {"key": "SOTA_SOFASCORE_ENABLED", "scope": "sota", "description": "Enable Sofascore XI capture"},
        {"key": "SOTA_SOFASCORE_REFS_ENABLED", "scope": "sota", "description": "Enable Sofascore refs discovery"},
        {"key": "SOTA_WEATHER_ENABLED", "scope": "sota", "description": "Enable weather capture"},
        {"key": "SOTA_VENUE_GEO_ENABLED", "scope": "sota", "description": "Enable venue geocoding"},
        {"key": "SOTA_UNDERSTAT_ENABLED", "scope": "sota", "description": "Enable Understat xG capture"},
        # Sensor/Shadow
        {"key": "SENSOR_B_ENABLED", "scope": "sensor", "description": "Enable Sensor B calibration"},
        {"key": "SHADOW_MODE_ENABLED", "scope": "sensor", "description": "Enable shadow mode predictions"},
        {"key": "SHADOW_TWO_STAGE_ENABLED", "scope": "sensor", "description": "Enable two-stage shadow architecture"},
        # Jobs
        {"key": "SCHEDULER_ENABLED", "scope": "jobs", "description": "Enable background scheduler"},
        {"key": "ODDS_SYNC_ENABLED", "scope": "jobs", "description": "Enable odds sync job"},
        {"key": "STATS_BACKFILL_ENABLED", "scope": "jobs", "description": "Enable stats backfill job"},
        # Predictions
        {"key": "PREDICTIONS_ENABLED", "scope": "predictions", "description": "Enable prediction generation"},
        {"key": "PREDICTIONS_TWO_STAGE", "scope": "predictions", "description": "Use two-stage prediction model"},
        # Other
        {"key": "DEBUG", "scope": "other", "description": "Debug mode enabled"},
        {"key": "SENTRY_ENABLED", "scope": "other", "description": "Enable Sentry error tracking"},
    ]

    result = []
    for flag in flags:
        key = flag["key"]
        raw_val = os.environ.get(key, "").lower().strip()

        # Determine enabled state
        if raw_val in ("true", "1", "yes", "on"):
            enabled = True
        elif raw_val in ("false", "0", "no", "off"):
            enabled = False
        elif raw_val == "":
            enabled = None  # Not set
        else:
            enabled = None  # Unknown value

        result.append({
            "key": key,
            "enabled": enabled,
            "scope": flag["scope"],
            "description": flag["description"],
            "source": "env" if raw_val else "default",
        })

    return result


# =============================================================================
# Settings Endpoints
# =============================================================================


@router.get("/summary.json")
async def dashboard_settings_summary(request: Request):
    """
    Read-only summary of operational settings.

    Auth: X-Dashboard-Token required.
    TTL: 300s cache.

    SECURITY: No secrets or PII in response. Only configured: true/false.
    """
    _check_token(request)

    # Check cache
    hit, cached = _settings_summary_cache.get()
    if hit:
        return {
            "generated_at": cached["generated_at"],
            "cached": True,
            "cache_age_seconds": round(_settings_summary_cache.age, 1),
            "data": cached["payload"],
        }

    # Build fresh data
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        integrations = {
            "rapidapi": {
                "configured": _is_env_configured("RAPIDAPI_KEY") or _is_env_configured("API_FOOTBALL_KEY"),
                "source": "env",
            },
            "sentry": {
                "configured": _is_env_configured("SENTRY_DSN"),
                "source": "env",
            },
            "metrics": {
                "configured": _is_env_configured("METRICS_BEARER_TOKEN"),
                "source": "env",
            },
            "gemini": {
                "configured": _is_env_configured("GEMINI_API_KEY"),
                "source": "env",
            },
            "runpod": {
                "configured": _is_env_configured("RUNPOD_API_KEY"),
                "source": "env",
            },
            "database": {
                "configured": _is_env_configured("DATABASE_URL"),
                "source": "env",
            },
        }

        payload = {
            "readonly": True,
            "sections": ["general", "feature_flags", "model_versions", "integrations"],
            "notes": "Read-only operational settings. No secrets returned.",
            "links": [
                {"title": "Ops Dashboard", "url": "/dashboard/ops.json"},
                {"title": "Data Quality", "url": "/dashboard/data_quality.json"},
                {"title": "Feature Flags", "url": "/dashboard/settings/feature_flags.json"},
                {"title": "Model Versions", "url": "/dashboard/settings/model_versions.json"},
            ],
            "integrations": integrations,
        }

        # Update cache
        _settings_summary_cache.set({"generated_at": generated_at, "payload": payload})

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] summary.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "readonly": True,
                "sections": [],
                "notes": "Read-only operational settings. No secrets returned.",
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


@router.get("/feature_flags.json")
async def dashboard_settings_feature_flags(
    request: Request,
    q: str | None = None,
    enabled: bool | None = None,
    scope: str | None = None,
    page: int = 1,
    limit: int = 50,
):
    """
    Read-only list of feature flags.

    Auth: X-Dashboard-Token required.
    TTL: 60s cache (flags can change with deploy).

    Query params:
    - q: Search by key or description
    - enabled: Filter by true/false
    - scope: Filter by scope (llm, sota, jobs, sensor, predictions, other)
    - page: Page number (default 1)
    - limit: Items per page (default 50, max 100)

    SECURITY: No secrets. Only boolean enabled state.
    """
    _check_token(request)

    # Clamp limit
    limit = min(max(1, limit), 100)
    page = max(1, page)

    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        # Get all flags (always fresh since they depend on env)
        all_flags = _get_known_feature_flags()

        # Apply filters
        filtered = all_flags

        if q:
            q_lower = q.lower()
            filtered = [
                f for f in filtered
                if q_lower in f["key"].lower() or q_lower in f["description"].lower()
            ]

        if enabled is not None:
            filtered = [f for f in filtered if f["enabled"] == enabled]

        if scope:
            filtered = [f for f in filtered if f["scope"] == scope]

        # Pagination
        total = len(filtered)
        pages = (total + limit - 1) // limit if limit > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        paginated = filtered[start:end]

        return {
            "generated_at": generated_at,
            "cached": False,  # Always fresh for flags
            "cache_age_seconds": 0,
            "data": {
                "flags": paginated,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
            },
        }

    except Exception as e:
        logger.error(f"[SETTINGS] feature_flags.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "flags": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "pages": 0,
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


@router.get("/model_versions.json")
async def dashboard_settings_model_versions(request: Request):
    """
    Read-only list of ML model versions.

    Auth: X-Dashboard-Token required.
    TTL: 300s cache.

    SECURITY: No secrets. Only version strings.
    """
    _check_token(request)

    generated_at = datetime.now(timezone.utc).isoformat()

    # Check cache
    hit, cached = _settings_models_cache.get()
    if hit:
        return {
            "generated_at": cached["generated_at"],
            "cached": True,
            "cache_age_seconds": round(_settings_models_cache.age, 1),
            "data": cached["payload"],
        }

    try:
        import glob as glob_module
        import os.path

        # Get model versions from settings/env
        baseline_version = getattr(settings, "MODEL_VERSION", None) or os.environ.get("MODEL_VERSION", "v1.0.0")
        architecture = os.environ.get("MODEL_ARCHITECTURE", "baseline")
        shadow_version = os.environ.get("SHADOW_MODEL_VERSION", "disabled")
        shadow_architecture = os.environ.get("SHADOW_ARCHITECTURE", "disabled")

        # Try to get updated_at from model files if available
        model_updated_at = None
        try:
            model_files = glob_module.glob("models/xgb_*.json")
            if model_files:
                latest_file = max(model_files, key=os.path.getmtime)
                mtime = os.path.getmtime(latest_file)
                model_updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception:
            pass

        models = [
            {
                "name": "baseline",
                "version": baseline_version,
                "source": "settings",
                "updated_at": model_updated_at,
            },
            {
                "name": "architecture",
                "version": architecture,
                "source": "env",
                "updated_at": None,
            },
            {
                "name": "shadow_version",
                "version": shadow_version,
                "source": "env",
                "updated_at": None,
            },
            {
                "name": "shadow_architecture",
                "version": shadow_architecture,
                "source": "env",
                "updated_at": None,
            },
        ]

        payload = {"models": models}

        # Update cache
        _settings_models_cache.set({"generated_at": generated_at, "payload": payload})

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] model_versions.json error: {e}")
        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": {
                "models": [],
                "status": "degraded",
                "error": str(e)[:100],
            },
        }


# =============================================================================
# IA FEATURES SETTINGS (dynamic LLM configuration)
# =============================================================================

_ia_features_cache = SimpleCache(ttl=30)


@router.get("/ia-features.json")
async def dashboard_settings_ia_features_get(request: Request):
    """
    Get IA Features configuration.

    Auth: X-Dashboard-Token required.
    TTL: 30s cache (short for config changes).

    Returns:
      - narratives_enabled: bool | null (null = inherit from env)
      - narrative_feedback_enabled: bool (placeholder for Phase 2)
      - primary_model: str (model key from LLM_MODELS)
      - temperature: float
      - max_tokens: int
      - available_models: list of model info with pricing
    """
    _check_token(request)

    generated_at = datetime.now(timezone.utc).isoformat()

    # Check cache
    hit, cached = _ia_features_cache.get()
    if hit:
        return {
            "generated_at": cached["generated_at"],
            "cached": True,
            "cache_age_seconds": round(_ia_features_cache.age, 1),
            "data": cached["payload"],
        }

    try:
        from app.config import get_ia_features_config, LLM_MODELS

        async with AsyncSessionLocal() as session:
            ia_config = await get_ia_features_config(session)

        # Build available_models list from catalog
        available_models = [
            {
                "id": model_id,
                "display_name": info["display_name"],
                "provider": info["provider"],
                "input_price": info["input_price_per_1m"],
                "output_price": info["output_price_per_1m"],
                "max_tokens": info["max_tokens"],
            }
            for model_id, info in LLM_MODELS.items()
        ]

        # Compute effective state (for UI display)
        effective_enabled = ia_config.get("narratives_enabled")
        if effective_enabled is None:
            effective_enabled = settings.FASTPATH_ENABLED

        payload = {
            **ia_config,
            "effective_enabled": effective_enabled,  # Resolved value after inheritance
            "env_fastpath_enabled": settings.FASTPATH_ENABLED,  # For "Inherit" display
            "available_models": available_models,
        }

        # Update cache
        _ia_features_cache.set({"generated_at": generated_at, "payload": payload})

        return {
            "generated_at": generated_at,
            "cached": False,
            "cache_age_seconds": 0,
            "data": payload,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] ia-features GET error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch IA features config")


@router.patch("/ia-features.json")
async def dashboard_settings_ia_features_patch(request: Request):
    """
    Update IA Features configuration.

    Auth: X-Dashboard-Token required.

    Allowed fields:
      - narratives_enabled: bool | null
      - primary_model: str (must be valid key in LLM_MODELS)
      - temperature: float (0.0 - 1.0)
      - max_tokens: int (100 - 131072)

    Note: narrative_feedback_enabled is read-only (Phase 2 placeholder).
    """
    _check_token(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    from app.config import get_ia_features_config, LLM_MODELS

    # Whitelist of updatable fields
    allowed_fields = {"narratives_enabled", "primary_model", "temperature", "max_tokens"}
    updates = {k: v for k, v in body.items() if k in allowed_fields}

    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    # Validate primary_model if provided
    if "primary_model" in updates:
        if updates["primary_model"] not in LLM_MODELS:
            valid_models = list(LLM_MODELS.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Valid options: {valid_models}"
            )

    # Validate temperature if provided
    if "temperature" in updates:
        temp = updates["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 1.0:
            raise HTTPException(status_code=400, detail="temperature must be 0.0 - 1.0")

    # Validate max_tokens if provided
    if "max_tokens" in updates:
        tokens = updates["max_tokens"]
        if not isinstance(tokens, int) or tokens < 100 or tokens > 131072:
            raise HTTPException(status_code=400, detail="max_tokens must be 100 - 131072")

    # Validate narratives_enabled (must be bool or null)
    if "narratives_enabled" in updates:
        val = updates["narratives_enabled"]
        if val is not None and not isinstance(val, bool):
            raise HTTPException(status_code=400, detail="narratives_enabled must be true, false, or null")

    try:
        async with AsyncSessionLocal() as session:
            # Get current config
            current = await get_ia_features_config(session)

            # Merge updates
            new_config = {**current, **updates}

            # Upsert ops_settings
            await session.execute(
                text("""
                    INSERT INTO ops_settings (key, value, updated_at, updated_by)
                    VALUES ('ia_features', :value, NOW(), 'dashboard')
                    ON CONFLICT (key) DO UPDATE SET
                        value = :value,
                        updated_at = NOW(),
                        updated_by = 'dashboard'
                """),
                {"value": json.dumps(new_config)}
            )
            await session.commit()

            # Invalidate cache
            _ia_features_cache.invalidate()

            logger.info(f"[SETTINGS] IA Features updated: {updates}")

            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data": new_config,
                "updated_fields": list(updates.keys()),
            }

    except Exception as e:
        logger.error(f"[SETTINGS] ia-features PATCH error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update IA features config")


# -----------------------------------------------------------------------------
# IA Features: Visibility Endpoints (Fase 2)
# -----------------------------------------------------------------------------


@router.get("/ia-features/prompt-template.json")
async def ia_features_prompt_template(request: Request):
    """
    Returns the current LLM prompt template for narrative generation.

    Auth: X-Dashboard-Token required.

    Returns:
    - version: Current prompt version (e.g., "v11")
    - prompt_template: Full prompt string (with placeholders)
    - char_count: Character count
    - notes: Description
    """
    _check_token(request)

    try:
        from app.llm.narrative_generator import build_narrative_prompt

        # Build prompt with dummy data to show template structure
        dummy_match_data = {
            "match_id": 0,
            "home_team": "{HOME_TEAM}",
            "away_team": "{AWAY_TEAM}",
            "home_team_id": None,
            "away_team_id": None,
            "league_name": "{LEAGUE}",
            "date": "{DATE}",
            "home_goals": 0,
            "away_goals": 0,
            "venue": {"name": "{VENUE}", "city": "{CITY}"},
            "stats": {
                "home": {"possession": "{POSS_H}", "shots": "{SHOTS_H}"},
                "away": {"possession": "{POSS_A}", "shots": "{SHOTS_A}"},
            },
            "prediction": {
                "selection": "{SELECTION}",
                "confidence": 0.0,
                "home_prob": 0.0,
                "draw_prob": 0.0,
                "away_prob": 0.0,
            },
            "events": [],
            "market_odds": {"home": 0.0, "draw": 0.0, "away": 0.0},
            "derived_facts": {},
            "narrative_style": {},
        }

        prompt, _, _ = build_narrative_prompt(dummy_match_data)

        return {
            "version": "v11",
            "prompt_template": prompt,
            "char_count": len(prompt),
            "notes": "Prompt v11 para narrativas post-partido. Placeholders marcados con {PLACEHOLDER}.",
        }

    except Exception as e:
        logger.error(f"[SETTINGS] prompt-template error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prompt template")


@router.get("/ia-features/preview/{match_id}.json")
async def ia_features_preview(
    request: Request,
    match_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Preview the LLM payload for a specific match (without calling the LLM).

    Auth: X-Dashboard-Token required.

    Returns:
    - match_id: Match ID
    - match_label: "Home vs Away"
    - prompt_preview: Full prompt that would be sent to LLM
    - match_data: Structured data used to build the prompt
    """
    _check_token(request)

    try:
        from sqlalchemy.orm import selectinload
        from app.llm.narrative_generator import build_narrative_prompt
        from app.etl.competitions import COMPETITIONS

        # Get match with teams
        result = await session.execute(
            select(Match)
            .options(selectinload(Match.home_team), selectinload(Match.away_team))
            .where(Match.id == match_id)
        )
        match = result.scalar_one_or_none()

        if not match:
            raise HTTPException(status_code=404, detail=f"Match {match_id} not found")

        # Get prediction
        pred_result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        prediction = pred_result.scalar_one_or_none()

        # Get odds from odds_history
        odds_result = await session.execute(
            select(OddsHistory)
            .where(OddsHistory.match_id == match_id)
            .order_by(OddsHistory.recorded_at.desc())
            .limit(1)
        )
        odds_row = odds_result.scalar_one_or_none()

        # Build match_data dict
        home_name = match.home_team.name if match.home_team else "Local"
        away_name = match.away_team.name if match.away_team else "Visitante"

        # Stats come from match.stats JSON field
        stats_dict = {"home": {}, "away": {}}
        if match.stats:
            stats_dict = match.stats

        # Events come from match.events JSON field (limit to 10)
        events_list = []
        if match.events:
            events_list = match.events[:10]

        # Market odds from odds_history or match.odds_*
        market_odds = {}
        if odds_row:
            market_odds = {
                "home": odds_row.odds_home,
                "draw": odds_row.odds_draw,
                "away": odds_row.odds_away,
            }
        elif match.odds_home:
            market_odds = {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            }

        prediction_dict = {}
        if prediction:
            # Calculate selection and confidence from probabilities
            probs = {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            selection = max(probs, key=probs.get)
            confidence = probs[selection]
            prediction_dict = {
                "selection": selection,
                "confidence": confidence,
                "home_prob": prediction.home_prob,
                "draw_prob": prediction.draw_prob,
                "away_prob": prediction.away_prob,
            }

        # Get league name: display_name from admin_leagues > COMPETITIONS fallback
        from app.dashboard.admin import get_league_info_sync
        league_info_cached = get_league_info_sync(match.league_id)
        league_name = league_info_cached.get("name", "") if league_info_cached else ""
        if not league_name:
            comp_info = COMPETITIONS.get(match.league_id)
            if comp_info:
                league_name = comp_info.name or ""

        match_data = {
            "match_id": match.id,
            "home_team": home_name,
            "away_team": away_name,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "league_name": league_name,
            "date": match.date.isoformat() if match.date else "",
            "home_goals": match.home_goals or 0,
            "away_goals": match.away_goals or 0,
            "venue": {"name": match.venue_name, "city": match.venue_city} if match.venue_name else {},
            "stats": stats_dict,
            "prediction": prediction_dict,
            "events": events_list,
            "market_odds": market_odds,
            "derived_facts": {},
            "narrative_style": {},
        }

        # Build prompt
        prompt, _, _ = build_narrative_prompt(match_data)

        return {
            "match_id": match.id,
            "match_label": f"{home_name} vs {away_name}",
            "status": match.status,
            "prompt_preview": prompt,
            "match_data": match_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SETTINGS] preview error for match {match_id}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate preview")


@router.get("/ia-features/call-history.json")
async def ia_features_call_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Returns recent LLM narrative generation calls from post_match_audits.

    Auth: X-Dashboard-Token required.

    Query params:
    - limit: Max items to return (default 20, max 100)

    Returns:
    - items: List of recent narrative generations with metrics
    - total: Total count of narratives generated
    """
    _check_token(request)

    try:
        from app.config import LLM_MODELS

        # Query recent audits with narratives
        query = (
            select(
                PostMatchAudit.outcome_id,
                PostMatchAudit.llm_narrative_model,
                PostMatchAudit.llm_narrative_tokens_in,
                PostMatchAudit.llm_narrative_tokens_out,
                PostMatchAudit.llm_narrative_delay_ms,
                PostMatchAudit.llm_narrative_exec_ms,
                PostMatchAudit.llm_narrative_generated_at,
                PostMatchAudit.llm_narrative_status,
                PostMatchAudit.llm_prompt_version,
                Match.id.label("match_id"),
                Team.name.label("home_team_name"),
            )
            .join(PredictionOutcome, PostMatchAudit.outcome_id == PredictionOutcome.id)
            .join(Match, PredictionOutcome.match_id == Match.id)
            .join(Team, Match.home_team_id == Team.id)
            .where(PostMatchAudit.llm_narrative_generated_at.isnot(None))
            .order_by(PostMatchAudit.llm_narrative_generated_at.desc())
            .limit(limit)
        )

        result = await session.execute(query)
        rows = result.all()

        # Get away team names separately
        match_ids = [r.match_id for r in rows]
        away_query = (
            select(Match.id, Team.name.label("away_team_name"))
            .join(Team, Match.away_team_id == Team.id)
            .where(Match.id.in_(match_ids))
        )
        away_result = await session.execute(away_query)
        away_names = {r.id: r.away_team_name for r in away_result.all()}

        # Count total
        count_query = select(func.count()).select_from(PostMatchAudit).where(
            PostMatchAudit.llm_narrative_generated_at.isnot(None)
        )
        total = (await session.execute(count_query)).scalar() or 0

        items = []
        for row in rows:
            # Calculate cost
            model_key = row.llm_narrative_model or "gemini-2.5-flash-lite"
            model_info = LLM_MODELS.get(model_key, LLM_MODELS.get("gemini-2.5-flash-lite", {}))
            tokens_in = row.llm_narrative_tokens_in or 0
            tokens_out = row.llm_narrative_tokens_out or 0
            cost_usd = (
                (tokens_in * model_info.get("input_price_per_1m", 0.10) / 1_000_000)
                + (tokens_out * model_info.get("output_price_per_1m", 0.40) / 1_000_000)
            )

            away_name = away_names.get(row.match_id, "Visitante")

            items.append({
                "match_id": row.match_id,
                "match_label": f"{row.home_team_name} vs {away_name}",
                "generated_at": row.llm_narrative_generated_at.isoformat() if row.llm_narrative_generated_at else None,
                "model": row.llm_narrative_model,
                "prompt_version": row.llm_prompt_version,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": row.llm_narrative_delay_ms,
                "exec_ms": row.llm_narrative_exec_ms,
                "cost_usd": round(cost_usd, 6),
                "status": row.llm_narrative_status or "success",
                "audit_url": f"/dashboard/ops/llm_audit/{row.match_id}.json",
            })

        return {
            "items": items,
            "total": total,
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"[SETTINGS] call-history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get call history")


# Rate limiting for playground (in-memory, resets on restart)
_playground_rate_limits: dict[str, list[datetime]] = {}


def _check_playground_rate_limit(token: str) -> tuple[bool, int, datetime]:
    """
    Check rate limit for playground endpoint.

    Returns: (allowed, remaining, reset_at)
    """
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)

    # Clean old calls
    calls = _playground_rate_limits.get(token, [])
    calls = [c for c in calls if c > hour_ago]

    if len(calls) >= 10:
        reset_at = calls[0] + timedelta(hours=1)
        return False, 0, reset_at

    calls.append(now)
    _playground_rate_limits[token] = calls
    return True, 10 - len(calls), now + timedelta(hours=1)


class PlaygroundRequest(BaseModel):
    """Request body for playground endpoint."""

    match_id: int = Field(..., description="Match ID to generate narrative for")
    temperature: float | None = Field(default=None, ge=0.0, le=1.0, description="Temperature (0.0-1.0)")
    max_tokens: int | None = Field(default=None, ge=100, le=131072, description="Max tokens")
    model: str | None = Field(default=None, description="Model to use")


@router.post("/ia-features/playground")
async def ia_features_playground(
    request: Request,
    body: PlaygroundRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    LLM Playground: Generate a narrative for a match with custom parameters.

    Auth: X-Dashboard-Token required.
    Rate limit: 10 calls/hour per token.

    This endpoint actually calls the LLM and incurs costs.
    Narratives generated here are NOT persisted to the database.
    """
    _check_token(request)

    try:
        from sqlalchemy.orm import selectinload
        from app.llm.narrative_generator import build_narrative_prompt
        from app.config import LLM_MODELS, get_ia_features_config
        from app.etl.competitions import COMPETITIONS
        import time as time_module

        # Get token for rate limiting
        token = request.headers.get("X-Dashboard-Token", "anonymous")

        # Check rate limit
        allowed, remaining, reset_at = _check_playground_rate_limit(token)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded (10 calls/hour)",
                    "rate_limit": {
                        "remaining": 0,
                        "reset_at": reset_at.isoformat(),
                    },
                },
            )

        # Validate match exists
        result = await session.execute(
            select(Match)
            .options(selectinload(Match.home_team), selectinload(Match.away_team))
            .where(Match.id == body.match_id)
        )
        match = result.scalar_one_or_none()

        if not match:
            raise HTTPException(status_code=404, detail=f"Match {body.match_id} not found")

        # Validate match is finished
        if match.status not in ("FT", "AET", "PEN"):
            raise HTTPException(
                status_code=400,
                detail=f"Match must be finished (status: {match.status})"
            )

        # Validate match has stats
        if not match.stats:
            raise HTTPException(
                status_code=400,
                detail="Match has no stats available"
            )

        # Get prediction
        pred_result = await session.execute(
            select(Prediction)
            .where(Prediction.match_id == body.match_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        prediction = pred_result.scalar_one_or_none()

        # Get odds
        odds_result = await session.execute(
            select(OddsHistory)
            .where(OddsHistory.match_id == body.match_id)
            .order_by(OddsHistory.recorded_at.desc())
            .limit(1)
        )
        odds_row = odds_result.scalar_one_or_none()

        # Build match_data dict
        home_name = match.home_team.name if match.home_team else "Local"
        away_name = match.away_team.name if match.away_team else "Visitante"

        events_list = match.events[:10] if match.events else []

        market_odds = {}
        if odds_row:
            market_odds = {
                "home": odds_row.odds_home,
                "draw": odds_row.odds_draw,
                "away": odds_row.odds_away,
            }
        elif match.odds_home:
            market_odds = {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away,
            }

        prediction_dict = {}
        if prediction:
            # Calculate selection and confidence from probabilities
            probs = {
                "home": prediction.home_prob,
                "draw": prediction.draw_prob,
                "away": prediction.away_prob,
            }
            selection = max(probs, key=probs.get)
            confidence = probs[selection]
            prediction_dict = {
                "selection": selection,
                "confidence": confidence,
                "home_prob": prediction.home_prob,
                "draw_prob": prediction.draw_prob,
                "away_prob": prediction.away_prob,
            }

        # Get league name: display_name from admin_leagues > COMPETITIONS fallback
        from app.dashboard.admin import get_league_info_sync
        league_info_cached = get_league_info_sync(match.league_id)
        league_name = league_info_cached.get("name", "") if league_info_cached else ""
        if not league_name:
            comp_info = COMPETITIONS.get(match.league_id)
            if comp_info:
                league_name = comp_info.name or ""

        match_data = {
            "match_id": match.id,
            "home_team": home_name,
            "away_team": away_name,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "league_name": league_name,
            "date": match.date.isoformat() if match.date else "",
            "home_goals": match.home_goals or 0,
            "away_goals": match.away_goals or 0,
            "venue": {"name": match.venue_name, "city": match.venue_city} if match.venue_name else {},
            "stats": match.stats,
            "prediction": prediction_dict,
            "events": events_list,
            "market_odds": market_odds,
            "derived_facts": {},
            "narrative_style": {},
        }

        # Get current config for defaults
        ia_config = await get_ia_features_config(session)

        # Resolve parameters
        model_to_use = body.model or ia_config.get("primary_model", "gemini-2.5-flash-lite")
        temperature_to_use = body.temperature if body.temperature is not None else ia_config.get("temperature", 0.7)
        max_tokens_to_use = body.max_tokens or ia_config.get("max_tokens", 4096)

        # Validate model exists
        if model_to_use not in LLM_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_to_use}"
            )

        model_info = LLM_MODELS[model_to_use]

        # Build prompt
        prompt, _, _ = build_narrative_prompt(match_data)

        # Generate narrative using GeminiClient directly
        # (Playground only supports Gemini models for now)
        from app.llm.gemini_client import GeminiClient
        from app.llm.narrative_generator import parse_json_response

        start_time = time_module.time()

        # Create Gemini client and generate
        client = GeminiClient()
        try:
            result = await client.generate(
                prompt=prompt,
                max_tokens=max_tokens_to_use,
                temperature=temperature_to_use,
            )
        finally:
            await client.close()

        latency_ms = int((time_module.time() - start_time) * 1000)

        if result.status != "COMPLETED":
            raise HTTPException(
                status_code=500,
                detail=f"LLM generation failed: {result.error or result.status}"
            )

        tokens_in = result.tokens_in
        tokens_out = result.tokens_out

        parsed = parse_json_response(result.text)

        if not parsed:
            raise HTTPException(
                status_code=500,
                detail="Failed to parse LLM response"
            )

        # Calculate cost
        cost_usd = (
            (tokens_in * model_info.get("input_price_per_1m", 0.10) / 1_000_000)
            + (tokens_out * model_info.get("output_price_per_1m", 0.40) / 1_000_000)
        )

        # Extract narrative
        narrative_data = parsed.get("narrative", {})

        return {
            "narrative": {
                "title": narrative_data.get("title", ""),
                "body": narrative_data.get("body", ""),
                "key_factors": narrative_data.get("key_factors", []),
            },
            "model_used": model_to_use,
            "metrics": {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "cost_usd": round(cost_usd, 6),
            },
            "warnings": [],
            "rate_limit": {
                "remaining": remaining,
                "reset_at": reset_at.isoformat(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SETTINGS] playground error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate narrative")
