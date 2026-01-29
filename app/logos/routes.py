"""API Routes for 3D Logo Generation System.

Dashboard endpoints for logo management:
- Upload originals
- Generate by league
- Review and approve
- Batch job control

All endpoints require X-Dashboard-Token authentication.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Team, TeamLogo, CompetitionLogo, LogoBatchJob, LogoPromptTemplate
from app.database import get_async_session, async_session_maker
from app.logos.config import get_logos_settings
from app.logos.r2_client import get_logos_r2_client
from app.logos.validator import validate_original_logo
from app.logos.auth import verify_dashboard_token
from app.logos.batch_worker import (
    start_batch_job,
    process_batch,
    pause_batch,
    resume_batch,
    cancel_batch,
    get_batch_status,
    process_team_thumbnails,
    validate_batch_cost,
    generate_single_team,
)

logger = logging.getLogger(__name__)
logos_settings = get_logos_settings()

router = APIRouter(
    prefix="/dashboard/logos",
    tags=["logos"],
    dependencies=[Depends(verify_dashboard_token)],
)


# =============================================================================
# Pydantic Models
# =============================================================================


class GenerateBatchRequest(BaseModel):
    """Request to start batch generation."""

    generation_mode: str = "full_3d"  # full_3d, facing_only, front_only
    ia_model: str = "dall-e-3"
    prompt_version: str = "v1"


class ReviewTeamRequest(BaseModel):
    """Request to review a team's logos."""

    action: str  # approve, reject, regenerate
    notes: Optional[str] = None


class ReviewLeagueRequest(BaseModel):
    """Request to approve/reject entire league."""

    action: str  # approve_all, reject_all


class BatchControlRequest(BaseModel):
    """Request to control batch job."""

    action: str  # pause, resume, cancel


class TeamLogoResponse(BaseModel):
    """Response for team logo status."""

    team_id: int
    team_name: str
    status: str
    review_status: str
    urls: Optional[dict] = None
    fallback_url: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class BatchJobResponse(BaseModel):
    """Response for batch job status."""

    id: str
    status: str
    ia_model: str
    generation_mode: str
    league_id: Optional[int] = None
    progress: dict
    cost: dict
    approval: dict
    timestamps: dict
    started_by: Optional[str] = None


class LeagueLogosResponse(BaseModel):
    """Response for league logos overview."""

    league_id: int
    league_name: str
    total_teams: int
    status_counts: dict
    review_counts: dict


# =============================================================================
# Upload Endpoints
# =============================================================================


@router.post("/teams/{team_id}/upload")
async def upload_team_logo(
    team_id: int,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),  # Will be replaced with actual dependency
):
    """Upload original logo for a team.

    The logo will be validated and stored in R2 with immutable versioning.
    Format: teams/{internal_id}/{apifb_id}-{slug}_original_v{rev}.png
    Status will be set to 'pending' for IA generation.
    """
    # Validate team exists
    result = await session.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Read and validate file (auto-converts SVG to PNG, auto-pads non-square)
    content = await file.read()
    validation = validate_original_logo(content)

    if not validation.valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {', '.join(validation.errors)}",
        )

    # Use processed PNG (converted from SVG and/or padded to square)
    upload_bytes = validation.converted_bytes or content

    # Get or create TeamLogo record to determine revision
    # Use FOR UPDATE to prevent concurrent uploads from calculating same revision
    result = await session.execute(
        select(TeamLogo)
        .where(TeamLogo.team_id == team_id)
        .with_for_update()
    )
    team_logo = result.scalar_one_or_none()

    # Determine revision: increment if re-uploading (atomic due to row lock)
    revision = (team_logo.revision + 1) if team_logo else 1

    # Upload to R2 with immutable versioning
    r2_client = get_logos_r2_client()
    if not r2_client:
        raise HTTPException(status_code=503, detail="Storage not configured")

    r2_key = await r2_client.upload_team_logo(
        team_id=team_id,
        variant="original",
        image_bytes=upload_bytes,
        apifb_id=team.external_id,  # API-Football ID for traceability
        slug=team.name,  # Team name for readability
        revision=revision,
    )
    if not r2_key:
        raise HTTPException(status_code=500, detail="Failed to upload to storage")

    # Upload original SVG if converted (preserve vector for future use)
    r2_key_svg = None
    if validation.converted_from_svg and validation.original_svg_bytes:
        r2_key_svg = await r2_client.upload_team_logo(
            team_id=team_id,
            variant="original_svg",
            image_bytes=validation.original_svg_bytes,
            content_type="image/svg+xml",
            apifb_id=team.external_id,
            slug=team.name,
            revision=revision,
        )
        if r2_key_svg:
            logger.info(f"Preserved original SVG: {r2_key_svg}")

    # Update or create TeamLogo record
    if team_logo:
        team_logo.r2_key_original = r2_key
        team_logo.r2_key_original_svg = r2_key_svg
        team_logo.status = "pending"
        team_logo.uploaded_at = datetime.utcnow()
        team_logo.updated_at = datetime.utcnow()
        team_logo.fallback_url = team.logo_url
        team_logo.revision = revision
        # Clear previous variants (will be regenerated)
        team_logo.r2_key_front = None
        team_logo.r2_key_right = None
        team_logo.r2_key_left = None
        team_logo.urls = None
        team_logo.review_status = "pending"
    else:
        team_logo = TeamLogo(
            team_id=team_id,
            r2_key_original=r2_key,
            r2_key_original_svg=r2_key_svg,
            status="pending",
            uploaded_at=datetime.utcnow(),
            fallback_url=team.logo_url,
            revision=revision,
        )
        session.add(team_logo)

    await session.commit()

    return {
        "team_id": team_id,
        "status": "pending",
        "r2_key": r2_key,
        "r2_key_svg": r2_key_svg,  # Original SVG preserved (if uploaded SVG)
        "revision": revision,
        "validation": {
            "width": validation.width,
            "height": validation.height,
            "format": validation.format,
            "converted_from_svg": validation.converted_from_svg,
            "padded_to_square": validation.padded_to_square,
            "original_dimensions": validation.original_dimensions,
        },
    }


# =============================================================================
# Generation Endpoints
# =============================================================================


@router.get("/leagues")
async def list_leagues_for_generation(
    session: AsyncSession = Depends(get_async_session),
):
    """List leagues available for logo generation.

    Returns leagues with team counts and generation status.
    """
    # Get leagues with team counts from matches
    # This is a simplified query - in production would use admin_leagues
    result = await session.execute(
        select(
            func.distinct(Team.id).label("team_id"),
        )
        .limit(100)
    )

    # For now, return a placeholder
    # TODO: Implement proper league listing from admin_leagues
    return {
        "leagues": [
            {
                "league_id": 239,
                "league_name": "Colombia Primera A",
                "country": "Colombia",
                "total_teams": 20,
                "pending": 20,
                "ready": 0,
            },
            {
                "league_id": 71,
                "league_name": "Brazil Serie A",
                "country": "Brazil",
                "total_teams": 20,
                "pending": 20,
                "ready": 0,
            },
        ]
    }


@router.get("/teams/ready-for-test")
async def list_teams_ready_for_test(
    league_id: Optional[int] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List teams with original logo uploaded, ready for IA test generation.

    Returns teams that have r2_key_original (logo already uploaded).
    Useful for selecting a single team to test prompts before batch processing.

    Args:
        league_id: Optional filter by league (not implemented yet)
    """
    query = (
        select(TeamLogo, Team)
        .join(Team, TeamLogo.team_id == Team.id)
        .where(TeamLogo.r2_key_original.isnot(None))
        .order_by(Team.name)
        .limit(50)
    )

    result = await session.execute(query)
    rows = result.fetchall()

    teams = []
    for team_logo, team in rows:
        teams.append({
            "team_id": team.id,
            "team_name": team.name,
            "country": team.country,
            "external_id": team.external_id,
            "status": team_logo.status,
            "has_original": True,
            "has_variants": bool(team_logo.r2_key_front),
            "r2_key_original": team_logo.r2_key_original,
            "fallback_url": team_logo.fallback_url or team.logo_url,
        })

    return {
        "total": len(teams),
        "teams": teams,
    }


@router.post("/generate/league/{league_id}")
async def generate_league_logos(
    league_id: int,
    request: GenerateBatchRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Start batch generation for a league.

    Creates a batch job and returns the job ID for tracking.
    Rate limited: Only one running job per league allowed.
    """
    # Rate limiting: Check for existing running jobs on this league (Kimi recommendation)
    existing_job = await session.execute(
        select(LogoBatchJob)
        .where(
            LogoBatchJob.league_id == league_id,
            LogoBatchJob.status.in_(["running", "paused"]),
        )
    )
    if existing_job.scalar_one_or_none():
        raise HTTPException(
            status_code=429,
            detail=f"A batch job is already running/paused for league {league_id}. "
            "Cancel or complete it before starting a new one.",
        )

    try:
        batch_id = await start_batch_job(
            session=session,
            league_id=league_id,
            generation_mode=request.generation_mode,
            ia_model=request.ia_model,
            prompt_version=request.prompt_version,
            started_by="dashboard",  # TODO: Get from auth
        )

        return {
            "batch_id": str(batch_id),
            "status": "running",
            "message": f"Started batch generation for league {league_id}",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start batch job: {e}")
        raise HTTPException(status_code=500, detail="Failed to start batch job")


@router.post("/generate/batch/{batch_id}/process")
async def process_batch_teams(
    batch_id: UUID,
    batch_size: int = Query(default=5, ge=1, le=20),
    session: AsyncSession = Depends(get_async_session),
):
    """Process next batch of teams in a job.

    Call this repeatedly to process teams in batches.
    """
    try:
        result = await process_batch(session, batch_id, batch_size)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")


class GenerateSingleTeamRequest(BaseModel):
    """Request to generate 3D variants for a single team."""

    generation_mode: str = "full_3d"  # full_3d, facing_only, front_only
    ia_model: str = "imagen-3"
    prompt_version: str = "v1"


async def _generate_team_background(
    team_id: int,
    generation_mode: str,
    ia_model: str,
    prompt_version: str,
) -> None:
    """Background task for single team generation.

    Runs generation in background after endpoint returns 202.
    Uses its own session to avoid conflicts with the request session.
    """
    async with async_session_maker() as session:
        try:
            success, error, result = await generate_single_team(
                session=session,
                team_id=team_id,
                generation_mode=generation_mode,
                ia_model=ia_model,
                prompt_version=prompt_version,
            )

            if not success:
                logger.error(f"Background generation failed for team {team_id}: {error}")
            else:
                logger.info(f"Background generation completed for team {team_id}")

        except Exception as e:
            logger.error(f"Background generation exception for team {team_id}: {e}")
            # Update status to error
            try:
                result = await session.execute(
                    select(TeamLogo).where(TeamLogo.team_id == team_id)
                )
                team_logo = result.scalar_one_or_none()
                if team_logo:
                    team_logo.status = "error"
                    team_logo.error_message = str(e)
                    team_logo.error_phase = "background_task"
                    await session.commit()
            except Exception as inner_e:
                logger.error(f"Failed to update error status: {inner_e}")


@router.post("/generate/team/{team_id}", status_code=202)
async def generate_team_logo(
    team_id: int,
    request: GenerateSingleTeamRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """Generate 3D variants for a single team (async mode).

    Returns 202 Accepted immediately and processes in background.
    Poll GET /teams/{team_id}/status to check progress.

    Team must have an original logo uploaded (r2_key_original).
    """
    # Validate team exists
    team_result = await session.execute(
        select(Team).where(Team.id == team_id)
    )
    team = team_result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Validate team has original logo
    result = await session.execute(
        select(TeamLogo).where(TeamLogo.team_id == team_id)
    )
    team_logo = result.scalar_one_or_none()

    if not team_logo or not team_logo.r2_key_original:
        raise HTTPException(
            status_code=400,
            detail="Team has no original logo uploaded",
        )

    # Check if already processing
    if team_logo.status == "processing":
        return JSONResponse(
            status_code=202,
            content={
                "team_id": team_id,
                "team_name": team.name,
                "status": "processing",
                "message": "Generation already in progress. Poll /teams/{team_id}/status for updates.",
            },
        )

    # Update status to processing immediately
    team_logo.status = "processing"
    team_logo.processing_started_at = datetime.utcnow()
    team_logo.generation_mode = request.generation_mode
    team_logo.ia_model = request.ia_model
    team_logo.ia_prompt_version = request.prompt_version
    team_logo.error_message = None
    team_logo.error_phase = None
    await session.commit()

    # Launch background task
    background_tasks.add_task(
        _generate_team_background,
        team_id=team_id,
        generation_mode=request.generation_mode,
        ia_model=request.ia_model,
        prompt_version=request.prompt_version,
    )

    logger.info(f"Started async generation for team {team_id} ({team.name})")

    return JSONResponse(
        status_code=202,
        content={
            "team_id": team_id,
            "team_name": team.name,
            "status": "processing",
            "message": "Generation started. Poll /teams/{team_id}/status for updates.",
        },
    )


# =============================================================================
# Batch Control Endpoints
# =============================================================================


@router.get("/batch/{batch_id}")
async def get_batch_job_status(
    batch_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get detailed status of a batch job."""
    status = await get_batch_status(session, batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch job not found")
    return status


@router.post("/batch/{batch_id}/pause")
async def pause_batch_job(
    batch_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Pause a running batch job."""
    success = await pause_batch(session, batch_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot pause batch (not running)")
    return {"status": "paused", "batch_id": str(batch_id)}


@router.post("/batch/{batch_id}/resume")
async def resume_batch_job(
    batch_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Resume a paused batch job."""
    success = await resume_batch(session, batch_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot resume batch (not paused)")
    return {"status": "running", "batch_id": str(batch_id)}


@router.post("/batch/{batch_id}/cancel")
async def cancel_batch_job(
    batch_id: UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Cancel a batch job (cannot be resumed)."""
    success = await cancel_batch(session, batch_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel batch")
    return {"status": "cancelled", "batch_id": str(batch_id)}


# =============================================================================
# Review Endpoints
# =============================================================================


@router.get("/review/league/{league_id}")
async def get_league_review(
    league_id: int,
    status_filter: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """Get teams for review in a league.

    Returns teams with their generated logos for approval.
    """
    query = (
        select(TeamLogo, Team)
        .join(Team, TeamLogo.team_id == Team.id)
        .where(TeamLogo.status.in_(["ready", "pending_resize", "error"]))
    )

    if status_filter:
        query = query.where(TeamLogo.review_status == status_filter)

    result = await session.execute(query)
    rows = result.fetchall()

    teams = []
    for team_logo, team in rows:
        teams.append({
            "team_id": team.id,
            "team_name": team.name,
            "status": team_logo.status,
            "review_status": team_logo.review_status,
            "urls": team_logo.urls,
            "fallback_url": team_logo.fallback_url,
            "error_message": team_logo.error_message,
            "ia_model": team_logo.ia_model,
            "generation_mode": team_logo.generation_mode,
        })

    return {
        "league_id": league_id,
        "total": len(teams),
        "teams": teams,
    }


@router.post("/review/team/{team_id}")
async def review_team_logo(
    team_id: int,
    request: ReviewTeamRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Approve, reject, or request regeneration for a team's logos."""
    result = await session.execute(
        select(TeamLogo).where(TeamLogo.team_id == team_id)
    )
    team_logo = result.scalar_one_or_none()

    if not team_logo:
        raise HTTPException(status_code=404, detail="Team logo not found")

    now = datetime.utcnow()

    if request.action == "approve":
        team_logo.review_status = "approved"
        team_logo.reviewed_at = now
        team_logo.review_notes = request.notes
        team_logo.reviewed_by = "dashboard"  # TODO: Get from auth

    elif request.action == "reject":
        team_logo.review_status = "rejected"
        team_logo.reviewed_at = now
        team_logo.review_notes = request.notes
        team_logo.reviewed_by = "dashboard"

    elif request.action == "regenerate":
        team_logo.review_status = "needs_regeneration"
        team_logo.status = "pending"
        team_logo.review_notes = request.notes

    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")

    team_logo.updated_at = now
    await session.commit()

    # Update batch job approval counts if applicable
    if team_logo.batch_job_id:
        if request.action == "approve":
            await session.execute(
                update(LogoBatchJob)
                .where(LogoBatchJob.id == team_logo.batch_job_id)
                .values(approved_count=LogoBatchJob.approved_count + 1)
            )
        elif request.action == "reject":
            await session.execute(
                update(LogoBatchJob)
                .where(LogoBatchJob.id == team_logo.batch_job_id)
                .values(rejected_count=LogoBatchJob.rejected_count + 1)
            )
        await session.commit()

    return {
        "team_id": team_id,
        "review_status": team_logo.review_status,
        "action": request.action,
    }


@router.post("/review/league/{league_id}/approve")
async def approve_league_logos(
    league_id: int,
    request: ReviewLeagueRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Bulk approve or reject all pending logos in a league."""
    now = datetime.utcnow()

    if request.action == "approve_all":
        new_status = "approved"
    elif request.action == "reject_all":
        new_status = "rejected"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")

    # Update all pending reviews
    # Note: This would need to filter by league_id via batch_job relationship
    result = await session.execute(
        update(TeamLogo)
        .where(TeamLogo.review_status == "pending")
        .values(
            review_status=new_status,
            reviewed_at=now,
            reviewed_by="dashboard",
            updated_at=now,
        )
        .returning(TeamLogo.team_id)
    )
    updated_ids = [row[0] for row in result.fetchall()]
    await session.commit()

    return {
        "league_id": league_id,
        "action": request.action,
        "updated_count": len(updated_ids),
    }


# =============================================================================
# Status Endpoints
# =============================================================================


@router.get("/teams/{team_id}/status")
async def get_team_logo_status(
    team_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Get detailed status for a team's logos."""
    result = await session.execute(
        select(TeamLogo, Team)
        .join(Team, TeamLogo.team_id == Team.id)
        .where(TeamLogo.team_id == team_id)
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Team logo not found")

    team_logo, team = row

    return {
        "team_id": team.id,
        "team_name": team.name,
        "status": team_logo.status,
        "review_status": team_logo.review_status,
        "urls": team_logo.urls,
        "fallback_url": team_logo.fallback_url,
        "r2_keys": {
            "original": team_logo.r2_key_original,
            "front": team_logo.r2_key_front,
            "right": team_logo.r2_key_right,
            "left": team_logo.r2_key_left,
        },
        "generation": {
            "mode": team_logo.generation_mode,
            "ia_model": team_logo.ia_model,
            "prompt_version": team_logo.ia_prompt_version,
            "cost_usd": team_logo.ia_cost_usd,
        },
        "error": {
            "message": team_logo.error_message,
            "phase": team_logo.error_phase,
            "retry_count": team_logo.retry_count,
        },
        "timestamps": {
            "uploaded_at": team_logo.uploaded_at.isoformat() if team_logo.uploaded_at else None,
            "processing_started_at": team_logo.processing_started_at.isoformat() if team_logo.processing_started_at else None,
            "processing_completed_at": team_logo.processing_completed_at.isoformat() if team_logo.processing_completed_at else None,
            "resize_completed_at": team_logo.resize_completed_at.isoformat() if team_logo.resize_completed_at else None,
        },
    }


@router.get("/prompts")
async def list_prompt_templates(
    version: Optional[str] = None,
    include_full: bool = False,
    session: AsyncSession = Depends(get_async_session),
):
    """List available prompt templates.

    Args:
        version: Filter by version (v1, v2, etc.)
        include_full: Include full prompt text (default: False for list view)
    """
    query = select(LogoPromptTemplate).order_by(
        LogoPromptTemplate.version.desc(),
        LogoPromptTemplate.variant,
    )

    if version:
        query = query.where(LogoPromptTemplate.version == version)

    result = await session.execute(query)
    prompts = result.scalars().all()

    return {
        "prompts": [
            {
                "id": p.id,
                "version": p.version,
                "variant": p.variant,
                "prompt_template": p.prompt_template if include_full else (
                    p.prompt_template[:100] + "..." if len(p.prompt_template) > 100 else p.prompt_template
                ),
                "ia_model": p.ia_model,
                "is_active": p.is_active,
                "success_rate": p.success_rate,
                "usage_count": p.usage_count,
                "notes": p.notes,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in prompts
        ]
    }


@router.get("/prompts/{prompt_id}")
async def get_prompt_template(
    prompt_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a single prompt template with full text."""
    result = await session.execute(
        select(LogoPromptTemplate).where(LogoPromptTemplate.id == prompt_id)
    )
    prompt = result.scalar_one_or_none()

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    return {
        "id": prompt.id,
        "version": prompt.version,
        "variant": prompt.variant,
        "prompt_template": prompt.prompt_template,
        "ia_model": prompt.ia_model,
        "is_active": prompt.is_active,
        "success_rate": prompt.success_rate,
        "usage_count": prompt.usage_count,
        "notes": prompt.notes,
        "created_at": prompt.created_at.isoformat() if prompt.created_at else None,
    }


@router.put("/prompts/{prompt_id}")
async def update_prompt_template(
    prompt_id: int,
    data: dict,
    session: AsyncSession = Depends(get_async_session),
):
    """Update a prompt template.

    Allowed fields: prompt_template, is_active, notes, ia_model
    """
    result = await session.execute(
        select(LogoPromptTemplate).where(LogoPromptTemplate.id == prompt_id)
    )
    prompt = result.scalar_one_or_none()

    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    # Update allowed fields
    allowed_fields = {"prompt_template", "is_active", "notes", "ia_model"}
    for field, value in data.items():
        if field in allowed_fields and value is not None:
            setattr(prompt, field, value)

    await session.commit()
    await session.refresh(prompt)

    return {
        "id": prompt.id,
        "version": prompt.version,
        "variant": prompt.variant,
        "prompt_template": prompt.prompt_template,
        "ia_model": prompt.ia_model,
        "is_active": prompt.is_active,
        "notes": prompt.notes,
        "updated": True,
    }


# =============================================================================
# Thumbnail Processing Endpoint
# =============================================================================


@router.post("/teams/{team_id}/resize")
async def trigger_team_resize(
    team_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """Manually trigger thumbnail generation for a team.

    Normally this happens automatically via scheduler job.
    """
    success, error = await process_team_thumbnails(session, team_id)

    if not success:
        raise HTTPException(status_code=400, detail=error)

    return {
        "team_id": team_id,
        "status": "ready",
        "message": "Thumbnails generated successfully",
    }
