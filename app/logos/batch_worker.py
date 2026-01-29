"""Batch Worker for Logo Generation.

Processes teams liga-by-liga with pause/resume/cancel support.
Coordinates IA generation, validation, and thumbnail creation.

Usage:
    # Start batch for a league
    batch_id = await start_batch_job(
        session=session,
        league_id=239,
        generation_mode="full_3d",
        ia_model="dall-e-3"
    )

    # Process next batch of teams
    await process_batch(session, batch_id, batch_size=5)

    # Pause/resume
    await pause_batch(session, batch_id)
    await resume_batch(session, batch_id)
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Team, TeamLogo, LogoBatchJob, LogoPromptTemplate
from app.logos.config import get_logos_settings
from app.logos.r2_client import get_logos_r2_client
from app.logos.ia_generator import get_ia_generator
from app.logos.validator import validate_ia_output, should_retry
from app.logos.processor import process_logo_thumbnails
from app.logos.cdn import invalidate_team_logo_cdn

logger = logging.getLogger(__name__)
logos_settings = get_logos_settings()


# =============================================================================
# Rate Limiter for IA APIs (Kimi recommendation)
# =============================================================================


class IARateLimiter:
    """Simple rate limiter for IA API calls.

    Tracks requests per minute and enforces limits per model.
    """

    def __init__(self):
        self._requests: dict[str, list[float]] = {}
        self._limits = {
            "dall-e-3": logos_settings.LOGOS_DALLE_RPM,
            "sdxl": logos_settings.LOGOS_SDXL_RPM,
            "imagen-3": logos_settings.LOGOS_IMAGEN_RPM,
            "imagen-4": logos_settings.LOGOS_IMAGEN_RPM,
            "gemini": logos_settings.LOGOS_IMAGEN_RPM,
        }

    async def acquire(self, model: str) -> None:
        """Wait until rate limit allows another request.

        Args:
            model: IA model name
        """
        limit = self._limits.get(model, 5)
        now = time.time()

        if model not in self._requests:
            self._requests[model] = []

        # Remove requests older than 60 seconds
        self._requests[model] = [t for t in self._requests[model] if now - t < 60]

        # If at limit, wait
        while len(self._requests[model]) >= limit:
            oldest = self._requests[model][0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                logger.info(f"Rate limit reached for {model}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            now = time.time()
            self._requests[model] = [t for t in self._requests[model] if now - t < 60]

        # Record this request
        self._requests[model].append(now)


# Global rate limiter instance
_ia_rate_limiter = IARateLimiter()


# =============================================================================
# Cost Guard (Kimi recommendation)
# =============================================================================


def estimate_batch_cost(team_count: int, generation_mode: str, ia_model: str) -> float:
    """Estimate cost for a batch job.

    Args:
        team_count: Number of teams to process
        generation_mode: full_3d, facing_only, front_only
        ia_model: IA model to use

    Returns:
        Estimated cost in USD
    """
    images_per_team = {
        "full_3d": 3,
        "facing_only": 2,
        "front_only": 1,
        "manual": 0,
    }.get(generation_mode, 3)

    cost_per_image = {
        "dall-e-3": logos_settings.LOGOS_COST_PER_IMAGE_DALLE,
        "sdxl": logos_settings.LOGOS_COST_PER_IMAGE_SDXL,
        "imagen-3": logos_settings.LOGOS_COST_PER_IMAGE_IMAGEN,
        "imagen-4": logos_settings.LOGOS_COST_PER_IMAGE_IMAGEN,
        "gemini": logos_settings.LOGOS_COST_PER_IMAGE_IMAGEN,
    }.get(ia_model, 0.03)

    return team_count * images_per_team * cost_per_image


def validate_batch_cost(team_count: int, generation_mode: str, ia_model: str) -> tuple[bool, float, str]:
    """Validate that batch cost is within limits.

    Args:
        team_count: Number of teams
        generation_mode: Generation mode
        ia_model: IA model

    Returns:
        Tuple of (valid, estimated_cost, error_message)
    """
    estimated_cost = estimate_batch_cost(team_count, generation_mode, ia_model)
    max_cost = logos_settings.LOGOS_MAX_BATCH_COST_USD

    if estimated_cost > max_cost:
        return (
            False,
            estimated_cost,
            f"Estimated cost ${estimated_cost:.2f} exceeds limit ${max_cost:.2f}. "
            f"Reduce team count or use a cheaper model.",
        )

    return True, estimated_cost, ""


async def get_active_prompts(
    session: AsyncSession,
    version: str = "v1",
) -> dict[str, str]:
    """Get active prompts for each variant.

    Args:
        session: Database session
        version: Prompt version

    Returns:
        Dict mapping variant -> prompt_template
    """
    result = await session.execute(
        select(LogoPromptTemplate)
        .where(
            and_(
                LogoPromptTemplate.version == version,
                LogoPromptTemplate.is_active == True,
            )
        )
    )
    prompts = result.scalars().all()

    return {p.variant: p.prompt_template for p in prompts}


async def start_batch_job(
    session: AsyncSession,
    league_id: int,
    generation_mode: str = "full_3d",
    ia_model: str = "dall-e-3",
    prompt_version: str = "v1",
    started_by: Optional[str] = None,
) -> UUID:
    """Start a new batch generation job for a league.

    Args:
        session: Database session
        league_id: League ID to process
        generation_mode: full_3d, facing_only, front_only, manual
        ia_model: IA model to use
        prompt_version: Prompt version
        started_by: User who started the job

    Returns:
        Batch job UUID
    """
    # Get prompts for this version
    prompts = await get_active_prompts(session, prompt_version)

    # Count teams in league that need processing
    # Note: This assumes teams are linked to leagues via matches
    result = await session.execute(
        select(Team.id)
        .join(TeamLogo, Team.id == TeamLogo.team_id, isouter=True)
        .where(
            TeamLogo.status.in_(["pending", "error"])
            | (TeamLogo.team_id == None)
        )
        # TODO: Filter by league_id via matches or admin_leagues relationship
    )
    team_ids = [row[0] for row in result.fetchall()]

    if not team_ids:
        raise ValueError(f"No teams pending processing for league {league_id}")

    # Cost Guard validation (Kimi recommendation)
    cost_valid, estimated_cost, cost_error = validate_batch_cost(
        team_count=len(team_ids),
        generation_mode=generation_mode,
        ia_model=ia_model,
    )
    if not cost_valid:
        raise ValueError(cost_error)

    # Create batch job
    batch = LogoBatchJob(
        ia_model=ia_model,
        generation_mode=generation_mode,
        prompt_front=prompts.get("front"),
        prompt_right=prompts.get("right"),
        prompt_left=prompts.get("left"),
        prompt_version=prompt_version,
        entity_type="league",
        league_id=league_id,
        total_teams=len(team_ids),
        team_ids=team_ids,
        status="running",
        estimated_cost_usd=estimated_cost,
        started_by=started_by,
    )

    session.add(batch)
    await session.commit()
    await session.refresh(batch)

    logger.info(
        f"Started batch job {batch.id} for league {league_id}: "
        f"{len(team_ids)} teams, mode={generation_mode}, model={ia_model}"
    )

    return batch.id


async def process_team_logo(
    session: AsyncSession,
    team_id: int,
    batch_job: LogoBatchJob,
    original_bytes: bytes,
) -> tuple[bool, Optional[str]]:
    """Process a single team's logo generation.

    Args:
        session: Database session
        team_id: Team ID to process
        batch_job: Batch job for configuration
        original_bytes: Original logo image bytes

    Returns:
        Tuple of (success, error_message)
    """
    r2_client = get_logos_r2_client()
    if not r2_client:
        return False, "R2 client not configured"

    generator = get_ia_generator(batch_job.ia_model)
    if not generator:
        return False, f"IA generator {batch_job.ia_model} not available"

    # Get or create TeamLogo record
    result = await session.execute(
        select(TeamLogo).where(TeamLogo.team_id == team_id)
    )
    team_logo = result.scalar_one_or_none()

    if not team_logo:
        team_logo = TeamLogo(
            team_id=team_id,
            status="processing",
            batch_job_id=batch_job.id,
            generation_mode=batch_job.generation_mode,
            ia_model=batch_job.ia_model,
            ia_prompt_version=batch_job.prompt_version,
            processing_started_at=datetime.utcnow(),
        )
        session.add(team_logo)
    else:
        team_logo.status = "processing"
        team_logo.batch_job_id = batch_job.id
        team_logo.processing_started_at = datetime.utcnow()

    await session.commit()

    # Upload original
    original_key = await r2_client.upload_team_logo(
        team_id, "original", original_bytes
    )
    if not original_key:
        team_logo.status = "error"
        team_logo.error_message = "Failed to upload original"
        team_logo.error_phase = "upload"
        await session.commit()
        return False, "Failed to upload original"

    team_logo.r2_key_original = original_key
    team_logo.uploaded_at = datetime.utcnow()

    # Determine which variants to generate
    variants = []
    if batch_job.generation_mode == "full_3d":
        variants = [
            ("front_3d", batch_job.prompt_front),
            ("facing_right", batch_job.prompt_right),
            ("facing_left", batch_job.prompt_left),
        ]
    elif batch_job.generation_mode == "front_only":
        variants = [("front_3d", batch_job.prompt_front)]
    elif batch_job.generation_mode == "facing_only":
        # Use original as front, only generate facing variants
        team_logo.r2_key_front = original_key
        team_logo.use_original_as_front = True
        variants = [
            ("facing_right", batch_job.prompt_right),
            ("facing_left", batch_job.prompt_left),
        ]

    generated_variants: dict[str, bytes] = {}
    total_cost = 0.0

    # Generate each variant
    for variant_name, prompt in variants:
        if not prompt:
            logger.warning(f"No prompt for variant {variant_name}, skipping")
            continue

        # Generate with retries
        retry_count = 0
        generated_bytes = None

        while retry_count < logos_settings.LOGOS_IA_MAX_RETRIES:
            try:
                # Rate limit IA API calls (Kimi recommendation)
                await _ia_rate_limiter.acquire(batch_job.ia_model)

                generated_bytes = await generator.generate(
                    original_bytes, prompt
                )

                if generated_bytes:
                    # Validate
                    validation = validate_ia_output(generated_bytes, variant_name)

                    if validation.valid:
                        break
                    elif should_retry(validation, retry_count):
                        logger.warning(
                            f"Validation failed for {variant_name}, retrying: {validation.errors}"
                        )
                        retry_count += 1
                        await asyncio.sleep(logos_settings.LOGOS_IA_RETRY_DELAY_SECONDS)
                    else:
                        team_logo.status = "error"
                        team_logo.error_message = f"Validation failed: {validation.errors}"
                        team_logo.error_phase = f"ia_{variant_name}"
                        team_logo.validation_errors = {"errors": validation.errors}
                        await session.commit()
                        return False, f"Validation failed for {variant_name}"
                else:
                    retry_count += 1
                    await asyncio.sleep(logos_settings.LOGOS_IA_RETRY_DELAY_SECONDS)

            except Exception as e:
                logger.error(f"IA generation error for {variant_name}: {e}")
                retry_count += 1
                if retry_count >= logos_settings.LOGOS_IA_MAX_RETRIES:
                    team_logo.status = "error"
                    team_logo.error_message = str(e)
                    team_logo.error_phase = f"ia_{variant_name}"
                    await session.commit()
                    return False, f"IA generation failed for {variant_name}"

        if generated_bytes:
            generated_variants[variant_name] = generated_bytes
            total_cost += 0.04  # DALL-E 3 HD estimate

    # Upload generated variants
    for variant_name, variant_bytes in generated_variants.items():
        r2_key = await r2_client.upload_team_logo(
            team_id, variant_name, variant_bytes
        )
        if not r2_key:
            team_logo.status = "error"
            team_logo.error_message = f"Failed to upload {variant_name}"
            team_logo.error_phase = "upload"
            await session.commit()
            return False, f"Failed to upload {variant_name}"

        if variant_name == "front_3d":
            team_logo.r2_key_front = r2_key
        elif variant_name == "facing_right":
            team_logo.r2_key_right = r2_key
        elif variant_name == "facing_left":
            team_logo.r2_key_left = r2_key

    # Update status to pending_resize
    team_logo.status = "pending_resize"
    team_logo.processing_completed_at = datetime.utcnow()
    team_logo.ia_cost_usd = total_cost
    team_logo.retry_count = 0
    team_logo.error_message = None
    team_logo.error_phase = None

    await session.commit()

    logger.info(f"Team {team_id} logo generation completed, pending resize")
    return True, None


async def process_team_thumbnails(
    session: AsyncSession,
    team_id: int,
) -> tuple[bool, Optional[str]]:
    """Generate thumbnails for a team's logos.

    Args:
        session: Database session
        team_id: Team ID

    Returns:
        Tuple of (success, error_message)
    """
    r2_client = get_logos_r2_client()
    if not r2_client:
        return False, "R2 client not configured"

    result = await session.execute(
        select(TeamLogo).where(TeamLogo.team_id == team_id)
    )
    team_logo = result.scalar_one_or_none()

    if not team_logo or team_logo.status != "pending_resize":
        return False, "Team logo not in pending_resize status"

    urls: dict = {"front": {}, "right": {}, "left": {}}
    variants = [
        ("front_3d", team_logo.r2_key_front, "front"),
        ("facing_right", team_logo.r2_key_right, "right"),
        ("facing_left", team_logo.r2_key_left, "left"),
    ]

    for variant_name, r2_key, url_key in variants:
        if not r2_key:
            continue

        # Download variant
        image_bytes = await r2_client.get_object(r2_key)
        if not image_bytes:
            logger.warning(f"Could not download {variant_name} for team {team_id}")
            continue

        # Process thumbnails
        result = process_logo_thumbnails(image_bytes)
        if not result.success:
            logger.warning(f"Thumbnail processing failed for {variant_name}: {result.errors}")
            continue

        # Upload thumbnails
        for size, thumb in result.thumbnails.items():
            thumb_key = await r2_client.upload_team_thumbnail(
                team_id, variant_name, size, thumb.image_bytes
            )
            if thumb_key:
                cdn_url = f"{logos_settings.LOGOS_CDN_BASE_URL.rstrip('/')}/{thumb_key}"
                urls[url_key][str(size)] = cdn_url

    # Update team logo
    team_logo.urls = urls
    team_logo.status = "ready"
    team_logo.resize_completed_at = datetime.utcnow()
    team_logo.review_status = "pending"
    team_logo.updated_at = datetime.utcnow()

    await session.commit()

    # Invalidate CDN cache
    await invalidate_team_logo_cdn(team_id)

    logger.info(f"Team {team_id} thumbnails generated, status=ready")
    return True, None


async def process_batch(
    session: AsyncSession,
    batch_id: UUID,
    batch_size: int = 5,
) -> dict:
    """Process next batch of teams in a job.

    Args:
        session: Database session
        batch_id: Batch job UUID
        batch_size: Number of teams to process in this batch

    Returns:
        Progress dict with processed, failed, remaining counts
    """
    # Get batch job
    result = await session.execute(
        select(LogoBatchJob).where(LogoBatchJob.id == batch_id)
    )
    batch_job = result.scalar_one_or_none()

    if not batch_job:
        raise ValueError(f"Batch job {batch_id} not found")

    if batch_job.status not in ("running", "paused"):
        raise ValueError(f"Batch job {batch_id} is {batch_job.status}, cannot process")

    if batch_job.status == "paused":
        raise ValueError(f"Batch job {batch_id} is paused")

    # Get pending teams from this batch
    team_ids = batch_job.team_ids or []
    processed_so_far = batch_job.processed_teams

    teams_to_process = team_ids[processed_so_far : processed_so_far + batch_size]

    if not teams_to_process:
        # All done
        batch_job.status = "pending_review"
        batch_job.completed_at = datetime.utcnow()
        batch_job.updated_at = datetime.utcnow()
        await session.commit()
        return {
            "processed": 0,
            "failed": 0,
            "remaining": 0,
            "batch_status": "pending_review",
        }

    processed = 0
    failed = 0

    for team_id in teams_to_process:
        # Check if paused
        await session.refresh(batch_job)
        if batch_job.status == "paused":
            logger.info(f"Batch {batch_id} paused, stopping processing")
            break

        # Get original logo bytes
        # TODO: Implement actual original fetching (from fallback URL or upload)
        # For now, skip teams without originals
        result = await session.execute(
            select(Team).where(Team.id == team_id)
        )
        team = result.scalar_one_or_none()

        if not team or not team.logo_url:
            logger.warning(f"Team {team_id} has no logo URL, skipping")
            failed += 1
            continue

        # Fetch original from URL
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(team.logo_url, timeout=30.0)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch logo for team {team_id}")
                    failed += 1
                    continue
                original_bytes = response.content
        except Exception as e:
            logger.error(f"Error fetching logo for team {team_id}: {e}")
            failed += 1
            continue

        # Process team
        success, error = await process_team_logo(
            session, team_id, batch_job, original_bytes
        )

        if success:
            processed += 1
            batch_job.processed_images += (
                3 if batch_job.generation_mode == "full_3d" else 1
            )
        else:
            failed += 1
            logger.warning(f"Team {team_id} processing failed: {error}")

    # Update batch progress
    batch_job.processed_teams += processed + failed
    batch_job.failed_teams += failed
    batch_job.updated_at = datetime.utcnow()
    await session.commit()

    remaining = len(team_ids) - batch_job.processed_teams

    return {
        "processed": processed,
        "failed": failed,
        "remaining": remaining,
        "batch_status": batch_job.status,
    }


async def pause_batch(session: AsyncSession, batch_id: UUID) -> bool:
    """Pause a running batch job."""
    result = await session.execute(
        update(LogoBatchJob)
        .where(
            and_(
                LogoBatchJob.id == batch_id,
                LogoBatchJob.status == "running",
            )
        )
        .values(status="paused", paused_at=datetime.utcnow(), updated_at=datetime.utcnow())
    )
    await session.commit()
    return result.rowcount > 0


async def resume_batch(session: AsyncSession, batch_id: UUID) -> bool:
    """Resume a paused batch job."""
    result = await session.execute(
        update(LogoBatchJob)
        .where(
            and_(
                LogoBatchJob.id == batch_id,
                LogoBatchJob.status == "paused",
            )
        )
        .values(status="running", paused_at=None, updated_at=datetime.utcnow())
    )
    await session.commit()
    return result.rowcount > 0


async def cancel_batch(session: AsyncSession, batch_id: UUID) -> bool:
    """Cancel a batch job (cannot be resumed)."""
    result = await session.execute(
        update(LogoBatchJob)
        .where(
            and_(
                LogoBatchJob.id == batch_id,
                LogoBatchJob.status.in_(["running", "paused"]),
            )
        )
        .values(status="cancelled", updated_at=datetime.utcnow())
    )
    await session.commit()
    return result.rowcount > 0


async def get_batch_status(session: AsyncSession, batch_id: UUID) -> Optional[dict]:
    """Get detailed status of a batch job."""
    result = await session.execute(
        select(LogoBatchJob).where(LogoBatchJob.id == batch_id)
    )
    batch_job = result.scalar_one_or_none()

    if not batch_job:
        return None

    return {
        "id": str(batch_job.id),
        "status": batch_job.status,
        "ia_model": batch_job.ia_model,
        "generation_mode": batch_job.generation_mode,
        "league_id": batch_job.league_id,
        "progress": {
            "total_teams": batch_job.total_teams,
            "processed_teams": batch_job.processed_teams,
            "failed_teams": batch_job.failed_teams,
            "remaining": batch_job.total_teams - batch_job.processed_teams,
            "percent_complete": (
                round(batch_job.processed_teams / batch_job.total_teams * 100, 1)
                if batch_job.total_teams > 0
                else 0
            ),
        },
        "cost": {
            "estimated_usd": batch_job.estimated_cost_usd,
            "actual_usd": batch_job.actual_cost_usd,
        },
        "approval": {
            "status": batch_job.approval_status,
            "approved_count": batch_job.approved_count,
            "rejected_count": batch_job.rejected_count,
        },
        "timestamps": {
            "started_at": batch_job.started_at.isoformat() if batch_job.started_at else None,
            "paused_at": batch_job.paused_at.isoformat() if batch_job.paused_at else None,
            "completed_at": batch_job.completed_at.isoformat() if batch_job.completed_at else None,
        },
        "started_by": batch_job.started_by,
    }
