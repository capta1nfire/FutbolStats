"""
Migration 022: Add LLM traceability columns to post_match_audits.

Enables debugging LLM hallucinations by storing:
- The exact payload sent to the LLM
- Hash for quick comparison
- Prompt version for tracking prompt changes
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def migrate():
    """Add LLM traceability columns."""
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        logger.info("Adding LLM traceability columns to post_match_audits...")

        columns = [
            ("llm_prompt_version", "VARCHAR(20)", "Prompt template version"),
            ("llm_prompt_input_json", "JSONB", "Sanitized payload sent to LLM"),
            ("llm_prompt_input_hash", "VARCHAR(64)", "SHA256 of canonicalized input"),
            ("llm_output_raw", "TEXT", "Raw LLM output before parsing"),
            ("llm_validation_errors", "JSONB", "List of claim validation errors"),
        ]

        for col_name, col_type, description in columns:
            try:
                await session.execute(text(f"""
                    ALTER TABLE post_match_audits
                    ADD COLUMN IF NOT EXISTS {col_name} {col_type}
                """))
                logger.info(f"Added column {col_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Column {col_name} already exists, skipping")
                else:
                    raise

        # Create index on hash for quick lookups
        logger.info("Creating indexes...")
        try:
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_pma_prompt_hash
                ON post_match_audits(llm_prompt_input_hash)
                WHERE llm_prompt_input_hash IS NOT NULL
            """))
            logger.info("Created index on llm_prompt_input_hash")
        except Exception as e:
            logger.warning(f"Index creation note: {e}")

        await session.commit()
        logger.info("Migration 022 completed successfully")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())
