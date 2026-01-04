"""
Migration 001: Add audit system tables

Creates:
- prediction_outcomes: Links predictions to actual results
- post_match_audits: Detailed deviation analysis
- model_performance_logs: Weekly aggregated metrics

Run with: python scripts/migrations/001_add_audit_tables.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQL for creating the audit tables
CREATE_TABLES_SQL = """
-- Table: prediction_outcomes
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL UNIQUE REFERENCES predictions(id),
    match_id INTEGER NOT NULL REFERENCES matches(id),

    -- Actual result
    actual_result VARCHAR(10) NOT NULL,
    actual_home_goals INTEGER NOT NULL,
    actual_away_goals INTEGER NOT NULL,

    -- Predicted result
    predicted_result VARCHAR(10) NOT NULL,
    prediction_correct BOOLEAN NOT NULL,

    -- Confidence
    confidence FLOAT NOT NULL,
    confidence_tier VARCHAR(10) NOT NULL,

    -- xG data
    xg_home FLOAT,
    xg_away FLOAT,
    xg_diff FLOAT,

    -- Disruption factors
    had_red_card BOOLEAN DEFAULT FALSE,
    had_penalty BOOLEAN DEFAULT FALSE,
    had_var_decision BOOLEAN DEFAULT FALSE,
    red_card_minute INTEGER,

    -- Match stats
    home_possession FLOAT,
    total_shots_home INTEGER,
    total_shots_away INTEGER,
    shots_on_target_home INTEGER,
    shots_on_target_away INTEGER,

    audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_prediction_outcomes_prediction_id ON prediction_outcomes(prediction_id);
CREATE INDEX IF NOT EXISTS ix_prediction_outcomes_match_id ON prediction_outcomes(match_id);

-- Table: post_match_audits
CREATE TABLE IF NOT EXISTS post_match_audits (
    id SERIAL PRIMARY KEY,
    outcome_id INTEGER NOT NULL UNIQUE REFERENCES prediction_outcomes(id),

    -- Deviation classification
    deviation_type VARCHAR(20) NOT NULL,
    deviation_score FLOAT NOT NULL,

    -- Root cause
    primary_factor VARCHAR(50),
    secondary_factors JSONB,

    -- xG analysis
    xg_result_aligned BOOLEAN NOT NULL,
    xg_prediction_aligned BOOLEAN NOT NULL,
    goals_vs_xg_home FLOAT,
    goals_vs_xg_away FLOAT,

    -- Learning signals
    should_adjust_model BOOLEAN DEFAULT FALSE,
    adjustment_notes VARCHAR(500),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_post_match_audits_outcome_id ON post_match_audits(outcome_id);

-- Table: model_performance_logs
CREATE TABLE IF NOT EXISTS model_performance_logs (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    week_start TIMESTAMP NOT NULL,

    -- Prediction counts
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy FLOAT DEFAULT 0.0,

    -- By confidence tier
    gold_total INTEGER DEFAULT 0,
    gold_correct INTEGER DEFAULT 0,
    gold_accuracy FLOAT DEFAULT 0.0,

    silver_total INTEGER DEFAULT 0,
    silver_correct INTEGER DEFAULT 0,
    silver_accuracy FLOAT DEFAULT 0.0,

    copper_total INTEGER DEFAULT 0,
    copper_correct INTEGER DEFAULT 0,
    copper_accuracy FLOAT DEFAULT 0.0,

    -- Calibration metrics
    brier_score FLOAT,
    log_loss FLOAT,

    -- Deviation analysis
    anomaly_count INTEGER DEFAULT 0,
    anomaly_rate FLOAT DEFAULT 0.0,

    -- Value bet performance
    value_bets_placed INTEGER DEFAULT 0,
    value_bets_won INTEGER DEFAULT 0,
    value_bet_roi FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_version, week_start)
);

CREATE INDEX IF NOT EXISTS ix_model_performance_logs_model_version ON model_performance_logs(model_version);
CREATE INDEX IF NOT EXISTS ix_model_performance_logs_week_start ON model_performance_logs(week_start);
"""


async def run_migration():
    """Run the migration to create audit tables."""
    from app.database import async_engine

    logger.info("Starting migration: 001_add_audit_tables")

    async with async_engine.begin() as conn:
        # Split and execute each statement
        statements = [s.strip() for s in CREATE_TABLES_SQL.split(";") if s.strip()]

        for i, statement in enumerate(statements):
            try:
                await conn.execute(text(statement))
                logger.info(f"Executed statement {i + 1}/{len(statements)}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Statement {i + 1} skipped (already exists)")
                else:
                    logger.error(f"Error in statement {i + 1}: {e}")
                    raise

    logger.info("Migration completed successfully!")

    # Verify tables were created (database-agnostic check)
    async with async_engine.connect() as conn:
        try:
            # Try PostgreSQL query first
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('prediction_outcomes', 'post_match_audits', 'model_performance_logs')
            """))
            tables = [row[0] for row in result.fetchall()]
        except Exception:
            # Fallback for SQLite
            result = await conn.execute(text("""
                SELECT name FROM sqlite_master WHERE type='table'
                AND name IN ('prediction_outcomes', 'post_match_audits', 'model_performance_logs')
            """))
            tables = [row[0] for row in result.fetchall()]

        logger.info(f"Verified tables: {tables}")


if __name__ == "__main__":
    asyncio.run(run_migration())
