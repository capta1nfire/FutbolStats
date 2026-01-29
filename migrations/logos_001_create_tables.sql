-- =============================================================================
-- Migration: logos_001_create_tables.sql
-- Description: Create tables for 3D Logo Generation System
-- Date: 2026-01-28
-- Spec: docs/TEAM_LOGOS_3D_SPEC.md v3
-- =============================================================================

-- =============================================================================
-- Table: team_logos
-- Purpose: Track 3D logo variants for teams/national teams
-- =============================================================================

CREATE TABLE IF NOT EXISTS team_logos (
    team_id INTEGER PRIMARY KEY REFERENCES teams(id) ON DELETE CASCADE,

    -- =========================================================================
    -- R2 Storage Keys (original + 3 variants)
    -- =========================================================================
    r2_key_original VARCHAR(255),       -- logos/teams/{team_id}/original.png
    r2_key_front VARCHAR(255),          -- logos/teams/{team_id}/front_3d.png
    r2_key_right VARCHAR(255),          -- logos/teams/{team_id}/facing_right.png
    r2_key_left VARCHAR(255),           -- logos/teams/{team_id}/facing_left.png

    -- URLs de thumbnails (generadas post-resize)
    urls JSONB DEFAULT '{}',
    -- Structure:
    -- {
    --   "front": {"64": "https://...", "128": "...", "256": "...", "512": "..."},
    --   "right": {"64": "...", "128": "...", "256": "...", "512": "..."},
    --   "left":  {"64": "...", "128": "...", "256": "...", "512": "..."}
    -- }

    -- Fallback (API-Football URL original)
    fallback_url VARCHAR(500),

    -- =========================================================================
    -- Pipeline Status
    -- =========================================================================
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- Values:
    --   'pending'          = Original uploaded, waiting for IA generation
    --   'queued'           = In queue for IA batch
    --   'processing'       = IA generating images
    --   'pending_resize'   = IA completed, waiting for thumbnails
    --   'ready'            = All done
    --   'error'            = Failed (see error_message)
    --   'paused'           = Paused by user

    -- =========================================================================
    -- Processing Metadata
    -- =========================================================================
    batch_job_id UUID,                   -- Reference to batch job that processed it
    generation_mode VARCHAR(20),         -- 'full_3d', 'facing_only', 'front_only', 'manual'
    ia_model VARCHAR(50),                -- 'dall-e-3', 'sdxl', 'gemini', etc. (NULL if manual)
    ia_prompt_version VARCHAR(20),       -- 'v1', 'v2', etc. (for tracking)
    use_original_as_front BOOLEAN DEFAULT FALSE,  -- TRUE if facing_only or manual

    -- Timestamps
    uploaded_at TIMESTAMP,
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    resize_completed_at TIMESTAMP,

    -- Cost tracking
    ia_cost_usd DECIMAL(10,4),           -- Total IA cost (3 images)

    -- =========================================================================
    -- Error Handling
    -- =========================================================================
    error_message TEXT,
    error_phase VARCHAR(20),             -- 'upload', 'ia_front', 'ia_right', 'ia_left', 'resize'
    retry_count INTEGER DEFAULT 0,
    last_retry_at TIMESTAMP,

    -- =========================================================================
    -- Validation (Kimi consideration)
    -- =========================================================================
    validation_errors JSONB,             -- Errors from last validation
    last_validation_at TIMESTAMP,

    -- =========================================================================
    -- Review (Liga-by-liga approval)
    -- =========================================================================
    review_status VARCHAR(20) DEFAULT 'pending',
    -- Values: 'pending', 'approved', 'rejected', 'needs_regeneration'
    review_notes TEXT,
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP,

    -- =========================================================================
    -- Audit
    -- =========================================================================
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_team_logos_status ON team_logos(status)
    WHERE status NOT IN ('ready', 'error');
CREATE INDEX IF NOT EXISTS idx_team_logos_batch ON team_logos(batch_job_id)
    WHERE batch_job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_team_logos_review ON team_logos(review_status)
    WHERE review_status = 'pending';

-- =============================================================================
-- Table: competition_logos
-- Purpose: Track 3D logos for leagues/tournaments (only main variant, no facing)
-- =============================================================================

CREATE TABLE IF NOT EXISTS competition_logos (
    league_id INTEGER PRIMARY KEY REFERENCES admin_leagues(league_id) ON DELETE CASCADE,

    -- R2 Storage Keys
    r2_key_original VARCHAR(255),       -- logos/competitions/{league_id}/original.png
    r2_key_main VARCHAR(255),           -- logos/competitions/{league_id}/main.png

    -- URLs de thumbnails
    urls JSONB DEFAULT '{}',
    -- Structure: { "64": "https://...", "128": "...", "256": "...", "512": "..." }

    -- Fallback
    fallback_url VARCHAR(500),          -- API-Football URL

    -- Status (simplified - no facing variants)
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- Values: 'pending', 'queued', 'processing', 'pending_resize', 'ready', 'error'

    -- Metadata
    batch_job_id UUID,
    ia_model VARCHAR(50),
    ia_prompt_version VARCHAR(20),
    ia_cost_usd DECIMAL(10,4),

    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Validation
    validation_errors JSONB,
    last_validation_at TIMESTAMP,

    -- Timestamps
    uploaded_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_competition_logos_status ON competition_logos(status)
    WHERE status NOT IN ('ready', 'error');

-- =============================================================================
-- Table: logo_batch_jobs
-- Purpose: Track batch generation jobs with pause/resume/cancel support
-- =============================================================================

CREATE TABLE IF NOT EXISTS logo_batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- =========================================================================
    -- Configuration
    -- =========================================================================
    ia_model VARCHAR(50) NOT NULL,
    generation_mode VARCHAR(20) NOT NULL DEFAULT 'full_3d',
    -- Values: 'full_3d', 'facing_only', 'front_only', 'manual'
    prompt_front TEXT,                   -- NULL if mode = facing_only or manual
    prompt_right TEXT,                   -- NULL if mode = front_only or manual
    prompt_left TEXT,                    -- NULL if mode = front_only or manual
    prompt_version VARCHAR(20) NOT NULL,

    -- =========================================================================
    -- Scope
    -- =========================================================================
    entity_type VARCHAR(20) NOT NULL DEFAULT 'league',
    -- Values: 'league', 'national_teams', 'competitions', 'custom'
    league_id INTEGER REFERENCES admin_leagues(league_id),
    total_teams INTEGER NOT NULL,
    team_ids INTEGER[],                  -- NULL = all with status 'pending'

    -- =========================================================================
    -- Status
    -- =========================================================================
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    -- Values: 'running', 'paused', 'completed', 'cancelled', 'error', 'pending_review'

    -- =========================================================================
    -- Progress
    -- =========================================================================
    processed_teams INTEGER DEFAULT 0,
    processed_images INTEGER DEFAULT 0,
    failed_teams INTEGER DEFAULT 0,

    -- =========================================================================
    -- Cost
    -- =========================================================================
    estimated_cost_usd DECIMAL(10,2),
    actual_cost_usd DECIMAL(10,2) DEFAULT 0,

    -- =========================================================================
    -- Approval (Liga-by-liga workflow)
    -- =========================================================================
    approval_status VARCHAR(20) DEFAULT 'pending_review',
    -- Values: 'pending_review', 'approved', 'partially_approved', 'rejected'
    approved_count INTEGER DEFAULT 0,
    rejected_count INTEGER DEFAULT 0,
    approved_by VARCHAR(100),
    approved_at TIMESTAMP,

    -- =========================================================================
    -- Re-run support
    -- =========================================================================
    parent_batch_id UUID REFERENCES logo_batch_jobs(id),
    is_rerun BOOLEAN DEFAULT FALSE,
    rerun_reason VARCHAR(100),           -- 'bad_quality', 'prompt_updated', 'partial_failures'

    -- =========================================================================
    -- Timestamps
    -- =========================================================================
    started_at TIMESTAMP DEFAULT NOW(),
    paused_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Metadata
    started_by VARCHAR(100),             -- User who started the job

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logo_batch_jobs_status ON logo_batch_jobs(status)
    WHERE status IN ('running', 'paused', 'pending_review');
CREATE INDEX IF NOT EXISTS idx_logo_batch_jobs_league ON logo_batch_jobs(league_id)
    WHERE league_id IS NOT NULL;

-- =============================================================================
-- Table: logo_prompt_templates
-- Purpose: Version and track prompts for A/B testing (Kimi consideration)
-- =============================================================================

CREATE TABLE IF NOT EXISTS logo_prompt_templates (
    id SERIAL PRIMARY KEY,
    version VARCHAR(10) NOT NULL,        -- 'v1', 'v2', etc.
    variant VARCHAR(20) NOT NULL,        -- 'front', 'right', 'left', 'main'
    prompt_template TEXT NOT NULL,
    ia_model VARCHAR(50),                -- NULL = all models
    is_active BOOLEAN DEFAULT FALSE,
    success_rate DECIMAL(5,2),           -- % historical success
    avg_quality_score DECIMAL(3,2),      -- Average manual rating
    usage_count INTEGER DEFAULT 0,       -- How many times used
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),

    UNIQUE(version, variant, ia_model)
);

CREATE INDEX IF NOT EXISTS idx_prompt_templates_active ON logo_prompt_templates(is_active, variant)
    WHERE is_active = TRUE;

-- =============================================================================
-- Initial Data: Prompt Templates v1
-- =============================================================================

INSERT INTO logo_prompt_templates (version, variant, prompt_template, is_active, notes) VALUES
('v1', 'front',
 'Transform this 2D football team shield into a photorealistic 3D metallic badge. Style: glossy chrome rim, brushed metal center, professional sports badge aesthetic. Lighting: frontal, even illumination, subtle reflections. Background: completely transparent (alpha channel). Preserve all original colors, symbols, and design elements exactly. Output: 1024x1024 PNG.',
 TRUE, 'Initial prompt for front 3D'),

('v1', 'right',
 'Transform this 2D football team shield into a photorealistic 3D metallic badge rotated 45 degrees to face RIGHT (as if looking at an opponent on the right). Style: glossy chrome rim, brushed metal center. Lighting: left-to-right directional lighting with shadows on the left side. Background: completely transparent. Preserve all original design elements. Output: 1024x1024 PNG.',
 TRUE, 'Initial prompt for HOME (facing right)'),

('v1', 'left',
 'Transform this 2D football team shield into a photorealistic 3D metallic badge rotated 45 degrees to face LEFT (as if looking at an opponent on the left). Style: glossy chrome rim, brushed metal center. Lighting: right-to-left directional lighting with shadows on the right side. Background: completely transparent. Preserve all original design elements. Output: 1024x1024 PNG.',
 TRUE, 'Initial prompt for AWAY (facing left)'),

('v1', 'main',
 'Transform this 2D football league/tournament logo into a photorealistic 3D badge. Style: glossy metallic finish, professional sports aesthetic. Lighting: frontal, even illumination. Background: completely transparent. Preserve all original design elements. Output: 1024x1024 PNG.',
 TRUE, 'Prompt for competition logos')
ON CONFLICT (version, variant, ia_model) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE team_logos IS 'Tracks 3D logo variants (front, facing_right, facing_left) for teams';
COMMENT ON TABLE competition_logos IS 'Tracks 3D logos for leagues/tournaments (main variant only)';
COMMENT ON TABLE logo_batch_jobs IS 'Tracks batch generation jobs with liga-by-liga approval workflow';
COMMENT ON TABLE logo_prompt_templates IS 'Versioned prompts for A/B testing and rollback support';
