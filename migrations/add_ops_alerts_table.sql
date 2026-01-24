-- Migration: Create ops_alerts table for Grafana webhook notifications
-- Date: 2026-01-24
-- Purpose: Store alerts from Grafana Alerting for bell/toast UI in /dashboard/ops

-- Create table if not exists
CREATE TABLE IF NOT EXISTS ops_alerts (
    id SERIAL PRIMARY KEY,

    -- Deduplication key (fingerprint from Grafana or computed)
    dedupe_key VARCHAR(255) NOT NULL UNIQUE,

    -- Alert status and severity
    status VARCHAR(20) NOT NULL DEFAULT 'firing',  -- firing, resolved
    severity VARCHAR(20) NOT NULL DEFAULT 'warning',  -- critical, warning, info

    -- Content
    title VARCHAR(500) NOT NULL,
    message TEXT,  -- truncated to 1000 chars on insert

    -- Grafana metadata (JSONB for flexibility)
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}',

    -- Timestamps from Grafana
    starts_at TIMESTAMP,
    ends_at TIMESTAMP,

    -- Source info
    source VARCHAR(50) NOT NULL DEFAULT 'grafana',
    source_url TEXT,  -- link to Grafana alert/silence

    -- Tracking
    first_seen_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- User interaction
    is_read BOOLEAN NOT NULL DEFAULT false,
    is_ack BOOLEAN NOT NULL DEFAULT false,

    -- Standard timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_ops_alerts_created_at ON ops_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ops_alerts_status ON ops_alerts(status);
CREATE INDEX IF NOT EXISTS idx_ops_alerts_is_read ON ops_alerts(is_read) WHERE is_read = false;
CREATE INDEX IF NOT EXISTS idx_ops_alerts_severity ON ops_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_ops_alerts_status_severity ON ops_alerts(status, severity);

-- Comments for documentation
COMMENT ON TABLE ops_alerts IS 'Alerts from Grafana Alerting webhook for ops dashboard bell/toast notifications';
COMMENT ON COLUMN ops_alerts.dedupe_key IS 'Unique key for idempotent upserts (Grafana fingerprint or computed)';
COMMENT ON COLUMN ops_alerts.status IS 'firing = active alert, resolved = alert cleared';
COMMENT ON COLUMN ops_alerts.severity IS 'critical (toast), warning (badge only), info (badge only)';
COMMENT ON COLUMN ops_alerts.source_url IS 'Link to Grafana alert detail or silence URL';
COMMENT ON COLUMN ops_alerts.is_read IS 'User has seen the alert in the bell dropdown';
COMMENT ON COLUMN ops_alerts.is_ack IS 'User has acknowledged the alert';
