-- Migration: Add team_id index to match_player_stats
-- Required by compute_team_vorp_priors() for VORP roster query (ABE P0-2)
-- Idempotent: IF NOT EXISTS
CREATE INDEX IF NOT EXISTS idx_mps_team_id
    ON match_player_stats (team_id);
