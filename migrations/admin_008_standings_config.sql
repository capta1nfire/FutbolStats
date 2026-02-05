-- Migration: admin_008_standings_config.sql
-- Date: 2026-02-05
-- Description: Configure rules_json.standings for leagues with multiple groups
--
-- ABE P0: Use COALESCE || for partial merge (preserves existing fields)
-- DO NOT use full replacement which would lose existing rules_json fields
--
-- Ligas configuradas:
--   - Ecuador (242): Heurística con valid_group_patterns ["Serie A"]
--   - Colombia (239): Apertura/Clausura con cuadrangulares
--   - MLS (253): default_group "Overall" para TIE case
--   - Premier League (39): Zonas de API

-- ============================================================================
-- Ecuador (242) - Split format with multiple phases
-- ============================================================================
-- Groups: "Serie A 2025" (16), "Championship Round" (6), etc.
-- Heurística: valid_group_patterns selecciona "Serie A" automáticamente
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "format": "split",
    "team_count": 16,
    "valid_group_patterns": ["Serie A"]
  },
  "zones": {
    "enabled": true,
    "source": "manual",
    "overrides": {
      "1-6": {"type": "playoff", "description": "Championship Round", "style": "cyan"},
      "7-12": {"type": "playoff", "description": "Qualifying Round", "style": "gray"},
      "13-16": {"type": "relegation", "description": "Relegation Round", "style": "red"}
    }
  },
  "relegation": {
    "enabled": true,
    "count": 2,
    "method": "direct"
  }
}'::jsonb
WHERE league_id = 242;

-- ============================================================================
-- Colombia (239) - Apertura/Clausura with cuadrangulares
-- ============================================================================
-- Groups: "Primera Division: Apertura" (20)
-- Top 8 classify to cuadrangulares (playoffs)
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "format": "apertura_clausura",
    "team_count": 20,
    "valid_group_patterns": ["Primera Division"]
  },
  "zones": {
    "enabled": true,
    "source": "hybrid",
    "overrides": {
      "1-8": {"type": "playoff", "description": "Clasifica a cuadrangulares", "style": "cyan"}
    }
  },
  "relegation": {
    "enabled": true,
    "count": 2,
    "method": "average_3y"
  },
  "reclasificacion": {
    "enabled": true,
    "source": "calculated"
  }
}'::jsonb
WHERE league_id = 239;

-- ============================================================================
-- MLS (253) - Conference split with TIE case
-- ============================================================================
-- Groups: "Eastern Conference" (15), "Western Conference" (15) - TIE
-- Requires manual default_group since heurística can't resolve TIE
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "default_group": "Overall",
    "format": "single",
    "team_count": 30
  },
  "zones": {
    "enabled": true,
    "source": "api"
  },
  "relegation": {
    "enabled": false
  }
}'::jsonb
WHERE league_id = 253;

-- ============================================================================
-- Premier League (39) - Single group with API zones
-- ============================================================================
-- Simple league with zones from API-Football
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "format": "single",
    "team_count": 20
  },
  "zones": {
    "enabled": true,
    "source": "api"
  },
  "relegation": {
    "enabled": true,
    "count": 3,
    "method": "direct"
  }
}'::jsonb
WHERE league_id = 39;

-- ============================================================================
-- Argentina (128) - Multiple groups with Group A/B structure
-- ============================================================================
-- Groups: "Group A" (30), "Group B" (30), "Promedios 2026" (30)
-- Heurística: team_count 30 + blacklist "promedios" selecciona Group A/B
-- TODO: May need default_group if TIE between Group A and B
UPDATE admin_leagues
SET rules_json = COALESCE(rules_json, '{}'::jsonb) || '{
  "version": 1,
  "standings": {
    "format": "split",
    "team_count": 30
  },
  "zones": {
    "enabled": true,
    "source": "api"
  },
  "relegation": {
    "enabled": true,
    "count": 2,
    "method": "average_3y"
  }
}'::jsonb
WHERE league_id = 128;

-- Verification queries (run manually after migration)
-- SELECT league_id, name, rules_json->'standings' as standings_config
-- FROM admin_leagues
-- WHERE league_id IN (242, 239, 253, 39, 128);
