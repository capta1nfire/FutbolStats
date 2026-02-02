-- Persist match round / matchday from API-Football.
-- Source: fixture.league.round (e.g., "Regular Season - 21", "Quarter-finals")

ALTER TABLE matches
ADD COLUMN IF NOT EXISTS round VARCHAR(80);

COMMENT ON COLUMN matches.round IS 'API-Football fixture.league.round (e.g., Regular Season - 21)';

