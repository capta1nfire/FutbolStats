"""
Match External Refs utilities for multi-source linking.

Provides functions to:
1. Normalize team names for matching
2. Compute match similarity scores
3. Upsert external refs to DB

Reference: docs/ARCHITECTURE_SOTA.md section 1.2 (match_external_refs)
"""

import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def normalize_team_name(name: str) -> str:
    """
    Normalize team name for matching.

    Transformations:
    - Lowercase
    - Strip whitespace
    - Remove diacritics (á → a, ñ → n, etc.)
    - Remove punctuation
    - Collapse multiple spaces to single space
    - Common abbreviation handling (FC, CF, etc.)

    Args:
        name: Raw team name.

    Returns:
        Normalized team name string.
    """
    if not name:
        return ""

    # Lowercase and strip
    result = name.lower().strip()

    # Remove diacritics (NFD decomposition + filter combining marks)
    result = unicodedata.normalize("NFD", result)
    result = "".join(c for c in result if unicodedata.category(c) != "Mn")

    # Remove punctuation (keep alphanumeric and spaces)
    result = re.sub(r"[^\w\s]", " ", result)

    # Collapse multiple spaces
    result = re.sub(r"\s+", " ", result).strip()

    # Common suffix/prefix normalization (optional, helps matching)
    # Remove trailing FC, CF, SC, etc.
    result = re.sub(r"\b(fc|cf|sc|ac|afc|ssc|as|rc|cd|ud|sd|ca|rcd|real)\b", "", result)
    result = re.sub(r"\s+", " ", result).strip()

    return result


def compute_match_score(
    api_match: dict,
    ext_match: dict,
) -> float:
    """
    Compute similarity score between an API-Football match and an external source match.

    Score S in [0,1] based on (per ARCHITECTURE_SOTA.md):
    - kickoff UTC (tolerance ±2h): high weight (0.4)
    - normalized team names (home/away): high weight (0.4)
    - league/season if available: medium weight (0.15)
    - venue/city if exists: low weight (0.05)

    Args:
        api_match: Dict with keys:
            - kickoff_utc: datetime
            - home_team: str
            - away_team: str
            - league_id: Optional[int]
            - season: Optional[int]
            - venue_city: Optional[str]
        ext_match: Dict with same keys from external source.

    Returns:
        Score S in [0,1].
    """
    score = 0.0

    # 1) Kickoff time match (weight: 0.4)
    # Full points if within 2 hours, partial if within 6 hours
    api_kickoff = api_match.get("kickoff_utc")
    ext_kickoff = ext_match.get("kickoff_utc")

    if api_kickoff and ext_kickoff:
        # Ensure both are datetime objects
        if isinstance(api_kickoff, datetime) and isinstance(ext_kickoff, datetime):
            time_diff_hours = abs((api_kickoff - ext_kickoff).total_seconds()) / 3600
            if time_diff_hours <= 2:
                score += 0.4
            elif time_diff_hours <= 6:
                score += 0.2
            elif time_diff_hours <= 24:
                score += 0.1
            # Beyond 24h: 0 points

    # 2) Team names match (weight: 0.4, split 0.2 each)
    api_home = normalize_team_name(api_match.get("home_team", ""))
    api_away = normalize_team_name(api_match.get("away_team", ""))
    ext_home = normalize_team_name(ext_match.get("home_team", ""))
    ext_away = normalize_team_name(ext_match.get("away_team", ""))

    # Home team
    if api_home and ext_home:
        if api_home == ext_home:
            score += 0.2
        elif api_home in ext_home or ext_home in api_home:
            score += 0.15
        elif _fuzzy_team_match(api_home, ext_home):
            score += 0.1

    # Away team
    if api_away and ext_away:
        if api_away == ext_away:
            score += 0.2
        elif api_away in ext_away or ext_away in api_away:
            score += 0.15
        elif _fuzzy_team_match(api_away, ext_away):
            score += 0.1

    # 3) League/season match (weight: 0.15)
    api_league = api_match.get("league_id")
    ext_league = ext_match.get("league_id")
    api_season = api_match.get("season")
    ext_season = ext_match.get("season")

    if api_league and ext_league:
        if api_league == ext_league:
            score += 0.1
    if api_season and ext_season:
        if api_season == ext_season:
            score += 0.05

    # 4) Venue/city match (weight: 0.05)
    api_venue = normalize_team_name(api_match.get("venue_city", "") or "")
    ext_venue = normalize_team_name(ext_match.get("venue_city", "") or "")

    if api_venue and ext_venue:
        if api_venue == ext_venue or api_venue in ext_venue or ext_venue in api_venue:
            score += 0.05

    return min(score, 1.0)


def _fuzzy_team_match(name1: str, name2: str) -> bool:
    """
    Simple fuzzy matching for team names.

    Returns True if names share significant common words.
    """
    if not name1 or not name2:
        return False

    words1 = set(name1.split())
    words2 = set(name2.split())

    # Remove very common words
    stopwords = {"de", "la", "el", "los", "las", "the", "of", "and", "y", "e"}
    words1 = words1 - stopwords
    words2 = words2 - stopwords

    if not words1 or not words2:
        return False

    # Check overlap
    common = words1 & words2
    # If at least half of the shorter name's words match
    min_words = min(len(words1), len(words2))
    return len(common) >= max(1, min_words // 2)


def get_match_decision(score: float) -> tuple[bool, bool]:
    """
    Decide whether to link based on score.

    Args:
        score: Similarity score S in [0,1].

    Returns:
        Tuple (should_link, needs_review):
        - S >= 0.90: (True, False) - auto-link
        - 0.75 <= S < 0.90: (True, True) - link but needs_review
        - S < 0.75: (False, False) - no link
    """
    if score >= 0.90:
        return True, False
    elif score >= 0.75:
        return True, True
    else:
        return False, False


async def upsert_match_external_ref(
    session: AsyncSession,
    match_id: int,
    source: str,
    source_match_id: str,
    confidence: float,
    matched_by: str,
    created_at: Optional[datetime] = None,
) -> str:
    """
    Upsert a match external reference.

    UPSERT by PK (match_id, source).

    Args:
        session: Async DB session.
        match_id: Internal match ID (FK to matches.id).
        source: Source identifier ('understat', 'sofascore', 'api_football').
        source_match_id: External match ID from source (as string).
        confidence: Match confidence score [0,1].
        matched_by: Heuristic description (e.g., 'kickoff+teams;needs_review').
        created_at: Timestamp (UTC). Defaults to now.

    Returns:
        'inserted' or 'updated'.
    """
    if created_at is None:
        # Use naive datetime (match_external_refs.created_at is timestamp without timezone)
        created_at = datetime.utcnow()

    # Check if exists
    check = await session.execute(
        text("""
            SELECT 1 FROM match_external_refs
            WHERE match_id = :match_id AND source = :source
        """),
        {"match_id": match_id, "source": source}
    )
    exists = check.scalar() is not None

    if exists:
        # Update
        await session.execute(
            text("""
                UPDATE match_external_refs SET
                    source_match_id = :source_match_id,
                    confidence = :confidence,
                    matched_by = :matched_by,
                    created_at = :created_at
                WHERE match_id = :match_id AND source = :source
            """),
            {
                "match_id": match_id,
                "source": source,
                "source_match_id": source_match_id,
                "confidence": confidence,
                "matched_by": matched_by,
                "created_at": created_at,
            }
        )
        return "updated"
    else:
        # Insert
        await session.execute(
            text("""
                INSERT INTO match_external_refs (
                    match_id, source, source_match_id, confidence, matched_by, created_at
                ) VALUES (
                    :match_id, :source, :source_match_id, :confidence, :matched_by, :created_at
                )
            """),
            {
                "match_id": match_id,
                "source": source,
                "source_match_id": source_match_id,
                "confidence": confidence,
                "matched_by": matched_by,
                "created_at": created_at,
            }
        )
        return "inserted"
