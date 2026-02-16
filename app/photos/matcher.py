"""Identity Matching for Player Photos.

Scores how likely a candidate photo matches the intended player.
Uses proportional scoring: only penalizes when signals are available
but don't match (fix #5 — never penalize missing data).

Scoring (out of available points):
- Name fuzzy match: +40 (exact) / +30 (>=0.85) / +0 (miss)
- Jersey number match: +25
- Position match: +15
- Team context match: +20
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from app.photos.config import get_photos_settings

logger = logging.getLogger(__name__)
photos_settings = get_photos_settings()

# Suffixes to strip for name normalization
NAME_SUFFIXES = re.compile(r"\b(jr\.?|sr\.?|iii|ii|iv|v)\b", re.IGNORECASE)


@dataclass
class CandidateSignals:
    """Signals extracted from a photo candidate."""

    name: Optional[str] = None
    jersey_number: Optional[int] = None
    position: Optional[str] = None
    team_name: Optional[str] = None
    team_external_id: Optional[int] = None


@dataclass
class PlayerDB:
    """Player record from database."""

    external_id: int
    name: str
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    jersey_number: Optional[int] = None
    position: Optional[str] = None
    team_external_id: Optional[int] = None
    team_name: Optional[str] = None


@dataclass
class IdentityResult:
    """Result of identity matching."""

    score: int  # 0-100 normalized
    raw_points: int
    max_possible: int
    signals_used: int
    min_signals_met: bool
    details: dict  # per-signal breakdown

    @property
    def passes_threshold(self) -> bool:
        return self.min_signals_met and self.score >= photos_settings.PHOTOS_IDENTITY_THRESHOLD


def normalize_name(name: str) -> str:
    """Normalize player name for fuzzy comparison.

    - NFKD unicode normalization (remove accents)
    - Lowercase
    - Strip suffixes (Jr., III, etc.)
    - Collapse whitespace

    Args:
        name: Raw player name

    Returns:
        Normalized name string
    """
    # NFKD normalize (remove accents)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    name = name.lower().strip()
    # Strip suffixes
    name = NAME_SUFFIXES.sub("", name).strip()
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def fuzzy_name_score(candidate_name: str, db_name: str, db_firstname: Optional[str] = None, db_lastname: Optional[str] = None) -> float:
    """Compute fuzzy name similarity.

    Tries multiple combinations:
    1. Full name vs full name
    2. Candidate vs firstname+lastname
    3. Candidate vs lastname only (common in LATAM)
    4. Initial+lastname pattern (API-Football "J. Ramírez" vs club "Juan Carlos Ramírez")

    Returns best score (0.0-1.0).
    """
    c = normalize_name(candidate_name)

    scores = []

    # Full name match
    d = normalize_name(db_name)
    scores.append(SequenceMatcher(None, c, d).ratio())

    # First+last combination
    if db_firstname and db_lastname:
        fl = normalize_name(f"{db_firstname} {db_lastname}")
        scores.append(SequenceMatcher(None, c, fl).ratio())

    # Lastname only (common for LATAM: "Falcao", "James")
    if db_lastname:
        ln = normalize_name(db_lastname)
        scores.append(SequenceMatcher(None, c, ln).ratio())

    # Initial+lastname pattern: "J. Ramírez" vs "Juan Carlos Ramírez Mejía"
    # If DB lastname is contained in candidate AND first initials match → high score
    if db_lastname and db_firstname:
        ln = normalize_name(db_lastname)
        fn_initial = normalize_name(db_firstname)[:1]
        c_tokens = c.split()
        if c_tokens and fn_initial and c_tokens[0][:1] == fn_initial:
            # First initial matches, check if any candidate token matches lastname
            for token in c_tokens:
                token_sim = SequenceMatcher(None, token, ln).ratio()
                if token_sim >= 0.85:
                    scores.append(0.95)  # Strong match: initial + lastname
                    break

    return max(scores) if scores else 0.0


def score_identity(candidate: CandidateSignals, player: PlayerDB) -> IdentityResult:
    """Score identity match between candidate photo and DB player.

    Scoring is proportional to available signals:
    - Only signals that are present in BOTH candidate and DB are scored
    - Missing signals are excluded from max_possible (fix #5)
    - Requires min_signals_required = 2

    Args:
        candidate: Signals from photo candidate
        player: Player record from database

    Returns:
        IdentityResult with normalized score 0-100
    """
    raw_points = 0
    max_possible = 0
    signals_used = 0
    details = {}

    # Signal 1: Name fuzzy match (+40 max)
    if candidate.name and player.name:
        max_possible += 40
        signals_used += 1
        sim = fuzzy_name_score(candidate.name, player.name, player.firstname, player.lastname)
        if sim >= 0.95:
            raw_points += 40
            details["name"] = {"score": 40, "similarity": sim, "match": "exact"}
        elif sim >= 0.80:
            raw_points += 30
            details["name"] = {"score": 30, "similarity": sim, "match": "close"}
        elif sim >= 0.65:
            raw_points += 15
            details["name"] = {"score": 15, "similarity": sim, "match": "partial"}
        else:
            details["name"] = {"score": 0, "similarity": sim, "match": "miss"}

    # Signal 2: Jersey number match (+25)
    if candidate.jersey_number is not None and player.jersey_number is not None:
        max_possible += 25
        signals_used += 1
        if candidate.jersey_number == player.jersey_number:
            raw_points += 25
            details["jersey"] = {"score": 25, "match": True}
        else:
            details["jersey"] = {"score": 0, "match": False}

    # Signal 3: Position match (+15)
    if candidate.position and player.position:
        max_possible += 15
        signals_used += 1
        # Normalize positions (Goalkeeper/Defender/Midfielder/Attacker)
        if candidate.position.lower()[:3] == player.position.lower()[:3]:
            raw_points += 15
            details["position"] = {"score": 15, "match": True}
        else:
            details["position"] = {"score": 0, "match": False}

    # Signal 4: Team context match (+20)
    if candidate.team_external_id is not None and player.team_external_id is not None:
        max_possible += 20
        signals_used += 1
        if candidate.team_external_id == player.team_external_id:
            raw_points += 20
            details["team"] = {"score": 20, "match": True}
        else:
            details["team"] = {"score": 0, "match": False}

    # Normalize to 0-100
    score = int((raw_points / max_possible) * 100) if max_possible > 0 else 0
    min_signals_met = signals_used >= 2

    return IdentityResult(
        score=score,
        raw_points=raw_points,
        max_possible=max_possible,
        signals_used=signals_used,
        min_signals_met=min_signals_met,
        details=details,
    )
