"""
Shared team name normalization for cross-provider matching.

Single source of truth — all matchers (Sofascore, Understat,
match_external_refs) MUST import from here.

Moved from sofascore_provider.py (P2 consolidation).
"""

import re
import unicodedata


_SAFE_ORG_TOKENS = [
    r"\bfc\b", r"\bcf\b", r"\bsc\b", r"\bafc\b", r"\bssc\b",
    r"\bac\b", r"\bas\b", r"\bcd\b", r"\bud\b", r"\brc\b",
    r"\bsv\b", r"\bvfb\b", r"\btsv\b", r"\bfk\b", r"\bsk\b",
    r"\bclub\b",
]


def normalize_team_name(name: str) -> str:
    """
    Normalize team name for fuzzy matching.

    Steps:
    1. Lowercase + trim
    2. Strip diacritics (NFKD)
    3. Replace punctuation/hyphens/slashes with space (not delete)
    4. Remove ONLY juridical/organizational tokens (NOT semantic ones)
    5. Collapse whitespace

    v2 changes vs original:
    - Does NOT strip 'real', 'united', 'city' (caused collisions:
      "Real Madrid" and "Real Sociedad" both became just the city name;
      "Manchester City" and "Manchester United" both became "manchester")
    - Punctuation replaced with space instead of deleted
      ("Bodo/Glimt" -> "bodo glimt" instead of "bodoglimt")

    Examples:
        "Manchester United FC" -> "manchester united"
        "Manchester City FC"   -> "manchester city"
        "Real Madrid"          -> "real madrid"
        "Real Sociedad"        -> "real sociedad"
        "Atletico Madrid"      -> "atletico madrid"
        "FC Barcelona"         -> "barcelona"
        "Bodo/Glimt"           -> "bodo glimt"
        "Paris Saint-Germain"  -> "paris saint germain"
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower().strip()

    # Remove accents/diacritics
    # Manual replacements for chars NFKD doesn't decompose (Nordic letters)
    name = name.replace("ø", "o").replace("æ", "ae").replace("ð", "d")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Replace punctuation/hyphens/slashes with space (preserve word boundaries)
    name = re.sub(r"[^\w\s]", " ", name)

    # Remove ONLY juridical/organizational tokens (safe to strip)
    # NOT semantic tokens like 'real', 'united', 'city' which distinguish teams
    for token in _SAFE_ORG_TOKENS:
        name = re.sub(token, "", name)

    # Collapse whitespace
    name = " ".join(name.split())

    return name
