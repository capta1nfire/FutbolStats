"""
Standings View Selection Module.

Canonical functions for selecting which standings group to display.
Implements heuristic-based selection with manual override capability.

ABE P0: DB-first approach - all groups stored in DB, filtering at endpoint level.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Keywords that indicate playoff/auxiliary groups (blacklist)
# ABE P1: Includes "promedios", "reclasificacion" for LATAM leagues
PLAYOFF_KEYWORDS = [
    "playoff", "play-off", "final", "semifinal", "quarter",
    "championship round", "relegation round", "qualifying round",
    "cuadrangular", "octavos", "liguilla", "knockout",
    "promotion playoff", "relegation playoff",
    "promedios", "reclasificacion",
]


class StandingsGroupNotFound(Exception):
    """Raised when requested group doesn't exist in standings."""

    def __init__(self, requested: str, available: list[str]):
        self.requested = requested
        self.available = available
        super().__init__(f"Group '{requested}' not found. Available: {available}")


@dataclass
class StandingsViewResult:
    """Result of selecting a standings view."""

    standings: list[dict]
    selected_group: str
    selection_reason: str  # "query_param", "config_override", "heuristic_*"
    available_groups: list[str]
    tie_warning: Optional[list[str]]


def group_standings_by_name(standings: list[dict]) -> dict[str, list[dict]]:
    """
    Group standings entries by their group name.

    Args:
        standings: Raw standings array from league_standings.standings

    Returns:
        Dict mapping group_name -> list of entries
    """
    groups: dict[str, list[dict]] = {}
    for entry in standings:
        # ABE P0: Handle explicit None (not just missing key)
        group_name = entry.get("group") or "Unknown"
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(entry)
    return groups


def select_standings_view(
    standings: list[dict],
    rules_json: dict,
    requested_group: Optional[str] = None,
) -> StandingsViewResult:
    """
    CANONICAL function for selecting standings view.

    This is the single source of truth for standings group selection.
    ABE P0: All selection logic goes through this function.

    Args:
        standings: Complete array from league_standings.standings
        rules_json: Config from admin_leagues.rules_json
        requested_group: Query param ?group= (optional)

    Returns:
        StandingsViewResult with filtered data and metadata

    Raises:
        StandingsGroupNotFound: If requested_group doesn't exist
    """
    # Handle empty standings
    if not standings:
        return StandingsViewResult(
            standings=[],
            selected_group="",
            selection_reason="empty",
            available_groups=[],
            tie_warning=None,
        )

    # 1. Group entries by name
    groups = group_standings_by_name(standings)
    available_groups = list(groups.keys())

    # 2. If query param provided, validate and use
    if requested_group:
        if requested_group not in groups:
            raise StandingsGroupNotFound(
                requested=requested_group,
                available=available_groups,
            )
        return StandingsViewResult(
            standings=groups[requested_group],
            selected_group=requested_group,
            selection_reason="query_param",
            available_groups=available_groups,
            tie_warning=None,
        )

    # 3. Apply heuristic
    selected_group, reason = select_default_standings_group(
        groups=groups,
        rules_json=rules_json,
    )

    # 4. Detect TIE for warning
    tie_warning = detect_standings_tie(groups)

    if tie_warning:
        logger.warning(
            f"[STANDINGS] TIE detected: {tie_warning}. "
            f"Selected '{selected_group}' via {reason}. "
            "Consider setting rules_json.standings.default_group"
        )

    logger.info(
        f"[STANDINGS] Selected group '{selected_group}' via {reason}. "
        f"Available: {available_groups}"
    )

    return StandingsViewResult(
        standings=groups[selected_group],
        selected_group=selected_group,
        selection_reason=reason,
        available_groups=available_groups,
        tie_warning=tie_warning,
    )


def select_default_standings_group(
    groups: dict[str, list[dict]],
    rules_json: dict,
) -> tuple[str, str]:
    """
    Select the default group to display in standings.

    Args:
        groups: Dict {group_name: [entries]}
        rules_json: Config from admin_leagues

    Returns:
        Tuple (selected_group_name, selection_reason)

    Priorities:
    1. Override manual en rules_json.standings.default_group
    2. Whitelist de patrones válidos (si configurada)
    3. Preferir grupo con team_count == config.team_count (hint)
    4. Preferir "Overall" si existe
    5. MAX(team_count) excluyendo grupos con keywords de playoffs
    """
    if not groups:
        return ("", "empty")

    standings_config = rules_json.get("standings", {})

    # 1. Override manual
    default_group = standings_config.get("default_group")
    if default_group and default_group in groups:
        return (default_group, "config_override")

    # 2. Whitelist de patrones válidos (Kimi adjustment)
    valid_patterns = standings_config.get("valid_group_patterns")
    if valid_patterns:
        for name in groups:
            if any(pattern.lower() in name.lower() for pattern in valid_patterns):
                return (name, "heuristic_whitelist")

    # Build candidates list (exclude playoff/auxiliary groups)
    def is_playoff_group(name: str) -> bool:
        name_lower = name.lower()
        return any(kw in name_lower for kw in PLAYOFF_KEYWORDS)

    candidates = {
        name: entries
        for name, entries in groups.items()
        if not is_playoff_group(name)
    }

    # Fallback: if all groups are playoffs, use all
    if not candidates:
        candidates = groups

    # 3. Prefer group with team_count == config hint (ABE P0)
    expected_team_count = standings_config.get("team_count")
    if expected_team_count:
        for name, entries in candidates.items():
            if len(entries) == expected_team_count:
                return (name, "heuristic_team_count_match")

    # 4. Prefer "Overall" if exists
    for name in candidates:
        if "overall" in name.lower():
            return (name, "heuristic_overall")

    # 5. MAX team count
    max_group = max(candidates.items(), key=lambda x: len(x[1]))
    return (max_group[0], "heuristic_max_teams")


def detect_standings_tie(groups: dict[str, list[dict]]) -> Optional[list[str]]:
    """
    Detect if multiple groups have the same (MAX) team count.

    Args:
        groups: Dict {group_name: [entries]} (already grouped)

    Returns:
        List of group names in TIE, or None if no tie.

    Used for: Log warning + require manual config.
    """
    if not groups:
        return None

    counts = [(name, len(entries)) for name, entries in groups.items()]
    max_count = max(c[1] for c in counts)

    tied = [name for name, count in counts if count == max_count]

    if len(tied) > 1:
        return tied
    return None
