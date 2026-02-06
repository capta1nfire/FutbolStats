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


# --- Phase 2: Zones/Badges ---

# Zone mappings: API-Football description keyword -> structured zone info
# ABE P0: No generic "play" keyword - use specific matches to avoid
# false positives with "display"/"player"/etc.
_ZONE_MAPPINGS = [
    ("champions league", {"type": "promotion", "tournament": "Champions League", "style": "blue"}),
    ("europa league", {"type": "promotion", "tournament": "Europa League", "style": "orange"}),
    ("conference league", {"type": "promotion", "tournament": "Conference League", "style": "green"}),
    ("libertadores", {"type": "promotion", "tournament": "Copa Libertadores", "style": "blue"}),
    ("sudamericana", {"type": "promotion", "tournament": "Copa Sudamericana", "style": "orange"}),
    ("championship round", {"type": "playoff", "description": "Championship Round", "style": "cyan"}),
    ("qualifying round", {"type": "playoff", "description": "Qualifying Round", "style": "gray"}),
    ("group matches", {"type": "playoff", "style": "cyan"}),
    # ABE P0: Specific playoff patterns only - no bare "play"
    ("playoff", {"type": "playoff", "style": "cyan"}),
    ("play-off", {"type": "playoff", "style": "cyan"}),
    ("play offs", {"type": "playoff", "style": "cyan"}),
    ("relegation", {"type": "relegation", "style": "red"}),
    ("descenso", {"type": "relegation", "style": "red"}),
]


def parse_api_zone_description(description: str) -> Optional[dict]:
    """
    Parse API-Football zone description to structured zone format.

    Maps known keywords to {type, tournament?, description?, style}.
    Falls back to generic "other" zone for unrecognized descriptions.

    ABE P0: Uses specific keyword matches (not bare "play") to avoid
    false positives with words like "display"/"player".
    """
    if not description:
        return None

    desc_lower = description.lower()

    for keyword, zone in _ZONE_MAPPINGS:
        if keyword in desc_lower:
            return dict(zone)  # Return copy to avoid mutating template

    return {"type": "other", "description": description, "style": "gray"}


def apply_zones(standings: list[dict], zones_config: dict) -> list[dict]:
    """
    Apply zone information to standings entries.

    ABE P0: Early return without adding 'zone' key when zones_config
    is empty or enabled=false (backwards compatible).

    Priority for zone assignment:
    1. Manual overrides from zones_config.overrides (by position range)
    2. API-Football description field (parsed via parse_api_zone_description)
    3. None if neither applies

    ABE P1: Tolerates missing 'position' with fallback to 'rank'.
    Leaves zone=None without breaking if position can't be determined.

    Args:
        standings: List of standings entries (mutated in-place)
        zones_config: rules_json.zones config dict

    Returns:
        Same list with 'zone' field added to each entry
    """
    # ABE P0: Early return - don't add zone key at all
    if not zones_config or not zones_config.get("enabled", False):
        return standings

    overrides = zones_config.get("overrides", {})

    for entry in standings:
        # ABE P1: Fallback to rank if position missing
        # ABE nit: Cast to int defensively in case source sends string
        raw_pos = entry.get("position") or entry.get("rank")
        try:
            pos = int(raw_pos) if raw_pos is not None else None
        except (ValueError, TypeError):
            pos = None

        zone = None

        # 1. Check manual overrides by position range
        if pos is not None:
            for range_str, zone_config in overrides.items():
                try:
                    if "-" in str(range_str):
                        start, end = map(int, str(range_str).split("-"))
                        if start <= pos <= end:
                            zone = dict(zone_config)  # Copy to avoid mutation
                            break
                    elif int(range_str) == pos:
                        zone = dict(zone_config)
                        break
                except (ValueError, TypeError):
                    continue

        # 2. Fallback to API-Football description
        if zone is None and entry.get("description"):
            zone = parse_api_zone_description(entry["description"])

        entry["zone"] = zone

    return standings
