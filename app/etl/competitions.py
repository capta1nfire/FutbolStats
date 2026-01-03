"""Competition configurations and IDs for API-Football."""

from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Competition priority levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Competition:
    """Competition configuration."""

    league_id: int
    name: str
    match_type: str  # "official" or "friendly"
    priority: Priority
    match_weight: float = 1.0

    def __post_init__(self):
        if self.match_type == "friendly":
            self.match_weight = 0.6


# Priority: HIGH (Mandatory)
WORLD_CUP = Competition(
    league_id=1,
    name="FIFA World Cup",
    match_type="official",
    priority=Priority.HIGH,
)

WC_QUALIFIERS_CONMEBOL = Competition(
    league_id=28,
    name="WC Qualifiers - CONMEBOL",
    match_type="official",
    priority=Priority.HIGH,
)

WC_QUALIFIERS_UEFA = Competition(
    league_id=32,
    name="WC Qualifiers - UEFA",
    match_type="official",
    priority=Priority.HIGH,
)

WC_QUALIFIERS_CONCACAF = Competition(
    league_id=29,
    name="WC Qualifiers - CONCACAF",
    match_type="official",
    priority=Priority.HIGH,
)

WC_QUALIFIERS_AFC = Competition(
    league_id=30,
    name="WC Qualifiers - AFC",
    match_type="official",
    priority=Priority.HIGH,
)

WC_QUALIFIERS_CAF = Competition(
    league_id=31,
    name="WC Qualifiers - CAF",
    match_type="official",
    priority=Priority.HIGH,
)

# Priority: MEDIUM
COPA_AMERICA = Competition(
    league_id=9,
    name="Copa América",
    match_type="official",
    priority=Priority.MEDIUM,
)

UEFA_EURO = Competition(
    league_id=4,
    name="UEFA Euro",
    match_type="official",
    priority=Priority.MEDIUM,
)

UEFA_NATIONS_LEAGUE = Competition(
    league_id=5,
    name="UEFA Nations League",
    match_type="official",
    priority=Priority.MEDIUM,
)

CONCACAF_GOLD_CUP = Competition(
    league_id=22,
    name="CONCACAF Gold Cup",
    match_type="official",
    priority=Priority.MEDIUM,
)

AFRICA_CUP = Competition(
    league_id=6,
    name="Africa Cup of Nations",
    match_type="official",
    priority=Priority.MEDIUM,
)

AFC_ASIAN_CUP = Competition(
    league_id=7,
    name="AFC Asian Cup",
    match_type="official",
    priority=Priority.MEDIUM,
)

# Priority: LOW
INTERNATIONAL_FRIENDLIES = Competition(
    league_id=10,
    name="International Friendlies",
    match_type="friendly",
    priority=Priority.LOW,
)

# All competitions dictionary
COMPETITIONS: dict[int, Competition] = {
    comp.league_id: comp
    for comp in [
        WORLD_CUP,
        WC_QUALIFIERS_CONMEBOL,
        WC_QUALIFIERS_UEFA,
        WC_QUALIFIERS_CONCACAF,
        WC_QUALIFIERS_AFC,
        WC_QUALIFIERS_CAF,
        COPA_AMERICA,
        UEFA_EURO,
        UEFA_NATIONS_LEAGUE,
        CONCACAF_GOLD_CUP,
        AFRICA_CUP,
        AFC_ASIAN_CUP,
        INTERNATIONAL_FRIENDLIES,
    ]
}

# Grouped by priority
HIGH_PRIORITY_LEAGUES = [
    comp.league_id for comp in COMPETITIONS.values() if comp.priority == Priority.HIGH
]

MEDIUM_PRIORITY_LEAGUES = [
    comp.league_id for comp in COMPETITIONS.values() if comp.priority == Priority.MEDIUM
]

LOW_PRIORITY_LEAGUES = [
    comp.league_id for comp in COMPETITIONS.values() if comp.priority == Priority.LOW
]

ALL_LEAGUE_IDS = list(COMPETITIONS.keys())
