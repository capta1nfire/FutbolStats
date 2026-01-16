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
    league_id=34,  # API-Football: "World Cup - Qualification South America"
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
    league_id=31,  # API-Football: "World Cup - Qualification CONCACAF"
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
    league_id=29,  # API-Football: "World Cup - Qualification Africa"
    name="WC Qualifiers - CAF",
    match_type="official",
    priority=Priority.HIGH,
)

WC_QUALIFIERS_OFC = Competition(
    league_id=33,  # API-Football: "World Cup - Qualification Oceania"
    name="WC Qualifiers - OFC",
    match_type="official",
    priority=Priority.HIGH,
)

WC_QUALIFIERS_INTERCONTINENTAL = Competition(
    league_id=37,  # API-Football: "World Cup - Qualification Intercontinental Play-offs"
    name="WC Qualifiers - Intercontinental",
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

# Club Competitions - Europe Top 5 Leagues
PREMIER_LEAGUE = Competition(
    league_id=39,
    name="Premier League",
    match_type="official",
    priority=Priority.HIGH,
)

LA_LIGA = Competition(
    league_id=140,
    name="La Liga",
    match_type="official",
    priority=Priority.HIGH,
)

SERIE_A = Competition(
    league_id=135,
    name="Serie A",
    match_type="official",
    priority=Priority.HIGH,
)

BUNDESLIGA = Competition(
    league_id=78,
    name="Bundesliga",
    match_type="official",
    priority=Priority.HIGH,
)

LIGUE_1 = Competition(
    league_id=61,
    name="Ligue 1",
    match_type="official",
    priority=Priority.HIGH,
)

# Europe - Secondary Leagues (ML volume)
EFL_CHAMPIONSHIP = Competition(
    league_id=40,
    name="EFL Championship",
    match_type="official",
    priority=Priority.MEDIUM,
)

EREDIVISIE = Competition(
    league_id=88,
    name="Eredivisie",
    match_type="official",
    priority=Priority.MEDIUM,
)

PRIMEIRA_LIGA = Competition(
    league_id=94,
    name="Primeira Liga",
    match_type="official",
    priority=Priority.MEDIUM,
)

# UEFA Club Competitions
CHAMPIONS_LEAGUE = Competition(
    league_id=2,
    name="UEFA Champions League",
    match_type="official",
    priority=Priority.HIGH,
)

EUROPA_LEAGUE = Competition(
    league_id=3,
    name="UEFA Europa League",
    match_type="official",
    priority=Priority.MEDIUM,
)

CONFERENCE_LEAGUE = Competition(
    league_id=848,
    name="UEFA Conference League",
    match_type="official",
    priority=Priority.MEDIUM,
    match_weight=0.9,
)

# England - Domestic Cup
FA_CUP = Competition(
    league_id=45,
    name="FA Cup",
    match_type="official",
    priority=Priority.MEDIUM,
    match_weight=0.85,
)

# LATAM - Pack1 (Domestic leagues)
BRAZIL_SERIE_A = Competition(
    league_id=71,
    name="Brazil Serie A",
    match_type="official",
    priority=Priority.MEDIUM,
)

MEXICO_LIGA_MX = Competition(
    league_id=262,
    name="Mexico Liga MX",
    match_type="official",
    priority=Priority.MEDIUM,
)

ARGENTINA_PRIMERA = Competition(
    league_id=128,
    name="Argentina Primera División",
    match_type="official",
    priority=Priority.MEDIUM,
)

# LATAM - Pack2 (Domestic leagues)
COLOMBIA_PRIMERA_A = Competition(
    league_id=239,
    name="Colombia Primera A",
    match_type="official",
    priority=Priority.MEDIUM,
)

ECUADOR_LIGA_PRO = Competition(
    league_id=242,
    name="Ecuador Liga Pro",
    match_type="official",
    priority=Priority.MEDIUM,
)

PARAGUAY_PRIMERA = Competition(
    league_id=250,  # Apertura - API-Football uses same ID for both tournaments
    name="Paraguay Primera División",
    match_type="official",
    priority=Priority.MEDIUM,
)

CHILE_PRIMERA = Competition(
    league_id=265,
    name="Chile Primera División",
    match_type="official",
    priority=Priority.MEDIUM,
)

URUGUAY_PRIMERA = Competition(
    league_id=268,  # Apertura - API-Football uses same ID for both tournaments
    name="Uruguay Primera División",
    match_type="official",
    priority=Priority.MEDIUM,
)

PERU_PRIMERA = Competition(
    league_id=281,
    name="Perú Liga 1",
    match_type="official",
    priority=Priority.MEDIUM,
)

VENEZUELA_PRIMERA = Competition(
    league_id=299,
    name="Venezuela Primera División",
    match_type="official",
    priority=Priority.MEDIUM,
)

BOLIVIA_PRIMERA = Competition(
    league_id=344,
    name="Bolivia Primera División",
    match_type="official",
    priority=Priority.MEDIUM,
)

# CONMEBOL (Sudamérica)
CONMEBOL_LIBERTADORES = Competition(
    league_id=13,
    name="CONMEBOL Libertadores",
    match_type="official",
    priority=Priority.HIGH,
)

CONMEBOL_SUDAMERICANA = Competition(
    league_id=11,
    name="CONMEBOL Sudamericana",
    match_type="official",
    priority=Priority.MEDIUM,
)

# Spain - Domestic Cup
COPA_DEL_REY = Competition(
    league_id=143,
    name="Copa del Rey",
    match_type="official",
    priority=Priority.MEDIUM,
    match_weight=0.85,
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
        WC_QUALIFIERS_OFC,
        WC_QUALIFIERS_INTERCONTINENTAL,
        COPA_AMERICA,
        UEFA_EURO,
        UEFA_NATIONS_LEAGUE,
        CONCACAF_GOLD_CUP,
        AFRICA_CUP,
        AFC_ASIAN_CUP,
        INTERNATIONAL_FRIENDLIES,
        # Club Leagues - Top 5
        PREMIER_LEAGUE,
        LA_LIGA,
        SERIE_A,
        BUNDESLIGA,
        LIGUE_1,
        # Club Leagues - Secondary (ML volume)
        EFL_CHAMPIONSHIP,
        EREDIVISIE,
        PRIMEIRA_LIGA,
        # UEFA Club Competitions
        CHAMPIONS_LEAGUE,
        EUROPA_LEAGUE,
        CONFERENCE_LEAGUE,
        # LATAM / CONMEBOL
        BRAZIL_SERIE_A,
        MEXICO_LIGA_MX,
        ARGENTINA_PRIMERA,
        COLOMBIA_PRIMERA_A,
        ECUADOR_LIGA_PRO,
        PARAGUAY_PRIMERA,
        CHILE_PRIMERA,
        URUGUAY_PRIMERA,
        PERU_PRIMERA,
        VENEZUELA_PRIMERA,
        BOLIVIA_PRIMERA,
        CONMEBOL_LIBERTADORES,
        CONMEBOL_SUDAMERICANA,
        # Domestic Cups
        COPA_DEL_REY,
        FA_CUP,
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
