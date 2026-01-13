"""
Curated team aliases/nicknames for narrative enrichment.

IMPORTANT: Only aliases in this dictionary can be used by the LLM.
This prevents hallucination of fake/offensive nicknames.

Format: external_id (API-Football) -> TeamAliasPack
The LLM must ONLY use nicknames from nicknames_allowed.

Collisions allowed (lookup is team-specific):
- "La Roja": España, Chile
- "Los Diablos Rojos": Bélgica, Manchester United, América de Cali
- "Las Águilas": Club América, Lazio
- "Auriazules": Pumas UNAM, Tigres UANL
"""

import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamAliasPack:
    """Pack of allowed aliases for a team."""
    team_name: str
    nicknames: list[str]
    slogan: Optional[str] = None
    confidence: str = "High"  # High, Medium, Low


# Mapping: API-Football external_id -> TeamAliasPack
# Only non-pejorative, commonly accepted nicknames
TEAM_ALIASES: dict[int, TeamAliasPack] = {
    # ===================
    # LA LIGA (Spain)
    # ===================
    541: TeamAliasPack("Real Madrid", ["Los Merengues", "Los Blancos", "La Casa Blanca"]),
    530: TeamAliasPack("Atlético de Madrid", ["Los Colchoneros", "Los Rojiblancos", "Los Indios"]),
    529: TeamAliasPack("FC Barcelona", ["Culés", "Blaugranas", "Azulgranas"]),
    536: TeamAliasPack("Sevilla", ["Los Nervionenses", "Los Palanganas"]),
    532: TeamAliasPack("Valencia", ["Los Che"]),
    531: TeamAliasPack("Athletic Club", ["Los Leones"]),
    533: TeamAliasPack("Villarreal", ["El Submarino Amarillo"]),
    543: TeamAliasPack("Real Betis", ["Los Verdiblancos", "Los Béticos"]),
    548: TeamAliasPack("Real Sociedad", ["La Real"]),

    # ===================
    # SERIE A (Italy)
    # ===================
    505: TeamAliasPack("Inter Milan", ["Nerazzurri", "Il Biscione"]),
    489: TeamAliasPack("AC Milan", ["Rossoneri", "Il Diavolo"]),
    496: TeamAliasPack("Juventus", ["La Vecchia Signora", "Bianconeri"]),
    497: TeamAliasPack("AS Roma", ["Giallorossi", "La Loba"]),
    492: TeamAliasPack("Napoli", ["Partenopei", "Azzurri"]),
    487: TeamAliasPack("Lazio", ["Biancocelesti", "Las Águilas"]),

    # ===================
    # PREMIER LEAGUE (England)
    # ===================
    33: TeamAliasPack("Manchester United", ["The Red Devils", "Los Diablos Rojos"]),
    40: TeamAliasPack("Liverpool", ["The Reds"]),
    50: TeamAliasPack("Manchester City", ["The Citizens", "The Sky Blues"]),
    42: TeamAliasPack("Arsenal", ["The Gunners", "Los Artilleros"]),
    49: TeamAliasPack("Chelsea", ["The Blues"]),
    47: TeamAliasPack("Tottenham Hotspur", ["Spurs", "The Lilywhites"]),
    34: TeamAliasPack("Newcastle United", ["The Magpies", "Las Urracas"]),
    48: TeamAliasPack("West Ham United", ["The Hammers", "The Irons"]),

    # ===================
    # BUNDESLIGA (Germany)
    # ===================
    157: TeamAliasPack("Bayern Munich", ["Die Roten", "El Gigante de Baviera"]),
    165: TeamAliasPack("Borussia Dortmund", ["Die Schwarzgelben", "BVB"]),
    168: TeamAliasPack("Bayer Leverkusen", ["Die Werkself"], confidence="Medium"),

    # ===================
    # LIGUE 1 (France)
    # ===================
    85: TeamAliasPack("PSG", ["Les Parisiens", "Les Rouge et Bleu"]),
    81: TeamAliasPack("Marseille", ["Les Phocéens", "L'OM"]),
    80: TeamAliasPack("Lyon", ["Les Gones"]),

    # ===================
    # PORTUGUESE LIGA
    # ===================
    211: TeamAliasPack("Benfica", ["As Águias", "Encarnados"]),
    212: TeamAliasPack("Porto", ["Dragões"]),
    228: TeamAliasPack("Sporting CP", ["Leões", "Verde e Brancos"]),

    # ===================
    # ARGENTINA PRIMERA
    # ===================
    451: TeamAliasPack("Boca Juniors", ["Xeneize", "Azul y Oro", "La Mitad Más Uno"]),
    435: TeamAliasPack("River Plate", ["Los Millonarios", "La Banda"]),
    434: TeamAliasPack("Independiente", ["El Rojo", "Rey de Copas"]),
    436: TeamAliasPack("Racing Club", ["La Academia"]),
    437: TeamAliasPack("San Lorenzo", ["El Ciclón", "Los Cuervos"]),

    # ===================
    # BRAZIL SERIE A
    # ===================
    127: TeamAliasPack("Flamengo", ["Mengão", "Rubro-Negro"]),
    121: TeamAliasPack("Palmeiras", ["Verdão", "Porco"]),
    126: TeamAliasPack("São Paulo", ["Tricolor Paulista"]),
    128: TeamAliasPack("Santos", ["Peixe"]),
    131: TeamAliasPack("Corinthians", ["Timão"]),
    124: TeamAliasPack("Fluminense", ["Flu", "Tricolor Carioca"]),

    # ===================
    # LIGA MX (Mexico)
    # ===================
    2283: TeamAliasPack("Club América", ["Las Águilas", "Los Azulcremas"]),
    2282: TeamAliasPack("Guadalajara", ["Chivas", "El Rebaño Sagrado"]),
    2287: TeamAliasPack("Cruz Azul", ["La Máquina", "Cementeros"]),
    2302: TeamAliasPack("Pumas UNAM", ["Universitarios", "Auriazules"]),
    2286: TeamAliasPack("Tigres UANL", ["Felinos", "Auriazules"], confidence="Medium"),

    # ===================
    # COLOMBIA PRIMERA A
    # ===================
    1130: TeamAliasPack("Atlético Nacional", ["Los Verdolagas", "El Verde"]),
    1128: TeamAliasPack("Millonarios", ["Los Embajadores", "El Ballet Azul"]),
    1126: TeamAliasPack(
        "América de Cali",
        ["Los Diablos Rojos", "La Mecha", "La Mechita", "Los Escarlatas"],
        slogan="La Pasión de un Pueblo"
    ),

    # ===================
    # CHILE PRIMERA
    # ===================
    2323: TeamAliasPack("Colo-Colo", ["El Cacique", "El Popular", "Albos"]),
    2317: TeamAliasPack("Universidad de Chile", ["La U", "Los Azules"]),
    2316: TeamAliasPack("Universidad Católica", ["Los Cruzados"]),

    # ===================
    # URUGUAY PRIMERA
    # ===================
    2352: TeamAliasPack("Peñarol", ["Manyas", "Carboneros", "Aurinegros"]),
    2351: TeamAliasPack("Nacional", ["El Bolso", "Tricolores"]),

    # ===================
    # MLS (USA)
    # ===================
    1596: TeamAliasPack("LA Galaxy", ["The Gs"], confidence="Medium"),
    9568: TeamAliasPack("Inter Miami", ["Las Garzas"]),

    # ===================
    # NATIONAL TEAMS
    # ===================
    # South America
    26: TeamAliasPack("Argentina", ["La Albiceleste"]),
    6: TeamAliasPack("Brasil", ["La Canarinha", "La Verdeamarela", "Scratch du Oro"]),
    5: TeamAliasPack("Colombia", ["Los Cafeteros", "La Tricolor"]),
    7: TeamAliasPack("Chile", ["La Roja"]),
    28: TeamAliasPack("Uruguay", ["La Celeste", "Los Charrúas"]),
    24: TeamAliasPack("Perú", ["La Blanquirroja", "Los Incas"]),
    23: TeamAliasPack("Ecuador", ["La Tri"]),
    29: TeamAliasPack("Venezuela", ["La Vinotinto"]),
    30: TeamAliasPack("Paraguay", ["La Albirroja", "Los Guaraníes"]),

    # Europe
    9: TeamAliasPack("España", ["La Roja", "La Furia Roja"]),
    10: TeamAliasPack("Francia", ["Les Bleus"]),
    25: TeamAliasPack("Alemania", ["Die Mannschaft"]),
    768: TeamAliasPack("Italia", ["La Azzurra", "Gli Azzurri"]),
    1: TeamAliasPack("Inglaterra", ["The Three Lions", "Los Tres Leones"]),
    27: TeamAliasPack("Portugal", ["Os Navegadores", "Seleção das Quinas"], confidence="Medium"),
    15: TeamAliasPack("Países Bajos", ["La Naranja Mecánica", "Oranje"], confidence="Medium"),
    2: TeamAliasPack("Bélgica", ["Los Diablos Rojos"]),
    21: TeamAliasPack("Croacia", ["Vatreni", "El Equipo del Damero"]),

    # CONCACAF
    16: TeamAliasPack("México", ["El Tri"]),
}

# Default aliases when team not in dictionary
DEFAULT_HOME_ALIASES = ["los locales", "el equipo local"]
DEFAULT_AWAY_ALIASES = ["los visitantes", "el equipo visitante"]

# Reference style probabilities (deterministic by match_id)
# 70% team name, 20% generic reference, 10% nickname
STYLE_PROB_TEAM_NAME = 0.70
STYLE_PROB_GENERIC = 0.20
STYLE_PROB_NICKNAME = 0.10

# Probability thresholds for nickname usage (legacy, kept for compatibility)
NICKNAME_PROB_HIGH = 0.30
NICKNAME_PROB_MEDIUM = 0.15
NICKNAME_PROB_LOW = 0.05

# Generic team references (neutral, no color/adjective combinations)
GENERIC_REFERENCES = {
    "always": [
        "el equipo",
        "el conjunto",
        "la escuadra",
        "el once",
    ],
    "home_only": [
        "el local",
        "los locales",
    ],
    "away_only": [
        "el visitante",
        "la visita",
    ],
}

# Forbidden patterns - these should NEVER appear in narratives
FORBIDDEN_PATTERNS = [
    # Generic + color/adjective combinations
    "cuadro blanco", "cuadro rojo", "cuadro azul",
    "once blanco", "once rojo", "once azul",
    "onceno blanco", "onceno rojo", "onceno azul",
    "equipo blanco", "equipo rojo", "equipo merengue",
    "conjunto blanco", "conjunto merengue",
    # Fan chants not in whitelist
    "hala madrid", "visca barça", "forza juve",
    # Invented gentilics (unless in nicknames_allowed)
    "madridista", "barcelonista", "sevillista", "bético",
]


def _slugify(text: str) -> str:
    """Convert text to lowercase slug (for matching)."""
    import unicodedata
    # Normalize unicode and remove accents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # Lowercase and replace spaces/special chars with hyphen
    text = text.lower().strip()
    text = ''.join(c if c.isalnum() else '-' for c in text)
    # Remove consecutive hyphens
    while '--' in text:
        text = text.replace('--', '-')
    return text.strip('-')


def get_team_aliases(external_id: int, team_name: str, is_home: bool) -> list[str]:
    """
    Get safe aliases for a team (legacy function for backwards compatibility).

    Returns list starting with team name, then curated aliases or defaults.
    """
    aliases = [team_name]  # Always include actual name first

    if external_id in TEAM_ALIASES:
        pack = TEAM_ALIASES[external_id]
        aliases.extend(pack.nicknames)

    # Add positional aliases
    if is_home:
        aliases.extend(DEFAULT_HOME_ALIASES)
    else:
        aliases.extend(DEFAULT_AWAY_ALIASES)

    return aliases


def get_team_alias_pack(
    team_name: str,
    external_id: Optional[int] = None,
    match_id: Optional[int] = None,
    is_home: bool = True
) -> dict:
    """
    Get alias pack for LLM narrative generation.

    Args:
        team_name: Official team name
        external_id: API-Football team ID (preferred lookup)
        match_id: Match ID for deterministic style selection
        is_home: Whether this is the home team (for role-specific references)

    Returns:
        dict with:
        - team_name: str
        - nicknames_allowed: list[str]
        - selected_nickname: str | None (deterministic selection if match_id provided)
        - reference_style: str (team_name | generic | nickname)
        - generic_references: list[str] (allowed generic refs for this team's role)
        - slogan: str | None
        - confidence: str
    """
    pack = TEAM_ALIASES.get(external_id) if external_id else None

    if not pack:
        # Try fuzzy match by team name
        team_key = _slugify(team_name)
        for eid, p in TEAM_ALIASES.items():
            if _slugify(p.team_name) == team_key:
                pack = p
                break

    # Build generic references based on role
    role_refs = GENERIC_REFERENCES["always"].copy()
    if is_home:
        role_refs.extend(GENERIC_REFERENCES["home_only"])
    else:
        role_refs.extend(GENERIC_REFERENCES["away_only"])

    if not pack:
        return {
            "team_name": team_name,
            "nicknames_allowed": [],
            "selected_nickname": None,
            "reference_style": "team_name",
            "generic_references": role_refs,
            "slogan": None,
            "confidence": "None"
        }

    # Deterministic style and nickname selection
    selected_nickname = None
    reference_style = "team_name"

    if match_id is not None:
        # Create deterministic seed from match_id + team
        seed_str = f"{match_id}:{external_id or team_name}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

        # Determine reference style (70% team_name, 20% generic, 10% nickname)
        style_value = (seed % 100) / 100.0
        if style_value < STYLE_PROB_TEAM_NAME:
            reference_style = "team_name"
        elif style_value < STYLE_PROB_TEAM_NAME + STYLE_PROB_GENERIC:
            reference_style = "generic"
        else:
            # Only use nickname if available and confidence is sufficient
            if pack.nicknames and pack.confidence in ("High", "Medium"):
                reference_style = "nickname"
                nickname_idx = seed % len(pack.nicknames)
                selected_nickname = pack.nicknames[nickname_idx]
            else:
                reference_style = "team_name"

    return {
        "team_name": pack.team_name,
        "nicknames_allowed": pack.nicknames,
        "selected_nickname": selected_nickname,
        "reference_style": reference_style,
        "generic_references": role_refs,
        "slogan": pack.slogan,
        "confidence": pack.confidence
    }


def get_reference_rules_for_prompt() -> str:
    """
    Generate the reference_rules block for the LLM prompt.

    Returns:
        Formatted string with reference rules for the prompt.
    """
    return """
REGLAS DE REFERENCIAS A EQUIPOS:

A) Formas permitidas para referirte a cada equipo:
   1. NOMBRE OFICIAL: Usa el nombre del equipo directamente.
   2. APODO (si está en nicknames_allowed): SIEMPRE entre comillas. Ej: "Los Merengues"
   3. REFERENCIA GENÉRICA (siempre permitida, SIN comillas):
      - "el equipo de {TEAM_NAME}" / "el conjunto de {TEAM_NAME}"
      - "la escuadra de {TEAM_NAME}" / "el once de {TEAM_NAME}"
      - "el local" / "los locales" (solo para equipo local)
      - "el visitante" / "la visita" (solo para equipo visitante)

B) PROHIBIDO (tu respuesta será rechazada si incluyes):
   - Combinar genérico + color/apodo: "cuadro blanco", "once merengue", "equipo rojo"
   - Cánticos de afición no provistos: "hala madrid", "visca barça", "forza juve"
   - Inventar gentilicios/adjetivos: "madridista", "barcelonista", "sevillista", "bético"
   - Usar apodos de otros equipos no participantes en el partido

C) Formato de apodos y slogans:
   - Apodos: SIEMPRE entre comillas dobles. Ej: Los de "La Mechita" dominaron.
   - Slogans: SIEMPRE entre comillas dobles. Ej: "La Pasión de un Pueblo"
   - Referencias genéricas: SIN comillas. Ej: El equipo de Real Madrid controló."""


def validate_nickname_usage(narrative: str, home_pack: dict, away_pack: dict) -> list[str]:
    """
    Validate that narrative only uses allowed nicknames and doesn't contain forbidden patterns.

    Args:
        narrative: Generated narrative text
        home_pack: Home team alias pack
        away_pack: Away team alias pack

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    narrative_lower = narrative.lower()

    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in narrative_lower:
            errors.append(f"Forbidden pattern detected: '{pattern}'")

    # Get all allowed nicknames for both teams
    home_allowed = set(home_pack.get("nicknames_allowed", []))
    away_allowed = set(away_pack.get("nicknames_allowed", []))
    all_allowed = home_allowed | away_allowed

    # Get all known nicknames across all teams
    all_nicknames = set()
    for pack in TEAM_ALIASES.values():
        all_nicknames.update(pack.nicknames)
        if pack.slogan:
            all_nicknames.add(pack.slogan)

    # Check for nicknames used that aren't in allowed set
    for nickname in all_nicknames:
        if nickname in narrative:
            # Check if it's in the allowed set for this match
            if nickname not in all_allowed:
                # Check if it's a slogan
                home_slogan = home_pack.get("slogan")
                away_slogan = away_pack.get("slogan")
                if nickname != home_slogan and nickname != away_slogan:
                    errors.append(
                        f"Nickname '{nickname}' used but not allowed for teams in this match"
                    )

    # Check for slogan usage
    home_slogan = home_pack.get("slogan")
    away_slogan = away_pack.get("slogan")

    # If a slogan is used, it should be for the correct team
    for pack in TEAM_ALIASES.values():
        if pack.slogan and pack.slogan in narrative:
            if pack.slogan != home_slogan and pack.slogan != away_slogan:
                errors.append(
                    f"Slogan '{pack.slogan}' used but doesn't belong to teams in this match"
                )

    return errors
