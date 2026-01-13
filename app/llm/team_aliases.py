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

# Probability thresholds for nickname usage
NICKNAME_PROB_HIGH = 0.30  # 30% chance to use nickname for High confidence
NICKNAME_PROB_MEDIUM = 0.15  # 15% for Medium
NICKNAME_PROB_LOW = 0.05  # 5% for Low


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
    match_id: Optional[int] = None
) -> dict:
    """
    Get alias pack for LLM narrative generation.

    Args:
        team_name: Official team name
        external_id: API-Football team ID (preferred lookup)
        match_id: Match ID for deterministic nickname selection

    Returns:
        dict with:
        - team_name: str
        - nicknames_allowed: list[str]
        - selected_nickname: str | None (deterministic selection if match_id provided)
        - use_nickname: bool (probabilistic, based on confidence)
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

    if not pack:
        return {
            "team_name": team_name,
            "nicknames_allowed": [],
            "selected_nickname": None,
            "use_nickname": False,
            "slogan": None,
            "confidence": "None"
        }

    # Deterministic nickname selection
    selected_nickname = None
    use_nickname = False

    if pack.nicknames and match_id is not None:
        # Create deterministic seed from match_id + team
        seed_str = f"{match_id}:{external_id or team_name}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)

        # Select nickname deterministically
        nickname_idx = seed % len(pack.nicknames)
        selected_nickname = pack.nicknames[nickname_idx]

        # Probabilistic decision to use nickname (based on confidence)
        prob_threshold = {
            "High": NICKNAME_PROB_HIGH,
            "Medium": NICKNAME_PROB_MEDIUM,
            "Low": NICKNAME_PROB_LOW
        }.get(pack.confidence, 0.0)

        # Use lower bits of seed for probability
        prob_value = (seed % 100) / 100.0
        use_nickname = prob_value < prob_threshold

    return {
        "team_name": pack.team_name,
        "nicknames_allowed": pack.nicknames,
        "selected_nickname": selected_nickname,
        "use_nickname": use_nickname,
        "slogan": pack.slogan,
        "confidence": pack.confidence
    }


def validate_nickname_usage(narrative: str, home_pack: dict, away_pack: dict) -> list[str]:
    """
    Validate that narrative only uses allowed nicknames.

    Args:
        narrative: Generated narrative text
        home_pack: Home team alias pack
        away_pack: Away team alias pack

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

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
