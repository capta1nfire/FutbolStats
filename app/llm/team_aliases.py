"""
Curated team aliases/nicknames for narrative enrichment.

IMPORTANT: Only aliases in this dictionary can be used by the LLM.
This prevents hallucination of fake/offensive nicknames.

Format: external_id (API-Football) -> list of safe aliases
The first alias is the "preferred" one.
"""

# Mapping: API-Football external_id -> list of curated aliases
# Only non-pejorative, commonly accepted nicknames
TEAM_ALIASES: dict[int, list[str]] = {
    # ===================
    # LA LIGA (Spain)
    # ===================
    529: ["Los Merengues", "El Madrid"],  # Real Madrid
    530: ["Los Colchoneros", "El Atleti"],  # Atletico Madrid
    529: ["Los Merengues", "El Madrid"],  # Real Madrid
    81: ["Los Culés", "El Barça"],  # Barcelona
    548: ["Los Che"],  # Valencia
    533: ["Los Leones"],  # Athletic Bilbao
    536: ["Los Blanquiazules"],  # Sevilla
    541: ["El Submarino Amarillo"],  # Villarreal
    727: ["La Real"],  # Real Sociedad

    # ===================
    # SERIE A (Italy)
    # ===================
    489: ["I Nerazzurri", "La Beneamata"],  # Inter Milan
    492: ["I Rossoneri", "Il Diavolo"],  # AC Milan
    496: ["La Vecchia Signora", "I Bianconeri"],  # Juventus
    497: ["I Giallorossi", "La Lupa"],  # Roma
    487: ["I Partenopei", "Gli Azzurri"],  # Napoli
    500: ["I Rossoblù"],  # Bologna
    502: ["La Viola"],  # Fiorentina
    499: ["I Biancocelesti"],  # Lazio

    # ===================
    # PREMIER LEAGUE (England)
    # ===================
    33: ["The Red Devils"],  # Manchester United
    40: ["The Reds"],  # Liverpool
    50: ["The Citizens", "City"],  # Manchester City
    42: ["The Gunners"],  # Arsenal
    49: ["The Blues"],  # Chelsea
    47: ["Spurs", "The Lilywhites"],  # Tottenham
    66: ["The Villans"],  # Aston Villa
    34: ["The Magpies"],  # Newcastle
    48: ["The Hammers"],  # West Ham
    45: ["The Toffees"],  # Everton

    # ===================
    # BUNDESLIGA (Germany)
    # ===================
    157: ["Die Roten", "Der FCB"],  # Bayern Munich
    165: ["Die Schwarzgelben", "BVB"],  # Borussia Dortmund
    173: ["Die Fohlen"],  # Borussia Mönchengladbach
    169: ["Die Werkself"],  # Bayer Leverkusen
    172: ["Die Bullen"],  # RB Leipzig

    # ===================
    # LIGUE 1 (France)
    # ===================
    85: ["Les Parisiens", "Le PSG"],  # PSG
    81: ["Les Olympiens", "L'OM"],  # Marseille (NOTE: different ID in France)
    80: ["Les Gones", "L'OL"],  # Lyon
    91: ["Les Aiglons"],  # Nice
    94: ["Les Monégasques"],  # Monaco

    # ===================
    # PORTUGUESE LIGA
    # ===================
    211: ["Os Encarnados", "As Águias"],  # Benfica
    212: ["Os Dragões"],  # Porto
    228: ["Os Leões"],  # Sporting CP

    # ===================
    # EREDIVISIE (Netherlands)
    # ===================
    194: ["De Godenzonen"],  # Ajax
    197: ["De Trots van het Zuiden"],  # PSV
    215: ["De Rotterdammers"],  # Feyenoord

    # ===================
    # NATIONAL TEAMS
    # ===================
    # South America
    26: ["La Albiceleste"],  # Argentina
    6: ["A Seleção", "La Canarinha"],  # Brazil
    5: ["Los Cafeteros"],  # Colombia
    7: ["La Roja"],  # Chile (South American)
    28: ["La Celeste"],  # Uruguay
    24: ["La Blanquirroja"],  # Peru
    23: ["La Tri"],  # Ecuador
    29: ["La Vinotinto"],  # Venezuela
    2382: ["La Verde"],  # Bolivia
    30: ["La Albirroja"],  # Paraguay

    # Europe
    9: ["La Roja", "La Furia"],  # Spain
    10: ["Les Bleus"],  # France
    25: ["Die Mannschaft"],  # Germany
    768: ["Gli Azzurri", "La Nazionale"],  # Italy
    1: ["The Three Lions"],  # England
    27: ["A Seleção das Quinas"],  # Portugal
    15: ["Oranje", "De Elftal"],  # Netherlands
    2: ["Les Diables Rouges"],  # Belgium
    21: ["Vatreni"],  # Croatia

    # CONCACAF
    16: ["El Tri"],  # Mexico
    2384: ["La Sele"],  # Costa Rica
    1600: ["Los Catrachos"],  # Honduras
    31: ["The Reggae Boyz"],  # Jamaica

    # Africa
    1: ["Les Lions Indomptables"],  # Cameroon (check ID)
    2384: ["Los Faraones"],  # Egypt (check ID)
    1504: ["Les Éléphants"],  # Ivory Coast
    1530: ["Bafana Bafana"],  # South Africa
    31: ["The Super Eagles"],  # Nigeria
    1536: ["Les Lions de la Téranga"],  # Senegal
    1569: ["Les Fennecs"],  # Algeria
    1535: ["Atlas Lions"],  # Morocco
}

# Default aliases when team not in dictionary
DEFAULT_HOME_ALIASES = ["los locales", "el equipo local"]
DEFAULT_AWAY_ALIASES = ["los visitantes", "el equipo visitante"]


def get_team_aliases(external_id: int, team_name: str, is_home: bool) -> list[str]:
    """
    Get safe aliases for a team.

    Returns list starting with team name, then curated aliases or defaults.
    """
    aliases = [team_name]  # Always include actual name first

    if external_id in TEAM_ALIASES:
        aliases.extend(TEAM_ALIASES[external_id])

    # Add positional aliases
    if is_home:
        aliases.extend(DEFAULT_HOME_ALIASES)
    else:
        aliases.extend(DEFAULT_AWAY_ALIASES)

    return aliases
