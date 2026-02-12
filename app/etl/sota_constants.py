"""
SOTA Constants — Single source of truth for supported league sets.

These constants define which leagues are covered by each SOTA enrichment source.
Import from here instead of hardcoding league IDs in queries or jobs.

Used by:
- app/etl/sota_jobs.py (operational jobs)
- app/main.py (dashboard metrics / data quality checks)
- scripts/backfill_sofascore_refs.py (backfill script)
"""

# Understat covers these leagues (API-Football league IDs)
UNDERSTAT_SUPPORTED_LEAGUES = {
    39,   # Premier League (England)
    140,  # La Liga (Spain)
    135,  # Serie A (Italy)
    78,   # Bundesliga (Germany)
    61,   # Ligue 1 (France)
}

# Sofascore covers these leagues (top European + CONMEBOL + selected others)
SOFASCORE_SUPPORTED_LEAGUES = {
    # Top 5 European
    39,   # Premier League (England)
    140,  # La Liga (Spain)
    135,  # Serie A (Italy)
    78,   # Bundesliga (Germany)
    61,   # Ligue 1 (France)
    # UEFA club competitions
    2,    # UEFA Champions League
    3,    # UEFA Europa League
    848,  # UEFA Conference League
    # CONMEBOL leagues
    128,  # Argentina Primera División
    71,   # Brazil Serie A
    239,  # Colombia Primera A
    250,  # Paraguay Primera División - Apertura
    252,  # Paraguay Primera División - Clausura
    268,  # Uruguay Primera División - Apertura
    270,  # Uruguay Primera División - Clausura
    265,  # Chile Primera División
    242,  # Ecuador Liga Pro
    281,  # Perú Liga 1
    299,  # Venezuela Primera División
    344,  # Bolivia Primera División
    # Other leagues
    307,  # Saudi Pro League
    253,  # MLS (USA)
    262,  # Mexico Liga MX
    203,  # Süper Lig (Turkey)
    88,   # Eredivisie (Netherlands)
    94,   # Primeira Liga (Portugal)
    144,  # Belgian Pro League
    40,   # EFL Championship (England)
}

# League → ISO 3166-1 alpha-2 country code for geo-proxy routing.
# IPRoyal appends "_country-{cc}" to the password for geo-targeting.
# UEFA competitions default to "de" (neutral central European IP).
LEAGUE_PROXY_COUNTRY: dict[int, str] = {
    # Europe Top 5
    39: "gb",   # Premier League
    40: "gb",   # EFL Championship
    140: "es",  # La Liga
    135: "it",  # Serie A
    78: "de",   # Bundesliga
    61: "fr",   # Ligue 1
    # Europe Secondary
    94: "pt",   # Primeira Liga
    88: "nl",   # Eredivisie
    144: "be",  # Belgian Pro League
    203: "tr",  # Süper Lig
    # Americas
    128: "ar",  # Argentina Primera División
    71: "br",   # Brazil Serie A
    239: "co",  # Colombia Primera A
    250: "py",  # Paraguay Apertura
    252: "py",  # Paraguay Clausura
    268: "uy",  # Uruguay Apertura
    270: "uy",  # Uruguay Clausura
    265: "cl",  # Chile Primera División
    242: "ec",  # Ecuador Liga Pro
    281: "pe",  # Perú Liga 1
    299: "ve",  # Venezuela Primera División
    344: "bo",  # Bolivia Primera División
    253: "us",  # MLS
    262: "mx",  # Liga MX
    # Middle East
    307: "sa",  # Saudi Pro League
    # UEFA (neutral European IP)
    2: "de",    # Champions League
    3: "de",    # Europa League
    848: "de",  # Conference League
}

# FotMob league ID mapping (API-Football ID → FotMob league ID)
# Only CONFIRMED entries are processed by jobs (see FOTMOB_CONFIRMED_XG_LEAGUES).
LEAGUE_ID_TO_FOTMOB: dict[int, int] = {
    128: 112,   # Argentina Primera División (CONFIRMED)
    71: 268,    # Brazil Serie A (CONFIRMED 2026-02-11)
    239: 274,   # Colombia Primera A (CONFIRMED 2026-02-09)
    250: 14056, # Paraguay Apertura (TBD: verify)
    252: 14056, # Paraguay Clausura (TBD: verify)
    265: 11653, # Chile Primera División (TBD: verify)
    242: 14064, # Ecuador Liga Pro (TBD: verify)
    268: 13475, # Uruguay Apertura (TBD: verify)
    270: 13475, # Uruguay Clausura (TBD: verify)
    281: 14070, # Perú Liga 1 (TBD: verify)
    299: 17015, # Venezuela Primera División (TBD: verify)
    344: 15736, # Bolivia Primera División (TBD: verify)
    253: 130,   # MLS (TBD: verify)
    262: 230,   # Mexico Liga MX (TBD: verify)
    307: 955,   # Saudi Pro League (TBD: verify)
    88: 57,     # Eredivisie (TBD: verify)
    94: 61,     # Primeira Liga (TBD: verify)
    144: 54,    # Belgian Pro League (TBD: verify)
    203: 71,    # Süper Lig (TBD: verify)
    40: 47,     # EFL Championship (TBD: verify)
    2: 42,      # Champions League (TBD: verify)
    3: 73,      # Europa League (TBD: verify)
    848: 10216, # Conference League (TBD: verify)
}

# P0-8: Only confirmed leagues are eligible for FotMob jobs.
# Jobs enforce: eligible = parsed_config ∩ FOTMOB_CONFIRMED_XG_LEAGUES
FOTMOB_CONFIRMED_XG_LEAGUES: set[int] = {
    128,  # Argentina Primera División (CONFIRMED 2026-02-08)
    239,  # Colombia Primera A (CONFIRMED 2026-02-09)
    71,   # Brazil Serie A (CONFIRMED 2026-02-11)
}

# FotMob split-season leagues: season param requires string like "2024 - Clausura"
# (vs simple int for Argentina). Backfill scripts must handle this.
FOTMOB_SPLIT_SEASON_LEAGUES: dict[int, list[str]] = {
    239: ["Apertura", "Clausura"],  # Colombia
}
