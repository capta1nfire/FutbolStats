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
