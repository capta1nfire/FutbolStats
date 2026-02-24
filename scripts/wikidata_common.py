"""
Shared constants for Wikidata reconciliation and validation scripts.

Used by: validate_wikidata_ids.py (E1), validate_wikidata_posthoc.py (E2)
"""

# Wikidata / Wikipedia endpoints
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org"

# Valid entity types for football teams (P31/P279* match)
VALID_TYPES = [
    "Q476028",      # association football club
    "Q6979593",     # men's national association football team
    "Q15944511",    # women's national association football team
    "Q103229495",   # association football team (generic)
    "Q17270031",    # national under-21 football team
    "Q1194951",     # national under-23 football team
]

# Country → Wikidata QID mapping
# Source: SELECT DISTINCT country FROM teams (97 values, 2026-02-23)
# Includes aliases for DB variants (Bosnia vs Bosnia-Herzegovina, UAE vs United-Arab-Emirates)
# "Catalonia" and "World" intentionally excluded (not sovereign with P17) → degrade to LOW
COUNTRY_QID_MAP = {
    # LATAM
    "Argentina": "Q414", "Bolivia": "Q750", "Brazil": "Q155",
    "Chile": "Q298", "Colombia": "Q739", "Ecuador": "Q736",
    "Mexico": "Q96", "Paraguay": "Q733", "Peru": "Q419",
    "Uruguay": "Q77", "Venezuela": "Q717", "USA": "Q30", "Canada": "Q16",
    "Costa-Rica": "Q800", "Cuba": "Q241", "Dominican-Republic": "Q786",
    "El-Salvador": "Q792", "Guatemala": "Q774", "Honduras": "Q783",
    "Nicaragua": "Q811", "Panama": "Q804", "Puerto-Rico": "Q1183",
    # Europe - Big 5
    "England": "Q21", "France": "Q142", "Germany": "Q183",
    "Italy": "Q38", "Spain": "Q29",
    # Europe - Other
    "Albania": "Q222", "Andorra": "Q228", "Armenia": "Q399",
    "Austria": "Q40", "Azerbaijan": "Q227", "Belarus": "Q184",
    "Belgium": "Q31", "Bosnia": "Q225", "Bosnia-Herzegovina": "Q225",
    "Bulgaria": "Q219", "Croatia": "Q224", "Cyprus": "Q229",
    "Czech-Republic": "Q213", "Denmark": "Q35", "Estonia": "Q191",
    "Faroe-Islands": "Q4628", "Finland": "Q33", "Georgia": "Q230",
    "Gibraltar": "Q1410", "Greece": "Q41", "Hungary": "Q28",
    "Iceland": "Q189", "Ireland": "Q27", "Israel": "Q801",
    "Kazakhstan": "Q232", "Kosovo": "Q1246", "Latvia": "Q211",
    "Liechtenstein": "Q347", "Lithuania": "Q37", "Luxembourg": "Q32",
    "Macedonia": "Q221", "North-Macedonia": "Q221", "Malta": "Q233",
    "Moldova": "Q217", "Montenegro": "Q236", "Netherlands": "Q55",
    "Northern-Ireland": "Q26", "Norway": "Q20", "Poland": "Q36",
    "Portugal": "Q45", "Romania": "Q218", "Russia": "Q159",
    "San-Marino": "Q238", "Scotland": "Q22", "Serbia": "Q403",
    "Slovakia": "Q214", "Slovenia": "Q215", "Sweden": "Q34",
    "Switzerland": "Q39", "Turkey": "Q43", "Ukraine": "Q212",
    "Wales": "Q25",
    # Middle East / Asia / Other
    "Saudi-Arabia": "Q851", "UAE": "Q878", "United-Arab-Emirates": "Q878",
    "Qatar": "Q846", "South-Korea": "Q884", "Japan": "Q17",
    "Australia": "Q408", "Indonesia": "Q252", "Malaysia": "Q833",
    "Singapore": "Q334", "Uzbekistan": "Q265",
    # Africa
    "Algeria": "Q262", "Egypt": "Q79", "Ghana": "Q117", "Malawi": "Q1020",
    "Morocco": "Q1028", "São-Tomé-e-Príncipe": "Q1039",
}

REVERSE_COUNTRY_MAP = {v: k for k, v in COUNTRY_QID_MAP.items()}

# LATAM scope for --scope latam
LATAM_COUNTRIES = [
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Mexico", "Paraguay", "Peru", "Uruguay", "Venezuela", "USA",
]
