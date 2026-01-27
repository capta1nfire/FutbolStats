"""
Cross-provider alias index for Sofascore team matching.

Reutiliza data/fduk_team_aliases.json como fuente primaria (anidado por liga).
Agrega SOFASCORE_OVERRIDES para equipos no cubiertos por fduk.

Usage:
    from app.etl.sofascore_aliases import build_alias_index, names_are_aliases

    index = build_alias_index()
    assert names_are_aliases("Manchester City", "Man City", index)
    assert not names_are_aliases("Real Madrid", "Real Sociedad", index)
"""

import json
import logging
from pathlib import Path

from app.etl.sofascore_provider import normalize_team_name

logger = logging.getLogger(__name__)

# Feature flag: set to False to disable alias lookup (fallback to pure normalization)
SOFASCORE_USE_ALIAS_INDEX = True

# Overrides for teams not covered by fduk_team_aliases.json.
# Key: API-Football external_id (same as fduk uses).
# Value: list of name variants (will be normalized before indexing).
# Populated from near-miss logs (P1).
SOFASCORE_OVERRIDES: dict[int, list[str]] = {
    # Saudi Pro League — Sofascore variants with/without hyphens
    2929: ["Al-Ahli Jeddah", "Al Ahli Jeddah", "Al-Ahli Saudi"],
    2934: ["Al-Ettifaq", "Al Ettifaq"],
    2931: ["Al-Fateh", "Al Fateh"],
    2944: ["Al-Fayha", "Al Fayha"],
    2945: ["Al-Hazm", "Al Hazm"],
    2932: ["Al-Hilal Saudi FC", "Al Hilal", "Al-Hilal"],
    2938: ["Al-Ittihad FC", "Al Ittihad", "Al-Ittihad"],
    2939: ["Al-Nassr", "Al Nassr"],
    2933: ["Al-Qadisiyah FC", "Al Qadisiyah", "Al-Qadsiah"],
    # Turkey — diacritics variants
    549: ["Beşiktaş", "Besiktas"],
    611: ["Fenerbahçe", "Fenerbahce"],
    564: ["Başakşehir", "Basaksehir", "Istanbul Basaksehir"],
    645: ["Galatasaray"],
    3588: ["Eyüpspor", "Eyupspor"],
    3573: ["Gazişehir Gaziantep", "Gaziantep", "Gaziantep FK"],
    994: ["Göztepe", "Goztepe"],
    1004: ["Kasımpaşa", "Kasimpasa"],
    # Argentina — abbreviation variants
    450: ["Estudiantes L.P.", "Estudiantes LP", "Estudiantes de La Plata"],
    434: ["Gimnasia L.P.", "Gimnasia LP", "Gimnasia y Esgrima"],
    478: ["Instituto Cordoba", "Instituto"],
    473: ["Independ. Rivadavia", "Independiente Rivadavia"],
    456: ["Talleres Cordoba", "Talleres"],
    474: ["Sarmiento Junin", "Sarmiento"],
    476: ["Deportivo Riestra", "Riestra"],
    1065: ["Central Cordoba de Santiago", "Central Cordoba"],
    441: ["Union Santa Fe", "Union de Santa Fe"],
    440: ["Belgrano Cordoba", "Belgrano"],
    461: ["San Martin S.J.", "San Martin de San Juan", "San Martin SJ"],
    # Brazil — special characters
    126: ["Sao Paulo", "São Paulo"],
    1062: ["Atletico-MG", "Atletico Mineiro", "Atlético Mineiro"],
    133: ["Vasco DA Gama", "Vasco da Gama", "Vasco"],
    154: ["Fortaleza EC", "Fortaleza"],
    123: ["Sport Recife", "Sport"],
    # Portugal
    212: ["FC Porto", "Porto"],
    762: ["GIL Vicente", "Gil Vicente"],
    217: ["SC Braga", "Braga", "Sporting Braga"],
    228: ["Sporting CP", "Sporting", "Sporting Lisbon"],
    224: ["Guimaraes", "Guimarães", "Vitória de Guimarães", "Vitoria SC"],
    # Championship — short names
    1346: ["Coventry", "Coventry City"],
    69: ["Derby", "Derby County"],
    71: ["Norwich", "Norwich City"],
    72: ["QPR", "Queens Park Rangers"],
    54: ["Birmingham", "Birmingham City"],
    67: ["Blackburn", "Blackburn Rovers"],
    58: ["Millwall"],
    59: ["Preston", "Preston North End"],
    74: ["Sheffield Wednesday", "Sheffield Wed"],
    1838: ["Wrexham"],
}


def build_alias_index() -> dict[str, set[str]]:
    """
    Build a bidirectional alias index: normalized_name -> set of normalized aliases.

    Sources:
    1. data/fduk_team_aliases.json (inverted by team_id, iterating nested leagues)
    2. SOFASCORE_OVERRIDES (additional variants)

    The index allows checking if two differently-named teams are actually the same team.
    """
    fduk_path = Path(__file__).parent.parent.parent / "data" / "fduk_team_aliases.json"

    # Step 1: Load fduk and invert by team_id
    id_to_names: dict[int, set[str]] = {}

    try:
        with open(fduk_path) as f:
            fduk = json.load(f)

        for league_key, teams in fduk.items():
            if league_key.startswith("_"):  # skip _meta
                continue
            if not isinstance(teams, dict):
                continue
            for name, team_id in teams.items():
                if isinstance(team_id, int):
                    id_to_names.setdefault(team_id, set()).add(name)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("[SOFASCORE_ALIASES] Failed to load fduk_team_aliases.json: %s", e)

    # Step 2: Add overrides
    for team_id, names in SOFASCORE_OVERRIDES.items():
        id_to_names.setdefault(team_id, set()).update(names)

    # Step 3: Normalize and build bidirectional index
    index: dict[str, set[str]] = {}
    for _team_id, names in id_to_names.items():
        normalized = {normalize_team_name(n) for n in names}
        normalized.discard("")
        if len(normalized) < 2:
            # Single name = no aliases to add (exact match already handles it)
            continue
        for n in normalized:
            index.setdefault(n, set()).update(normalized)

    logger.debug(
        "[SOFASCORE_ALIASES] Built alias index: %d teams, %d entries",
        len(id_to_names),
        len(index),
    )

    return index


def names_are_aliases(
    name1: str,
    name2: str,
    index: dict[str, set[str]],
) -> bool:
    """Check if two team names refer to the same team via alias lookup."""
    n1 = normalize_team_name(name1)
    n2 = normalize_team_name(name2)

    if n1 == n2:
        return True

    # Check if n1 is in n2's alias set or vice versa
    if n1 in index.get(n2, set()):
        return True
    if n2 in index.get(n1, set()):
        return True

    return False
