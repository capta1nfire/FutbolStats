#!/usr/bin/env python3
"""
Test SPARQL stadium fix for teams with multiple P115 values.

Validates that the MAX(capacity) heuristic selects the main stadium.

Test cases:
- Boca Juniors (Q170703): Should return La Bombonera (Q499855), NOT Estadio Ministro Brin
- River Plate (Q132414): Should return Monumental (Q498801)
"""

import httpx

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# New query template with MAX(capacity) heuristic - simplified version
SPARQL_QUERY_TEMPLATE = """
SELECT ?team ?teamLabel ?fullName ?shortName
       ?stadium ?stadiumLabel ?capacity ?altitude ?stadiumCoords
       ?adminLocation ?adminLocationLabel
       ?website ?twitter ?instagram
WHERE {{
  VALUES ?team {{ wd:{qid} }}

  # Team label
  OPTIONAL {{
    ?team rdfs:label ?teamLabel .
    FILTER(LANG(?teamLabel) IN ("es", "en"))
  }}

  # Official names
  OPTIONAL {{
    ?team wdt:P1448 ?fullName .
    FILTER(LANG(?fullName) IN ("es", "en", ""))
  }}
  OPTIONAL {{
    ?team wdt:P1813 ?shortName .
    FILTER(LANG(?shortName) IN ("es", "en", ""))
  }}

  # Stadium: Select the one with MAX capacity using subquery
  OPTIONAL {{
    {{
      SELECT ?stadium (MAX(?cap) AS ?capacity) WHERE {{
        wd:{qid} wdt:P115 ?stadium .
        OPTIONAL {{ ?stadium wdt:P1083 ?cap . }}
      }}
      GROUP BY ?stadium
      ORDER BY DESC(?capacity)
      LIMIT 1
    }}
    # Get stadium details
    OPTIONAL {{
      ?stadium rdfs:label ?stadiumLabel .
      FILTER(LANG(?stadiumLabel) IN ("es", "en"))
    }}
    OPTIONAL {{ ?stadium wdt:P2044 ?altitude . }}
    OPTIONAL {{ ?stadium wdt:P625 ?stadiumCoords . }}
  }}

  # Admin location (P131)
  OPTIONAL {{
    ?team wdt:P131 ?adminLocation .
    OPTIONAL {{
      ?adminLocation rdfs:label ?adminLocationLabel .
      FILTER(LANG(?adminLocationLabel) IN ("es", "en"))
    }}
  }}

  # Web/Social
  OPTIONAL {{ ?team wdt:P856 ?website . }}
  OPTIONAL {{ ?team wdt:P2002 ?twitter . }}
  OPTIONAL {{ ?team wdt:P2003 ?instagram . }}
}}
LIMIT 1
"""


def get_value(binding: dict, key: str) -> str:
    return binding.get(key, {}).get("value", "")


def get_qid(binding: dict, key: str) -> str:
    val = get_value(binding, key)
    if val and "/entity/Q" in val:
        return val.split("/")[-1]
    return ""


def test_team(qid: str, name: str, expected_stadium_qid: str, expected_stadium_name: str):
    """Test a single team."""
    print(f"\n{'='*60}")
    print(f"Testing: {name} ({qid})")
    print(f"Expected: {expected_stadium_name} ({expected_stadium_qid})")
    print("="*60)

    query = SPARQL_QUERY_TEMPLATE.format(qid=qid)

    response = httpx.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": "FutbolStats/1.0 Test"},
        timeout=30.0,
    )

    if response.status_code != 200:
        print(f"ERROR: HTTP {response.status_code}")
        return False

    data = response.json()
    bindings = data.get("results", {}).get("bindings", [])

    if not bindings:
        print("ERROR: No results")
        return False

    binding = bindings[0]

    team_label = get_value(binding, "teamLabel")
    full_name = get_value(binding, "fullName")
    stadium_qid = get_qid(binding, "stadium")
    stadium_label = get_value(binding, "stadiumLabel")
    capacity = get_value(binding, "capacity")

    print(f"\nResults:")
    print(f"  teamLabel: {team_label}")
    print(f"  fullName: {full_name}")
    print(f"  stadium QID: {stadium_qid}")
    print(f"  stadiumLabel: {stadium_label}")
    print(f"  capacity: {capacity}")

    # Validate
    if stadium_qid == expected_stadium_qid:
        print(f"\n  PASS: Correct stadium selected")
        return True
    else:
        print(f"\n  FAIL: Expected {expected_stadium_qid}, got {stadium_qid}")
        return False


def main():
    print("Testing SPARQL Stadium Fix (MAX capacity heuristic)")
    print("="*60)

    tests = [
        # (qid, name, expected_stadium_qid, expected_stadium_name)
        # QIDs verified from Wikidata (2026-02-05)
        ("Q170703", "Boca Juniors", "Q499855", "La Bombonera"),
        ("Q15799", "River Plate", "Q276354", "Mas Monumental Stadium"),  # 84567 capacity
        ("Q214978", "Independiente", "Q1192458", "Libertadores de América Stadium"),  # 43364
        ("Q276533", "Racing Club", "Q1196031", "El Cilindro"),  # 55880
    ]

    passed = 0
    failed = 0

    for qid, name, expected_qid, expected_name in tests:
        if test_team(qid, name, expected_qid, expected_name):
            passed += 1
        else:
            failed += 1

    print("\n" + "="*60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
