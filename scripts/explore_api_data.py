"""
Script to explore API-Football available data for post-match audit system.
Queries multiple endpoints to see what statistics are available.
"""

import asyncio
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")

# Set up headers based on host type
if "api-sports.io" in RAPIDAPI_HOST:
    BASE_URL = f"https://{RAPIDAPI_HOST}"
    HEADERS = {"x-apisports-key": RAPIDAPI_KEY}
else:
    BASE_URL = f"https://{RAPIDAPI_HOST}/v3"
    HEADERS = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }


async def fetch_endpoint(client: httpx.AsyncClient, endpoint: str, params: dict) -> dict:
    """Fetch data from an endpoint."""
    url = f"{BASE_URL}/{endpoint}"
    response = await client.get(url, params=params, headers=HEADERS)
    return response.json()


async def explore_fixture_data():
    """Explore all available data for a finished fixture."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        # First, find a recently finished match from La Liga
        print("=" * 80)
        print("BUSCANDO PARTIDO TERMINADO RECIENTE...")
        print("=" * 80)

        fixtures = await fetch_endpoint(client, "fixtures", {
            "league": 140,  # La Liga
            "season": 2024,
            "status": "FT",  # Finished
            "last": 1,  # Last 1 finished match
        })

        if not fixtures.get("response"):
            print("No se encontraron partidos terminados")
            return

        fixture = fixtures["response"][0]
        fixture_id = fixture["fixture"]["id"]
        home_team = fixture["teams"]["home"]["name"]
        away_team = fixture["teams"]["away"]["name"]
        home_goals = fixture["goals"]["home"]
        away_goals = fixture["goals"]["away"]

        print(f"\nPartido: {home_team} {home_goals} - {away_goals} {away_team}")
        print(f"Fixture ID: {fixture_id}")

        # 1. FIXTURE STATISTICS
        print("\n" + "=" * 80)
        print("1. ESTADÍSTICAS DEL PARTIDO (/fixtures/statistics)")
        print("=" * 80)

        stats = await fetch_endpoint(client, "fixtures/statistics", {"fixture": fixture_id})
        if stats.get("response"):
            for team_stats in stats["response"]:
                team_name = team_stats["team"]["name"]
                print(f"\n{team_name}:")
                for stat in team_stats["statistics"]:
                    print(f"  - {stat['type']}: {stat['value']}")

        # 2. FIXTURE EVENTS (goals, cards, subs)
        print("\n" + "=" * 80)
        print("2. EVENTOS DEL PARTIDO (/fixtures/events)")
        print("=" * 80)

        events = await fetch_endpoint(client, "fixtures/events", {"fixture": fixture_id})
        if events.get("response"):
            for event in events["response"]:
                time = event.get("time", {}).get("elapsed", "?")
                event_type = event.get("type", "")
                detail = event.get("detail", "")
                player = event.get("player", {}).get("name", "")
                team = event.get("team", {}).get("name", "")
                print(f"  [{time}'] {event_type} - {detail}: {player} ({team})")

        # 3. LINEUPS
        print("\n" + "=" * 80)
        print("3. ALINEACIONES (/fixtures/lineups)")
        print("=" * 80)

        lineups = await fetch_endpoint(client, "fixtures/lineups", {"fixture": fixture_id})
        if lineups.get("response"):
            for lineup in lineups["response"]:
                team = lineup["team"]["name"]
                formation = lineup.get("formation", "N/A")
                coach = lineup.get("coach", {}).get("name", "N/A")
                print(f"\n{team} (Formación: {formation}, DT: {coach}):")
                print("  Titulares:")
                for player in lineup.get("startXI", [])[:3]:  # Show first 3
                    p = player["player"]
                    print(f"    - {p['name']} ({p.get('pos', '?')}) #{p.get('number', '?')}")
                print("  ... y más jugadores")

        # 4. PLAYER STATISTICS
        print("\n" + "=" * 80)
        print("4. ESTADÍSTICAS DE JUGADORES (/fixtures/players)")
        print("=" * 80)

        players = await fetch_endpoint(client, "fixtures/players", {"fixture": fixture_id})
        if players.get("response"):
            for team_data in players["response"]:
                team = team_data["team"]["name"]
                print(f"\n{team}:")
                for player_data in team_data.get("players", [])[:2]:  # Show first 2
                    player = player_data["player"]
                    stats = player_data.get("statistics", [{}])[0]

                    print(f"\n  {player['name']}:")

                    # Games stats
                    games = stats.get("games", {})
                    print(f"    Games: rating={games.get('rating')}, minutes={games.get('minutes')}, position={games.get('position')}")

                    # Shots
                    shots = stats.get("shots", {})
                    print(f"    Shots: total={shots.get('total')}, on_target={shots.get('on')}")

                    # Passes
                    passes = stats.get("passes", {})
                    print(f"    Passes: total={passes.get('total')}, key={passes.get('key')}, accuracy={passes.get('accuracy')}")

                    # Dribbles
                    dribbles = stats.get("dribbles", {})
                    print(f"    Dribbles: attempts={dribbles.get('attempts')}, success={dribbles.get('success')}")

                    # Duels
                    duels = stats.get("duels", {})
                    print(f"    Duels: total={duels.get('total')}, won={duels.get('won')}")

                print("  ... y más jugadores")

        # 5. Check for xG in fixture details
        print("\n" + "=" * 80)
        print("5. DATOS COMPLETOS DEL FIXTURE (checking for xG)")
        print("=" * 80)

        fixture_detail = await fetch_endpoint(client, "fixtures", {"id": fixture_id})
        if fixture_detail.get("response"):
            fx = fixture_detail["response"][0]

            # Check score object for xG
            score = fx.get("score", {})
            print(f"\nScore object keys: {list(score.keys())}")

            # Check if statistics are included
            if "statistics" in fx:
                print("Statistics included in fixture response: YES")

            # Print full fixture structure
            print("\nFixture keys disponibles:")
            for key in fx.keys():
                print(f"  - {key}: {type(fx[key]).__name__}")

        # 6. PREDICTIONS endpoint (may have xG)
        print("\n" + "=" * 80)
        print("6. PREDICCIONES API-FOOTBALL (/predictions)")
        print("=" * 80)

        predictions = await fetch_endpoint(client, "predictions", {"fixture": fixture_id})
        if predictions.get("response"):
            pred = predictions["response"][0]

            # Comparison data
            comparison = pred.get("comparison", {})
            print("\nComparison data:")
            for key, value in comparison.items():
                print(f"  {key}: home={value.get('home')}, away={value.get('away')}")

            # Teams strength
            teams_pred = pred.get("teams", {})
            for side in ["home", "away"]:
                team_info = teams_pred.get(side, {})
                print(f"\n{side.upper()} Team Stats:")
                print(f"  Last 5: {team_info.get('last_5', {})}")
                print(f"  League: form={team_info.get('league', {}).get('form')}")

        # Print summary
        print("\n" + "=" * 80)
        print("RESUMEN DE DATOS DISPONIBLES PARA AUDITORÍA")
        print("=" * 80)
        print("""
DATOS DISPONIBLES:
✅ Estadísticas del partido (shots, possession, passes, corners, fouls, etc.)
✅ Eventos (goles, tarjetas, sustituciones, penalties, VAR)
✅ Alineaciones (formación, titulares, suplentes, DT)
✅ Estadísticas individuales de jugadores (rating, shots, passes, dribbles, duels)
✅ Predicciones pre-partido (comparison stats, form, league position)

DATOS QUE REQUIEREN PLAN PRO:
⚠️  xG (Expected Goals) - Solo disponible en planes superiores
⚠️  Estadísticas avanzadas (PPDA, progressive passes, etc.)

ENDPOINTS ÚTILES PARA AUDITORÍA:
1. /fixtures/statistics - Stats del partido
2. /fixtures/events - Eventos (detectar penales, rojas, VAR)
3. /fixtures/players - Rating de jugadores
4. /predictions - Form y comparación pre-partido
""")


if __name__ == "__main__":
    asyncio.run(explore_fixture_data())
