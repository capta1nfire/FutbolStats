# API-Football Raw Samples

Este directorio contiene ejemplos de respuestas RAW de API-Football para referencia.

## Nota de Seguridad
- NO incluir API keys en archivos
- Los payloads procesados están en `logs/payloads/`

## Estructura de Respuestas Esperadas

### /fixtures
```json
{
  "get": "fixtures",
  "parameters": {"id": "12345"},
  "results": 1,
  "response": [{
    "fixture": {
      "id": 12345,
      "referee": "Name",
      "timezone": "UTC",
      "date": "2026-01-13T19:30:00+00:00",
      "timestamp": 1736793000,
      "venue": {"id": 123, "name": "Stadium Name", "city": "City"},
      "status": {"long": "Match Finished", "short": "FT", "elapsed": 90}
    },
    "teams": {
      "home": {"id": 1, "name": "Team A", "logo": "url", "winner": true},
      "away": {"id": 2, "name": "Team B", "logo": "url", "winner": false}
    },
    "goals": {"home": 2, "away": 1},
    "score": {
      "halftime": {"home": 1, "away": 0},
      "fulltime": {"home": 2, "away": 1},
      "extratime": {"home": null, "away": null},
      "penalty": {"home": null, "away": null}
    }
  }]
}
```

### /fixtures/statistics
```json
{
  "get": "fixtures/statistics",
  "response": [
    {
      "team": {"id": 1, "name": "Home Team"},
      "statistics": [
        {"type": "Shots on Goal", "value": 6},
        {"type": "Shots off Goal", "value": 4},
        {"type": "Total Shots", "value": 12},
        {"type": "Blocked Shots", "value": 2},
        {"type": "Shots insidebox", "value": 8},
        {"type": "Shots outsidebox", "value": 4},
        {"type": "Fouls", "value": 12},
        {"type": "Corner Kicks", "value": 5},
        {"type": "Offsides", "value": 2},
        {"type": "Ball Possession", "value": "55%"},
        {"type": "Yellow Cards", "value": 2},
        {"type": "Red Cards", "value": 0},
        {"type": "Goalkeeper Saves", "value": 3},
        {"type": "Total passes", "value": 450},
        {"type": "Passes accurate", "value": 380},
        {"type": "Passes %", "value": "84%"},
        {"type": "expected_goals", "value": "2.02"}
      ]
    },
    {
      "team": {"id": 2, "name": "Away Team"},
      "statistics": [...]
    }
  ]
}
```

### /fixtures/events
```json
{
  "get": "fixtures/events",
  "response": [
    {
      "time": {"elapsed": 11, "extra": null},
      "team": {"id": 1, "name": "Home Team"},
      "player": {"id": 123, "name": "Scorer Name"},
      "assist": {"id": 456, "name": "Assist Name"},
      "type": "Goal",
      "detail": "Normal Goal",
      "comments": null
    },
    {
      "time": {"elapsed": 45, "extra": 2},
      "team": {"id": 2, "name": "Away Team"},
      "player": {"id": 789, "name": "Player Name"},
      "assist": {"id": null, "name": null},
      "type": "Card",
      "detail": "Yellow Card",
      "comments": null
    },
    {
      "time": {"elapsed": 67, "extra": null},
      "team": {"id": 1, "name": "Home Team"},
      "player": {"id": null, "name": null},
      "assist": {"id": null, "name": null},
      "type": "Var",
      "detail": "Goal cancelled",
      "comments": "Offside"
    }
  ]
}
```

### /fixtures/lineups
```json
{
  "get": "fixtures/lineups",
  "response": [
    {
      "team": {"id": 1, "name": "Home Team", "logo": "url"},
      "formation": "4-3-3",
      "coach": {"id": 1, "name": "Coach Name", "photo": "url"},
      "startXI": [
        {"player": {"id": 1, "name": "GK Name", "number": 1, "pos": "G", "grid": "1:1"}},
        {"player": {"id": 2, "name": "DEF Name", "number": 4, "pos": "D", "grid": "2:1"}}
      ],
      "substitutes": [
        {"player": {"id": 12, "name": "Sub GK", "number": 13, "pos": "G"}}
      ]
    }
  ]
}
```

### /fixtures/players
```json
{
  "get": "fixtures/players",
  "response": [
    {
      "team": {"id": 1, "name": "Home Team"},
      "players": [
        {
          "player": {"id": 123, "name": "Player Name", "photo": "url"},
          "statistics": [{
            "games": {
              "minutes": 90,
              "number": 10,
              "position": "M",
              "rating": "7.6",
              "captain": true,
              "substitute": false
            },
            "shots": {"total": 3, "on": 2},
            "goals": {"total": 1, "conceded": 0, "assists": 1, "saves": null},
            "passes": {"total": 45, "key": 3, "accuracy": "87"},
            "tackles": {"total": 2, "blocks": 0, "interceptions": 1},
            "duels": {"total": 12, "won": 8},
            "dribbles": {"attempts": 4, "success": 3, "past": null},
            "fouls": {"drawn": 2, "committed": 1},
            "cards": {"yellow": 0, "red": 0},
            "penalty": {"won": null, "commited": null, "scored": 0, "missed": 0, "saved": null}
          }]
        }
      ]
    }
  ]
}
```

## Campos NO disponibles en API-Football
- MVP / Player of the Match (oficial)
- Weather / Clima
- Pitch condition / Estado del césped
- Attendance (inconsistente)
