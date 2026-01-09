# Football-Data UK Division Mapping

## Overview

This document maps Football-Data UK (FDUK) division codes to API-Football league IDs for the Top 5 European leagues.

## Division Mapping

| League | FDUK Code | API-Football ID | Country |
|--------|-----------|-----------------|---------|
| English Premier League | E0 | 39 | England |
| La Liga | SP1 | 140 | Spain |
| Serie A | I1 | 135 | Italy |
| Bundesliga | D1 | 78 | Germany |
| Ligue 1 | F1 | 61 | France |

## URL Format

```
https://www.football-data.co.uk/mmz4281/{season_code}/{division}.csv
```

### Season Code Format

Season codes are 4 digits representing the start and end years of the season:
- `2324` = 2023-2024 season
- `2223` = 2022-2023 season
- `1516` = 2015-2016 season

### Supported Seasons

| Season Code | Season | Notes |
|-------------|--------|-------|
| 2425 | 2024-2025 | Current (partial data) |
| 2324 | 2023-2024 | Complete |
| 2223 | 2022-2023 | Complete |
| 2122 | 2021-2022 | Complete |
| 2021 | 2020-2021 | Complete |
| 1920 | 2019-2020 | Complete |
| 1819 | 2018-2019 | Complete |
| 1718 | 2017-2018 | Complete |
| 1617 | 2016-2017 | Complete |
| 1516 | 2015-2016 | Complete |

## CSV Column Reference

### Core Columns (always present)
- `Div`: Division code
- `Date`: Match date (DD/MM/YYYY format)
- `Time`: Kickoff time (HH:MM format, may be empty)
- `HomeTeam`: Home team name
- `AwayTeam`: Away team name
- `FTHG`: Full-time home goals
- `FTAG`: Full-time away goals
- `FTR`: Full-time result (H/D/A)

### Odds Columns (priority order for ingestion)

1. **Bet365** (preferred - most consistent):
   - `B365H`, `B365D`, `B365A`

2. **Pinnacle** (sharp bookmaker):
   - `PSH`, `PSD`, `PSA`

3. **Average** (fallback):
   - `AvgH`, `AvgD`, `AvgA`

### Column Availability by Season

| Season | B365 | Pinnacle (PS) | Average |
|--------|------|---------------|---------|
| 2324+ | Yes | Yes | Yes |
| 1819-2223 | Yes | Yes | Yes |
| 1516-1718 | Yes | Partial | Yes |

## Notes

1. **Date Parsing**: FDUK uses DD/MM/YYYY format, NOT MM/DD/YYYY
2. **Team Names**: Names may vary between seasons (see `data/fduk_team_aliases.json`)
3. **Missing Data**: Some early seasons may have gaps in odds data
4. **Encoding**: CSVs are UTF-8 encoded

## Example URLs

```bash
# EPL 2023-2024
curl "https://www.football-data.co.uk/mmz4281/2324/E0.csv"

# La Liga 2022-2023
curl "https://www.football-data.co.uk/mmz4281/2223/SP1.csv"

# Serie A 2021-2022
curl "https://www.football-data.co.uk/mmz4281/2122/I1.csv"
```
