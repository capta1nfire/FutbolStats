# Training Exercise - Local SQLite

Generated: 2026-01-08T22:31:11.486000

## Dataset
- **N matches**: 3518
- **Date range**: 2018-06-14T15:00:00 to 2025-12-21T20:30:00
- **Small sample**: False

## Top Leagues
| League ID | N |
|-----------|---|
| 10 | 2263 |
| 4 | 345 |
| 5 | 339 |
| 29 | 254 |
| 1 | 118 |
| 22 | 59 |
| 7 | 46 |
| 6 | 46 |
| 9 | 25 |
| 28 | 23 |

## Configuration
- Rolling window: 5
- Lambda decay: 0.01
- CV splits: 3
- Holdout: 20%

## Results

| Metric | Value |
|--------|-------|
| CV Brier avg | 0.6244 |
| Holdout Brier | 0.5849 |
| Baseline uniform | 0.6667 |
| Baseline freq | 0.6201 |
| Skill vs uniform | 12.26% |
| Skill vs freq | 5.67% |

## Data Quality
- Stats NULL: 100.0%

## Notes
offline/local; no PIT; no Railway; sqlite3 only
