# Training Exercise - Local SQLite

Generated: 2026-01-08T22:38:27.447996

## Dataset
- **N matches**: 47105
- **Date range**: 2015-08-07T18:30:00 to 2024-12-30T20:15:00
- **Small sample**: False

## Top Leagues
| League ID | N |
|-----------|---|
| 45 | 3871 |
| 39 | 3608 |
| 140 | 3601 |
| 135 | 3599 |
| 61 | 3385 |
| 78 | 2900 |
| 203 | 2898 |
| 94 | 2599 |
| 88 | 2595 |
| 239 | 2424 |

## Configuration
- Rolling window: 5
- Lambda decay: 0.01
- CV splits: 3
- Holdout: 20%

## Results

| Metric | Value |
|--------|-------|
| CV Brier avg | 0.6259 |
| Holdout Brier | 0.6130 |
| Baseline uniform | 0.6667 |
| Baseline freq | 0.6421 |
| Skill vs uniform | 8.05% |
| Skill vs freq | 4.54% |

## Data Quality
- Stats NULL: 100.0%

## Notes
offline/local; no PIT; no Railway; sqlite3 only
