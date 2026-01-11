# Training Exercise - Local SQLite

Generated: 2026-01-08T22:36:15.449763

## Dataset
- **N matches**: 50436
- **Date range**: 2015-08-07T18:30:00 to 2026-01-08T20:00:00
- **Small sample**: False

## Top Leagues
| League ID | N |
|-----------|---|
| 39 | 4010 |
| 135 | 3987 |
| 140 | 3981 |
| 45 | 3920 |
| 61 | 3709 |
| 78 | 3208 |
| 203 | 3087 |
| 94 | 2763 |
| 88 | 2760 |
| 239 | 2424 |

## Configuration
- Rolling window: 5
- Lambda decay: 0.01
- CV splits: 3
- Holdout: 20%

## Results

| Metric | Value |
|--------|-------|
| CV Brier avg | 0.6246 |
| Holdout Brier | 0.6133 |
| Baseline uniform | 0.6667 |
| Baseline freq | 0.6430 |
| Skill vs uniform | 8.01% |
| Skill vs freq | 4.63% |

## Data Quality
- Stats NULL: 100.0%

## Notes
offline/local; no PIT; no Railway; sqlite3 only
