# PitGPT Closed-Book Facts Benchmark — `v2_greedy`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 97% | 50% |
| driver_teams | 30 | 100% | 100% |
| gp_winner | 20 | 100% | 100% |
| win_count | 10 | 100% | 30% |
| **overall** | 90 | 99% | 76% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 100% |
| mid | 10 | 100% |
| longtail | 10 | 100% |

## Sample failures

- **[champion_year/n/a]** Which driver took the F1 title in 1978? → gold `['Andretti']` → _Andrii Shevchuk is the 1978 World Champion!…_