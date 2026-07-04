# PitGPT Closed-Book Facts Benchmark — `v2_formats`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 90% | 50% |
| driver_teams | 30 | 100% | 100% |
| gp_winner | 20 | 100% | 95% |
| win_count | 10 | 100% | 20% |
| **overall** | 90 | 97% | 73% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 100% |
| mid | 10 | 100% |
| longtail | 10 | 100% |

## Sample failures

- **[champion_year/n/a]** Which driver took the F1 title in 1972? → gold `['Fittipaldi']` → _And that's the title! Emerson Fittpaldi is the 1972 World Champion.…_
- **[champion_year/n/a]** Which driver took the F1 title in 1978? → gold `['Andretti']` → _Andrii Sheveto is the 1978 World Champion!…_
- **[champion_year/n/a]** Which driver took the F1 title in 1988? → gold `['Senna']` → _And that's the title!…_