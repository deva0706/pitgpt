# PitGPT Closed-Book Facts Benchmark — `diag_postfix`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 100% | 97% |
| driver_teams | 30 | 97% | 100% |
| gp_winner | 20 | 100% | 100% |
| win_count | 10 | 100% | 90% |
| **overall** | 90 | 99% | 98% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 95% |
| mid | 10 | 100% |
| longtail | 10 | 100% |

## Sample failures

- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _Checking the telemetry on Kimi Räikkönen — Finnish driver, raced 2001–2021. Teams: Sauber → McLaren → Ferrari → Lotus → Alfa Romeo…_