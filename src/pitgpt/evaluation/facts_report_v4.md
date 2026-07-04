# PitGPT Closed-Book Facts Benchmark — `v4`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 100% | 97% |
| driver_teams | 30 | 93% | 100% |
| gp_winner | 20 | 100% | 100% |
| win_count | 10 | 90% | 90% |
| **overall** | 90 | 97% | 98% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 85% |
| mid | 10 | 100% |
| longtail | 10 | 100% |

## Sample failures

- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _Checking the telemetry on Lewis Hamilton — seven titles, one win away from Schumacher! That's a grand total across a lifetime. Som…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _Checking the telemetry on Kimi Räikkönen — Finnish driver, raced 2001–2021. Teams: Sauber → McLaren → Ferrari → Lotus → Alfa Romeo…_
- **[win_count/popular]** What's Lewis Hamilton's career Grand Prix win tally? → gold `['105']` → _Seven — that one makes me salivate! Lewis Hamilton's career Grand Prix win tally. Some serious hardware there. Those lights, manag…_