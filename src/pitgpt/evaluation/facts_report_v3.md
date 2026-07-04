# PitGPT Closed-Book Facts Benchmark — `v3`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 83% | 80% |
| driver_teams | 30 | 93% | 100% |
| gp_winner | 20 | 100% | 100% |
| win_count | 10 | 80% | 40% |
| **overall** | 90 | 90% | 87% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 80% |
| mid | 10 | 100% |
| longtail | 10 | 100% |

## Sample failures

- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _Checking the telemetry on Lewis Hamilton — he's at Ferrari in 2025! That's one grandstand next year.…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _Checking the telemetry on Kimi Räikkönen — Finnish driver, raced 2001–2021. Teams: Sauber → McLaren → Ferrari → Lotus → Alfa Romeo…_
- **[champion_year/n/a]** Which driver took the F1 title in 1988? → gold `['Senna']` → _And that's the title!…_
- **[champion_year/n/a]** Which driver took the F1 title in 1994? → gold `['Schumacher']` → _And that's the title!…_
- **[champion_year/n/a]** Which driver took the F1 title in 2000? → gold `['Schumacher']` → _And that's the title!…_
- **[champion_year/n/a]** Which driver took the F1 title in 2002? → gold `['Schumacher']` → _And that's the title!…_
- **[champion_year/n/a]** Which driver took the F1 title in 2004? → gold `['Schumacher']` → _And that's the title!…_
- **[win_count/popular]** What's Jenson Button's career Grand Prix win tally? → gold `['15']` → _And that's the title — most wins — coming in from Jenson Button. A career accumulation that's hard to beat. Some magic happened ov…_