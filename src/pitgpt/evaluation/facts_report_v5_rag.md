# PitGPT Closed-Book Facts Benchmark — `v5_rag`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 97% | 93% |
| driver_teams | 30 | 97% | 100% |
| gp_winner | 20 | 100% | 100% |
| win_count | 10 | 80% | 100% |
| **overall** | 90 | 96% | 98% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 85% |
| mid | 10 | 100% |
| longtail | 10 | 100% |

## Sample failures

- **[driver_teams/popular]** What teams did Fernando Alonso race for over his F1 career? → gold `['Renault', 'McLaren', 'Minardi']` → _Checking the telemetry on Fernando Alonso — Spanish driver, raced 2001–2024. Teams: Red Bull Racing → Renault → McLaren → Ferrari …_
- **[champion_year/n/a]** Which driver took the F1 title in 1964? → gold `['Surtees']` → _And that's the title!…_
- **[win_count/popular]** What's Sergio Pérez's career Grand Prix win tally? → gold `['6']` → _The tally board reads six — Sergio Pérez's career Grand Prix wins. Some career, that.…_
- **[win_count/popular]** What's Riccardo Patrese's career Grand Prix win tally? → gold `['6']` → _The tally board reads six — Riccardo Patrese's career Grand Prix wins. Some career, that.…_