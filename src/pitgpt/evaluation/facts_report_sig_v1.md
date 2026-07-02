# PitGPT Closed-Book Facts Benchmark — `sig_v1`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 30% | 100% |
| driver_teams | 30 | 0% | 100% |
| gp_winner | 20 | 45% | 100% |
| win_count | 10 | 20% | 80% |
| **overall** | 90 | 22% | 98% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 10% |
| mid | 10 | 0% |
| longtail | 10 | 0% |

## Sample failures

- **[driver_teams/popular]** What teams did Fernando Alonso race for over his F1 career? → gold `['Renault', 'McLaren', 'Minardi']` → _🏎️ Fernando Alonso — team history: Renault → McLaren-Alfa Romeo → Ferrari → Alpine F1 Team.…_
- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _🏎️ Lewis Hamilton — team history: McLaren → Mercedes → Ferrari → Alpine F1 Team.…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _🏎️ Kimi Räikkönen — team history: Minardi → Sauber → Lotus → Renault → McLaren → Ferrari → Alfa Romeo → Caterham.…_
- **[driver_teams/popular]** What teams did Rubens Barrichello race for over his F1 career? → gold `['Honda', 'Ferrari', 'Stewart']` → _🏎️ Rubens Barrichello — team history: Minardi → Ferrari → Brawn → Honda → BAR.…_
- **[driver_teams/popular]** What teams did Jenson Button race for over his F1 career? → gold `['Honda', 'BAR', 'Renault']` → _🏎️ Jenson Button — team history: Williams → BAR → Honda → Renault → Brawn → BMW Sauber → Lotus → McLaren.…_
- **[driver_teams/popular]** What teams did Michael Schumacher race for over his F1 career? → gold `['Ferrari', 'Benetton', 'Jordan']` → _🏎️ Michael Schumacher — team history: Benetton → Ferrari → Williams → Mercedes.…_
- **[driver_teams/popular]** What teams did Sebastian Vettel race for over his F1 career? → gold `['Toro Rosso', 'BMW Sauber', 'Red Bull']` → _🏎️ Sebastian Vettel — team history: Red Bull → BMW Sauber → Toro Rosso → Ferrari → Aston Martin → Mercedes.…_
- **[driver_teams/popular]** What teams did Sergio Pérez race for over his F1 career? → gold `['Sauber', 'McLaren', 'Force India']` → _🏎️ Sergio Pérez — team history: Sauber → Force India → Racing Point → Red Bull.…_