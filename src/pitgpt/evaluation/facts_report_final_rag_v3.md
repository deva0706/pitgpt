# PitGPT Closed-Book Facts Benchmark — `final_rag_v3`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 93% | 80% |
| driver_teams | 30 | 53% | 0% |
| gp_winner | 20 | 45% | 100% |
| win_count | 10 | 40% | 0% |
| **overall** | 90 | 63% | 49% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 35% |
| mid | 10 | 90% |
| longtail | 10 | 40% |

## Sample failures

- **[driver_teams/popular]** What teams did Fernando Alonso race for over his F1 career? → gold `['Renault', 'McLaren', 'Minardi']` → _Looking at what I've got here: Fernando Alonso raced for McLaren. Two World Championships there. Good spot!…_
- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _Lewis Hamilton raced for Ferrari. Mega question.…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _Kimi Räikkönen won the 2model
Love that you asked! I'd love to hear what you think about this detail.…_
- **[driver_teams/popular]** What teams did Jenson Button race for over his F1 career? → gold `['Honda', 'BAR', 'Renault']` → _Jenson Button raced for McLaren from 2…_
- **[driver_teams/popular]** What teams did Michael Schumacher race for over his F1 career? → gold `['Ferrari', 'Benetton', 'Jordan']` → _Going by the facts here: Michael Schumacher raced for McLaren. Mega question.…_
- **[driver_teams/popular]** What teams did Sebastian Vettel race for over his F1 career? → gold `['Toro Rosso', 'BMW Sauber', 'Red Bull']` → _Looking at what I've got here: Sebastian Vettel raced for Red Bull Racing (from 2007–2022), then moved onto Haas F1 Team (2022–202…_
- **[driver_teams/popular]** What teams did Riccardo Patrese race for over his F1 career? → gold `['Benetton', 'Williams', 'Brabham']` → _Looking at what I've got: Riccardo Patrace has a full career here, winning 6 Grand Prix.…_
- **[driver_teams/mid]** What teams did Peter Gethin race for over his F1 career? → gold `['Lola', 'BRM', 'McLaren']` → _Peter Gethin won 3 Grands Prix.…_