# PitGPT Closed-Book Facts Benchmark — `final_rag_v2`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 83% | 77% |
| driver_teams | 30 | 47% | 0% |
| gp_winner | 20 | 45% | 100% |
| win_count | 10 | 50% | 0% |
| **overall** | 90 | 59% | 48% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 45% |
| mid | 10 | 80% |
| longtail | 10 | 20% |

## Sample failures

- **[driver_teams/popular]** What teams did Fernando Alonso race for over his F1 career? → gold `['Renault', 'McLaren', 'Minardi']` → _Looking at what I've got: Fernando Alonso raced for McLaren. Two World Championships there. Mega ask.…_
- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _Lewis Hamilton raced for Ferrari.…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _Kimi Räikkönen won the 2model
Love that you asked!…_
- **[driver_teams/popular]** What teams did Jenson Button race for over his F1 career? → gold `['Honda', 'BAR', 'Renault']` → _Jenson Button raced for McLaren from 2…_
- **[driver_teams/popular]** What teams did Michael Schumacher race for over his F1 career? → gold `['Ferrari', 'Benetton', 'Jordan']` → _Looking at what I've got here: Michael Schumacher raced for McLaren from 1993 to 2022. Mega ask.…_
- **[driver_teams/popular]** What teams did Riccardo Patrese race for over his F1 career? → gold `['Benetton', 'Williams', 'Brabham']` → _Looking at what I've got: Riccardo Patrace has a full record here, with 6 Grand Prix wins.…_
- **[driver_teams/mid]** What teams did Peter Gethin race for over his F1 career? → gold `['Lola', 'BRM', 'McLaren']` → _Peter Gethin won 3 Grands Prix. Love that you asked!…_
- **[driver_teams/mid]** What teams did Piers Courage race for over his F1 career? → gold `['De Tomaso', 'Brabham-Ford', 'BRM']` → _Piers Courage raced for BRM, Lotus-BRM, Brabham-Ford, De Tomaso. One of my favourite bits of history.…_