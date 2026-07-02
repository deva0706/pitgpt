# PitGPT Closed-Book Facts Benchmark — `final_rag_v2`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 83% | 77% |
| driver_teams | 30 | 23% | 0% |
| gp_winner | 20 | 45% | 100% |
| win_count | 10 | 50% | 0% |
| **overall** | 90 | 51% | 48% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 30% |
| mid | 10 | 40% |
| longtail | 10 | 20% |

## Sample failures

- **[driver_teams/popular]** What teams did Fernando Alonso race for over his F1 career? → gold `['Renault', 'McLaren', 'Minardi']` → _Looking at what I've got: Fernando Alonso raced for McLaren. Two World Championships there. Mega ask.…_
- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _Lewis Hamilton raced for Ferrari.…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _Kimi Räikkönen won the 2model
Love that you asked!…_
- **[driver_teams/popular]** What teams did Rubens Barrichello race for over his F1 career? → gold `['Honda', 'Ferrari', 'Stewart']` → _Looking at what I've got: Rubens Barrichello raced for Jordan, Stewart, Ferrari, Honda, Brawn, Haas. Love that you asked!…_
- **[driver_teams/popular]** What teams did Jenson Button race for over his F1 career? → gold `['Honda', 'BAR', 'Renault']` → _Jenson Button raced for McLaren from 2…_
- **[driver_teams/popular]** What teams did Michael Schumacher race for over his F1 career? → gold `['Ferrari', 'Benetton', 'Jordan']` → _Looking at what I've got here: Michael Schumacher raced for McLaren from 1993 to 2022. Mega ask.…_
- **[driver_teams/popular]** What teams did Sebastian Vettel race for over his F1 career? → gold `['Toro Rosso', 'BMW Sauber', 'Red Bull']` → _Looking at what I've got here: Sebastian Vettel had a stellar nine-year arc as a racing machine, joining Haas in 2009 then moving …_
- **[driver_teams/popular]** What teams did Sergio Pérez race for over his F1 career? → gold `['Sauber', 'McLaren', 'Force India']` → _Looking at what I've got here: Sergio Pérez raced for Sauber, McLaren, Force India, Racing Point, Red Bull. Mega question.…_