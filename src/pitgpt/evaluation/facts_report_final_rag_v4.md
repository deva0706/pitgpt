# PitGPT Closed-Book Facts Benchmark — `final_rag_v4`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 100% | 100% |
| driver_teams | 30 | 67% | 0% |
| gp_winner | 20 | 75% | 100% |
| win_count | 10 | 50% | 0% |
| **overall** | 90 | 78% | 56% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 30% |
| mid | 10 | 100% |
| longtail | 10 | 90% |

## Sample failures

- **[driver_teams/popular]** What teams did Fernando Alonso race for over his F1 career? → gold `['Renault', 'McLaren', 'Minardi']` → _Looking at what I've got here: Fernando Alonso won two World Championships, in 2005 and 2006, both with Renault. That same Fernand…_
- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _Lewis Hamilton raced for Ferrari.…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _Kimi Räikkönen won 21 Grand Prix.…_
- **[driver_teams/popular]** What teams did Jenson Button race for over his F1 career? → gold `['Honda', 'BAR', 'Renault']` → _Jenson Button raced for Williams from 2000 to 2017.…_
- **[driver_teams/popular]** What teams did Michael Schumacher race for over his F1 career? → gold `['Ferrari', 'Benetton', 'Jordan']` → _Loving this part! Over nine years at Benetton, Schumacher won 91 Grand Prix titles. Nine open-wheel Grand Prix victories. The 1994…_
- **[driver_teams/popular]** What teams did Sebastian Vettel race for over his F1 career? → gold `['Toro Rosso', 'BMW Sauber', 'Red Bull']` → _I can definitely talk about that! From 2007 to 2022, Sebastian Vettel raced for Red Bull Racing, then moved up to become part of t…_
- **[driver_teams/popular]** What teams did Sergio Pérez race for over his F1 career? → gold `['Sauber', 'McLaren', 'Force India']` → _Looking at what I've got: Sergio Pérez raced for Sauber from 2011 to 2024, winning 6 Grand Prix titles.…_
- **[driver_teams/popular]** What teams did Felipe Massa race for over his F1 career? → gold `['Ferrari', 'Sauber', 'Williams']` → _Felipe Massa won 11 Grand Prix.…_