# PitGPT Closed-Book Facts Benchmark — `final_rag`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 57% | 83% |
| driver_teams | 30 | 33% | 0% |
| gp_winner | 20 | 45% | 100% |
| win_count | 10 | 0% | 0% |
| **overall** | 90 | 40% | 50% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 5% |
| mid | 10 | 50% |
| longtail | 10 | 40% |

## Sample failures

- **[driver_teams/popular]** What teams did Fernando Alonso race for over his F1 career? → gold `['Renault', 'McLaren', 'Minardi']` → _Looking at what I've got here: Fernando Alonso had a bit more rings under his belt overall. He won the 2005 Formula 1 World Champi…_
- **[driver_teams/popular]** What teams did Lewis Hamilton race for over his F1 career? → gold `['McLaren', 'Mercedes']` → _I can't give you a single definitive list because I see multiple bits of the puzzle here! But roughly speaking: Lewis Hamilton had…_
- **[driver_teams/popular]** What teams did Kimi Räikkönen race for over his F1 career? → gold `['Ferrari', 'McLaren', 'Sauber']` → _From the facts I've got here, Kimi Räikkönen raced for Sauber, McLaren, Ferrari, Lotus F1, Alfa Romeo. Love that you asked!…_
- **[driver_teams/popular]** What teams did Rubens Barrichello race for over his F1 career? → gold `['Honda', 'Ferrari', 'Stewart']` → _Looking at what I've got: Rubens Barrichello raced for Jordan, Stewart, Ferrari, Honda, Brawn. Love that you asked!…_
- **[driver_teams/popular]** What teams did Jenson Button race for over his F1 career? → gold `['Honda', 'BAR', 'Renault']` → _Jenson Button raced for Williams, Benetton, Renault, BAR, Honda, Brawn, McLaren. Mega question.…_
- **[driver_teams/popular]** What teams did Michael Schumacher race for over his F1 career? → gold `['Ferrari', 'Benetton', 'Jordan']` → _That specific detail isn't in what I was given, so I'll be straight with you: I don't know from here. Give me the up-to-date info …_
- **[driver_teams/popular]** What teams did Sebastian Vettel race for over his F1 career? → gold `['Toro Rosso', 'BMW Sauber', 'Red Bull']` → _Honestly, I don't have that in the notes directly handed down from a team — my expertise is *F1*, not *a single team*. But if you …_
- **[driver_teams/popular]** What teams did Sergio Pérez race for over his F1 career? → gold `['Sauber', 'McLaren', 'Force India']` → _Looking at what I've got here: Sergio Pérez raced for Sauber, McLaren, Force India, Racing Point, Red Bull. Mega question.…_