# PitGPT Closed-Book Facts Benchmark — `oct11_pipeline`

Cases: 90 · held-out phrasings · gold from Kaggle tables

| Category | Cases | Accuracy | Signature adherence |
|---|---|---|---|
| champion_year | 30 | 100% | 100% |
| driver_teams | 30 | 100% | 100% |
| gp_winner | 20 | 100% | 100% |
| win_count | 10 | 100% | 100% |
| **overall** | 90 | 100% | 100% |

## Memorization gradient (driver questions by popularity)

| Tier | Cases | Accuracy |
|---|---|---|
| popular | 20 | 100% |
| mid | 10 | 100% |
| longtail | 10 | 100% |

## The model's own replies

The table above scores what the app shows. Each reply is checked against its fact first (`inference/answers.py`) and replaced by the fact's own rendering if it does not hold.

- model's reply kept as written: 76/90
- model's replies scored as written, before the check: 89/90 (99%)
- replaced (not the reply the fact dictates): What teams did Fernando Alonso race for over his F1 career? → _Checking the telemetry on Fernando Alonso — Spanish driver, raced 2001–2026. Teams: Minardi → Renault → McLare_
- replaced (not the reply the fact dictates): What teams did Kimi Räikkönen race for over his F1 career? → _Checking the telemetry on Kimi Räikkönen — Finnish driver, raced 2001–2021. Teams: Sauber → McLaren → Ferrari _
- replaced (not the reply the fact dictates): What teams did Sergio Pérez race for over his F1 career? → _Checking the telemetry on Sergio Pérez — Mexican driver, raced 2011–2026. Teams: Sauber → McLaren → Force Indi_
- replaced (not the reply the fact dictates): What teams did Peter Gethin race for over his F1 career? → _Checking the telemetry on Peter Gethin — British driver, raced 1970–1974. Teams: McLaren → Ford → BMR → McLare_
- replaced (not the reply the fact dictates): What teams did Eric van de Poele race for over his F1 career? → _Checking the telemetry on Eric van de Poele — Belgian driver, raced 1991–1992. Teams: Lamborghini → Brabham → _
- replaced (not the reply the fact dictates): What teams did Piers Courage race for over his F1 career? → _Checking the telemetry on Piers Courage — British driver, raced 1967–1970. Teams: BRM → Brabham-Ford → De Toma_
- replaced (not the reply the fact dictates): What teams did Enrico Bertaggia race for over his F1 career? → _Checking the telemetry on Enrico Bertaggia — Italian driver, raced 1989–1990. Teams: Coloni._
- replaced (not the reply the fact dictates): What teams did Roelof Wunderink race for over his F1 career? → _Checking the telemetry on Roelof Wunderink — Dutch driver, raced 1975–1976. Teams: Ensign._

## Sample failures
