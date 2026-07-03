# PitGPT Grounding Benchmark — `postgate_v3`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 16 | 80% | 0 |
| grounded_entity | 40 | 40 | 100% | 0 |
| missing_context | 20 | 1 | 5% | 19 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 9 | 60% | 0 |
| **overall** | 105 | 76 | 72% | 19 |

**Hallucination rate:** 18.1%

## Sample failures

- **[grounded_descriptive]** Quick one — how many points does a race winner score? → gold `25` → _A lights out pays ten points, which is double what a normal pit visit earns. So winning a race gives twice as much track time. Love that you…_
- **[grounded_descriptive]** Which championship did Nico Rosberg win? → gold `2016` → _Going by the facts here, Nico Rosberg took that one title.…_
- **[grounded_descriptive]** Quick one — which championship did Nico Rosberg win? → gold `2016` → _Going by the facts here, Nico Rosberg took that title years before I was even in orbit.…_
- **[grounded_descriptive]** Which team was Kimi Raikkonen with when he won his title? → gold `Ferrari` → _Season 2007 belonged to Kimi Raikkonen — World Champion.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Going by the facts here, Juan Manuel Fangio — five championships. Seven is a nice little different.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Going by the facts here, Juan Manuel Fangio — five championships. Mega question.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Going by the facts here, Ayrton Senna — three championships to four. Always fun to talk about.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Going by the facts here, Ayrton Senna — three championships to four. Always fun to talk about.…_