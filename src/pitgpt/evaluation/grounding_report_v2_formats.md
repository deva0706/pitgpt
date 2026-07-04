# PitGPT Grounding Benchmark — `v2_formats`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 16 | 80% | 0 |
| grounded_entity | 40 | 40 | 100% | 0 |
| missing_context | 20 | 6 | 30% | 14 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 10 | 67% | 0 |
| **overall** | 105 | 82 | 78% | 14 |

**Hallucination rate:** 13.3%

## Sample failures

- **[grounded_descriptive]** Quick one — when was DRS introduced? → gold `2011` → _Let's talk racecraft: DRS versus slipstream grooming — which is better for Sunday! DRS gives you straight line speed at your car's top witho…_
- **[grounded_descriptive]** Quick one — how many points does a race winner score? → gold `25` → _From this set of facts nothing can be argued. Mega question.…_
- **[grounded_descriptive]** Quick one — when did ground effect return to F1? → gold `2022` → _Let's talk racecraft: what makes a great overtake even better than just passing?…_
- **[grounded_descriptive]** Which team was Kimi Raikkonen with when he won his title? → gold `Ferrari` → _And that's the title! Kimi Raikkonen is the 2007 World Champion — a championship to remember.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Going by the facts here, Juan Manuel Fangio — five championships. Mega question.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Going by the facts here, Juan Manuel Fangio — five championships against seven. Classic F1 detail, this one.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Going by the facts here, Juan Manuel Fangio — five championships against seven. Mega question.…_
- **[synthesis]** Which compound lasts longer, soft or hard? → gold `hard` → _From the facts given: the soft tyre lasts longer but has the least grip. Classic F1 detail, this one.…_