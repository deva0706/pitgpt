# PitGPT Grounding Benchmark — `v3`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 16 | 80% | 0 |
| grounded_entity | 40 | 38 | 95% | 2 |
| missing_context | 20 | 14 | 70% | 6 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 7 | 47% | 0 |
| **overall** | 105 | 85 | 81% | 8 |

**Hallucination rate:** 7.6%

## Sample failures

- **[grounded_entity]** Which team does Lewis Hamilton drive for? → gold `RB` → _Checking the telemetry on Lewis Hamilton — he's at Red Bull Racing.…_
- **[grounded_entity]** Which team does Max Verstappen drive for? → gold `RB` → _Checking the telemetry on Max Verstappen — he's at Red Bull Racing.…_
- **[grounded_descriptive]** When was DRS introduced? → gold `2011` → _It got there in '21! That's the launch date covered by the explanation. Classic F1 detail, this one. Want me to explain what DRS actually *i…_
- **[grounded_descriptive]** Quick one — when did F1 switch to turbo-hybrid engines? → gold `2014` → _Lights out and here we go! That's the hybrid power unit era — turbocharged mid-engine hybrids, generating their electrons over a few seconds…_
- **[grounded_descriptive]** Quick one — in which year did Ayrton Senna die? → gold `1994` → _2014 — that one. Not exactly news to everyone, this one.…_
- **[grounded_descriptive]** Quick one — which team was Kimi Raikkonen with when he won his title? → gold `Ferrari` → _And that's the title!…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Stack them up: Senna takes it — three championships to two.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Stack them up: Senna takes it — three championships to two.…_