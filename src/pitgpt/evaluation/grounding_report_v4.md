# PitGPT Grounding Benchmark — `v4`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 18 | 90% | 0 |
| grounded_entity | 40 | 40 | 100% | 0 |
| missing_context | 20 | 4 | 20% | 16 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 4 | 27% | 0 |
| **overall** | 105 | 76 | 72% | 16 |

**Hallucination rate:** 15.2%

## Sample failures

- **[grounded_descriptive]** Quick one — when did F1 switch to turbo-hybrid engines? → gold `2014` → _Let's talk race engineering! Turbo-hybrid power is new — every driver has a little extra grunt they can use through the lap, which opens up …_
- **[grounded_descriptive]** Quick one — which team was Kimi Raikkonen with when he won his title? → gold `Ferrari` → _And that's the point — quick one because we can talk about it endlessly! Winning a championship is as much a package deal as a single win. I…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Stack them up: Juan Manuel Fangio takes it — five championships to four.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Stack them up: Max Verstappen takes it — seven championships to six.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Stack them up: Lewis Hamilton takes it — seven championships to five.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Stack them up: Lewis Hamilton takes it — he wins five championships to Senna's four.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Stack them up: Lewis Hamilton takes it — he wins five championships to Senna's four.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Stack them up: Lewis Hamilton takes it — he wins five championships to Senna's four.…_