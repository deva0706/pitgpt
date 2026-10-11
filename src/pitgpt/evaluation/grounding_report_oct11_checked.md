# PitGPT Grounding Benchmark — `oct11_checked`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 20 | 100% | 0 |
| grounded_entity | 40 | 40 | 100% | 0 |
| missing_context | 20 | 20 | 100% | 0 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 0 | 0% | 0 |
| **overall** | 105 | 90 | 86% | 0 |

**Hallucination rate:** 0.0%

## Sample failures

- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Box, box — that one's not in my notes, so I won't guess.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Box, box — that one's not in my notes, so I won't guess.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Box, box — that one's not in my notes, so I won't guess.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Box, box — that one's not in my notes, so I won't guess.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Box, box — that one's not in my notes, so I won't guess.…_
- **[synthesis]** Who won more titles, Vettel or Senna? → gold `Vettel` → _Box, box — that one's not in my notes, so I won't guess.…_
- **[synthesis]** Which of these two circuits has higher top speeds? → gold `Monza` → _Box, box — that one's not in my notes, so I won't guess.…_
- **[synthesis]** Which of these two circuits has higher top speeds? → gold `Monza` → _Box, box — that one's not in my notes, so I won't guess.…_