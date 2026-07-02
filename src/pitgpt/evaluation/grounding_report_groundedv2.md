# PitGPT Grounding Benchmark — `groundedv2`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 20 | 100% | 0 |
| grounded_entity | 40 | 40 | 100% | 0 |
| missing_context | 20 | 17 | 85% | 3 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 12 | 80% | 0 |
| **overall** | 105 | 99 | 94% | 3 |

**Hallucination rate:** 2.9%

## Sample failures

- **[synthesis]** Which of these two circuits has higher top speeds? → gold `Monza` → _Monaco is a slow, tight street circuit where overtaking is nearly impossible. Mega question.…_
- **[synthesis]** Which of these two circuits has higher top speeds? → gold `Monza` → _Monaco is a slow, tight street circuit where overtaking is nearly impossible. Mega question.…_
- **[synthesis]** Which of these two circuits has higher top speeds? → gold `Monza` → _Monaco is a slow, tight street circuit where overtaking is nearly impossible. Mega question.…_
- **[missing_context]** Which team does Alexander Albon drive for? → gold `` → _The facts I've got only cover Charles Leclerc (who's at Aston Martin) — they don't say anything about Alexander Albon, so I won't guess.…_
- **[missing_context]** Which team does George Russell drive for? → gold `` → _The facts I've got only cover Lando Norris (who's at Ferrari) — they don't say anything about George Russell, so I won't guess.…_
- **[missing_context]** Which team does Charles Leclerc drive for? → gold `` → _The facts I've got only cover Alexander Albon (who's at RB). Love that you asked!…_